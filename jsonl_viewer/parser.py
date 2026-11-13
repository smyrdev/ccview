"""Parser for Claude Code JSONL session logs."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Any

from .models import RawNode, Session, TokenUsage, ToolCall, Turn


def parse_timestamp(ts_str: str | None) -> datetime | None:
    """Parse ISO-8601 timestamp string into timezone-aware datetime."""
    if not ts_str or not isinstance(ts_str, str):
        return None
    try:
        if ts_str.endswith("Z"):
            ts_str = ts_str[:-1] + "+00:00"
        return datetime.fromisoformat(ts_str)
    except ValueError:
        return None


def decode_project_dir_name(dir_name: str, cwd: str = "") -> str:
    """Produce a human-friendly project name from cwd or encoded directory name."""
    if cwd:
        p = Path(cwd)
        if len(p.parts) >= 2:
            return f"{p.parent.name}/{p.name}"
        return p.name or cwd
    if dir_name.startswith("-"):
        # Claude Code encodes '/home/samy/Work/my-app' as '-home-samy-Work-my-app'
        parts = [p for p in dir_name.split("-") if p]
        if len(parts) >= 2:
            return "/".join(parts[-2:])
        if parts:
            return parts[-1]
    return dir_name


def _extract_tool_result_text(content: Any) -> str:
    """Normalize tool_result content (string or list of text/image blocks) into text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        pieces: list[str] = []
        for item in content:
            if isinstance(item, str):
                pieces.append(item)
            elif isinstance(item, dict):
                btype = item.get("type", "")
                if btype == "text":
                    pieces.append(str(item.get("text", "")))
                elif btype == "image":
                    pieces.append("[Image attachment]")
                else:
                    pieces.append(json.dumps(item, ensure_ascii=False))
        return "\n".join(pieces)
    if isinstance(content, dict):
        return json.dumps(content, indent=2, ensure_ascii=False)
    return str(content)


def _extract_usage(usage_dict: dict[str, Any] | None, model: str = "") -> TokenUsage:
    if not usage_dict or not isinstance(usage_dict, dict):
        return TokenUsage(model=model)
    return TokenUsage(
        input_tokens=int(usage_dict.get("input_tokens") or 0),
        output_tokens=int(usage_dict.get("output_tokens") or 0),
        cache_creation_input_tokens=int(
            usage_dict.get("cache_creation_input_tokens") or 0
        ),
        cache_read_input_tokens=int(usage_dict.get("cache_read_input_tokens") or 0),
        model=model,
    )


def parse_session(path: Path | str) -> Session:
    """Parse a Claude Code .jsonl file into a structured Session."""
    file_path = Path(path).expanduser().resolve()
    session_id = file_path.stem
    parent_dir_name = file_path.parent.name

    session = Session(
        path=file_path,
        session_id=session_id,
        project_name=decode_project_dir_name(parent_dir_name),
    )

    tool_calls_by_id: dict[str, ToolCall] = {}
    turns_by_msg_id: dict[str, Turn] = {}
    usage_by_msg_id: dict[str, TokenUsage] = {}

    if not file_path.exists():
        return session

    with file_path.open("r", encoding="utf-8", errors="replace") as f:
        for line_no, raw_line in enumerate(f, start=1):
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            session.line_count += 1

            try:
                data = json.loads(raw_line)
            except json.JSONDecodeError:
                session.parse_errors += 1
                continue

            if not isinstance(data, dict):
                continue

            entry_type = str(data.get("type") or "")
            uuid = str(data.get("uuid") or f"line-{line_no}")
            parent_uuid = data.get("parentUuid")
            is_sidechain = bool(data.get("isSidechain", False))
            agent_id = data.get("agentId") or data.get("slug")
            ts = parse_timestamp(data.get("timestamp"))

            if ts:
                if session.start_time is None or ts < session.start_time:
                    session.start_time = ts
                if session.end_time is None or ts > session.end_time:
                    session.end_time = ts

            if not session.cwd and data.get("cwd"):
                session.cwd = str(data["cwd"])
                session.project_name = decode_project_dir_name(
                    parent_dir_name, session.cwd
                )
            if not session.git_branch and data.get("gitBranch"):
                session.git_branch = str(data["gitBranch"])
            if not session.version and data.get("version"):
                session.version = str(data["version"])
            if data.get("sessionId") and session.session_id == file_path.stem:
                session.session_id = str(data["sessionId"])

            # 1. Summary entry
            if entry_type == "summary":
                summary_text = str(data.get("summary") or "").strip()
                if summary_text:
                    session.summaries.append(summary_text)
                session.raw_nodes.append(
                    RawNode(
                        line_no=line_no,
                        uuid=uuid,
                        parent_uuid=parent_uuid,
                        node_type="summary",
                        timestamp=ts,
                        label=summary_text[:80],
                    )
                )
                continue

            # 2. System entry
            if entry_type == "system":
                content_str = str(
                    data.get("content")
                    or data.get("message", {}).get("content", "")
                    if isinstance(data.get("message"), dict)
                    else data.get("content") or ""
                ).strip()
                if content_str:
                    turn = Turn(
                        index=len(session.turns) + 1,
                        role="system",
                        uuid=uuid,
                        parent_uuid=parent_uuid,
                        timestamp=ts,
                        is_sidechain=is_sidechain,
                        agent_id=agent_id,
                        text_blocks=[content_str],
                        line_numbers=[line_no],
                    )
                    session.turns.append(turn)
                session.raw_nodes.append(
                    RawNode(
                        line_no=line_no,
                        uuid=uuid,
                        parent_uuid=parent_uuid,
                        node_type="system",
                        timestamp=ts,
                        is_sidechain=is_sidechain,
                        agent_id=agent_id,
                        label=content_str[:80] or "system event",
                    )
                )
                continue

            msg = data.get("message")
            if not isinstance(msg, dict):
                continue

            # 3. User entry (either human prompt OR tool_result carrier)
            if entry_type == "user" or msg.get("role") == "user":
                content = msg.get("content")
                extra_meta = (
                    data.get("toolUseResult")
                    if isinstance(data.get("toolUseResult"), dict)
                    else {}
                )
                user_texts: list[str] = []
                tool_results_count = 0

                if isinstance(content, str):
                    if content.strip():
                        user_texts.append(content.strip())
                elif isinstance(content, list):
                    for block in content:
                        if isinstance(block, str) and block.strip():
                            user_texts.append(block.strip())
                        elif isinstance(block, dict):
                            btype = block.get("type")
                            if btype == "text":
                                txt = str(block.get("text") or "").strip()
                                if txt:
                                    user_texts.append(txt)
                            elif btype == "tool_result":
                                tool_results_count += 1
                                tool_use_id = str(block.get("tool_use_id") or "")
                                is_err = bool(block.get("is_error", False))
                                res_text = _extract_tool_result_text(
                                    block.get("content")
                                )
                                if tool_use_id in tool_calls_by_id:
                                    tc = tool_calls_by_id[tool_use_id]
                                    tc.has_result = True
                                    tc.is_error = is_err
                                    tc.result_content = res_text
                                    tc.result_timestamp = ts
                                    if extra_meta:
                                        tc.extra_result_meta = extra_meta

                is_tool_only = tool_results_count > 0 and not user_texts
                label_preview = (
                    user_texts[0].splitlines()[0][:80]
                    if user_texts
                    else f"[tool_result x{tool_results_count}]"
                )
                session.raw_nodes.append(
                    RawNode(
                        line_no=line_no,
                        uuid=uuid,
                        parent_uuid=parent_uuid,
                        node_type="user",
                        timestamp=ts,
                        is_sidechain=is_sidechain,
                        agent_id=agent_id,
                        label=label_preview,
                        is_tool_result_only=is_tool_only,
                    )
                )

                if user_texts:
                    turn = Turn(
                        index=len(session.turns) + 1,
                        role="user",
                        uuid=uuid,
                        parent_uuid=parent_uuid,
                        timestamp=ts,
                        end_timestamp=ts,
                        is_sidechain=is_sidechain,
                        agent_id=agent_id,
                        text_blocks=user_texts,
                        line_numbers=[line_no],
                    )
                    session.turns.append(turn)
                continue

            # 4. Assistant entry
            if entry_type == "assistant" or msg.get("role") == "assistant":
                msg_id = str(msg.get("id") or uuid)
                model = str(msg.get("model") or "")
                if model and not session.primary_model and model != "<synthetic>":
                    session.primary_model = model

                usage = _extract_usage(msg.get("usage"), model=model)
                # Claude Code often emits multiple lines with the same message.id
                # as content blocks stream in; keep the highest token counts per msg_id.
                if msg_id in usage_by_msg_id:
                    prev_u = usage_by_msg_id[msg_id]
                    prev_u.input_tokens = max(prev_u.input_tokens, usage.input_tokens)
                    prev_u.output_tokens = max(
                        prev_u.output_tokens, usage.output_tokens
                    )
                    prev_u.cache_creation_input_tokens = max(
                        prev_u.cache_creation_input_tokens,
                        usage.cache_creation_input_tokens,
                    )
                    prev_u.cache_read_input_tokens = max(
                        prev_u.cache_read_input_tokens,
                        usage.cache_read_input_tokens,
                    )
                else:
                    usage_by_msg_id[msg_id] = TokenUsage(
                        input_tokens=usage.input_tokens,
                        output_tokens=usage.output_tokens,
                        cache_creation_input_tokens=usage.cache_creation_input_tokens,
                        cache_read_input_tokens=usage.cache_read_input_tokens,
                        model=model,
                    )

                if msg_id in turns_by_msg_id:
                    turn = turns_by_msg_id[msg_id]
                    turn.line_numbers.append(line_no)
                    if ts:
                        turn.end_timestamp = ts
                    turn.usage = usage_by_msg_id[msg_id]
                else:
                    turn = Turn(
                        index=len(session.turns) + 1,
                        role="assistant",
                        uuid=uuid,
                        parent_uuid=parent_uuid,
                        message_id=msg_id,
                        timestamp=ts,
                        end_timestamp=ts,
                        is_sidechain=is_sidechain,
                        agent_id=agent_id,
                        model=model,
                        usage=usage_by_msg_id[msg_id],
                        line_numbers=[line_no],
                    )
                    turns_by_msg_id[msg_id] = turn
                    session.turns.append(turn)

                content = msg.get("content")
                node_tools: list[str] = []
                node_has_thinking = False
                node_text_preview = ""

                if isinstance(content, str) and content.strip():
                    turn.text_blocks.append(content.strip())
                    node_text_preview = content.strip().splitlines()[0][:80]
                elif isinstance(content, list):
                    for block in content:
                        if not isinstance(block, dict):
                            continue
                        btype = block.get("type")
                        if btype == "thinking":
                            thinking_txt = str(block.get("thinking") or "").strip()
                            if thinking_txt:
                                turn.thinking_blocks.append(thinking_txt)
                                node_has_thinking = True
                        elif btype == "text":
                            txt = str(block.get("text") or "").strip()
                            if txt:
                                turn.text_blocks.append(txt)
                                if not node_text_preview:
                                    node_text_preview = txt.splitlines()[0][:80]
                        elif btype == "tool_use":
                            tc_id = str(
                                block.get("id")
                                or f"tool-{line_no}-{len(turn.tool_calls)}"
                            )
                            tc_name = str(block.get("name") or "unknown_tool")
                            tc_input = (
                                block.get("input")
                                if isinstance(block.get("input"), dict)
                                else {}
                            )
                            tc = ToolCall(
                                id=tc_id,
                                name=tc_name,
                                input=tc_input,
                                timestamp=ts,
                                is_sidechain=is_sidechain,
                            )
                            turn.tool_calls.append(tc)
                            tool_calls_by_id[tc_id] = tc
                            node_tools.append(tc_name)

                if not node_text_preview:
                    if node_tools:
                        node_text_preview = f"Tools: {', '.join(node_tools)}"
                    elif node_has_thinking:
                        node_text_preview = "(thinking)"
                    else:
                        node_text_preview = "assistant turn"

                session.raw_nodes.append(
                    RawNode(
                        line_no=line_no,
                        uuid=uuid,
                        parent_uuid=parent_uuid,
                        node_type="assistant",
                        timestamp=ts,
                        is_sidechain=is_sidechain,
                        agent_id=agent_id,
                        label=node_text_preview,
                        tool_names=node_tools,
                        has_thinking=node_has_thinking,
                        usage=usage,
                    )
                )

    # Aggregate deduplicated session token usage and cost
    total_usage = TokenUsage(model=session.primary_model)
    total_cost = 0.0
    for u in usage_by_msg_id.values():
        total_usage.add(u)
        total_cost += u.estimated_cost_usd(default_model=session.primary_model)

    session.total_usage = total_usage
    session.total_cost_usd = total_cost

    return session
