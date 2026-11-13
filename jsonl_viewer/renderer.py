"""Rich terminal renderers for Claude Code JSONL sessions."""

from __future__ import annotations

from collections import defaultdict
import difflib
from pathlib import Path
from typing import Iterable

from rich import box
from rich.console import Console, Group, RenderableType
from rich.markdown import Markdown
from rich.markup import escape
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

from .discovery import SearchHit
from .models import RawNode, Session, ToolCall, Turn
from .parser import parse_session


def fmt_tokens(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}k"
    return str(n)


def fmt_duration(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    if seconds < 1.0:
        return f"{seconds * 1000:.0f}ms"
    if seconds < 60.0:
        return f"{seconds:.1f}s"
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    if mins < 60:
        return f"{mins}m {secs:02d}s"
    hours = mins // 60
    mins = mins % 60
    return f"{hours}h {mins:02d}m"


def fmt_ms(ms: float | None) -> str:
    if ms is None:
        return "-"
    return fmt_duration(ms / 1000.0)


def _summarize_tool_input(tc: ToolCall) -> str:
    """Create a concise one-line summary of a tool call's primary argument."""
    inp = tc.input
    if not inp:
        return ""
    for key in (
        "command",
        "file_path",
        "path",
        "pattern",
        "query",
        "url",
        "description",
        "prompt",
    ):
        if key in inp and inp[key]:
            val = str(inp[key]).strip().splitlines()[0]
            return val[:90] + ("…" if len(val) > 90 else "")
    # Fallback to key=value preview
    pairs = [f"{k}={repr(v)[:30]}" for k, v in list(inp.items())[:2]]
    return ", ".join(pairs)


def _compute_tool_diff(tc: ToolCall) -> str | None:
    """Generate a unified diff string if the tool call is an Edit or MultiEdit."""
    name_low = tc.name.lower()
    inp = tc.input
    file_path = str(inp.get("file_path") or inp.get("path") or "file")

    if name_low in ("edit", "str_replace_editor", "replace"):
        old_str = inp.get("old_string") or inp.get("old_str") or inp.get("Target")
        new_str = inp.get("new_string") or inp.get("new_str") or inp.get("Replacement")
        if isinstance(old_str, str) and isinstance(new_str, str):
            diff_lines = list(
                difflib.unified_diff(
                    old_str.splitlines(),
                    new_str.splitlines(),
                    fromfile=f"a/{Path(file_path).name}",
                    tofile=f"b/{Path(file_path).name}",
                    lineterm="",
                )
            )
            if diff_lines:
                return "\n".join(diff_lines)

    elif name_low == "multiedit" and isinstance(inp.get("edits"), list):
        all_diffs: list[str] = []
        for idx, edit in enumerate(inp["edits"], start=1):
            if not isinstance(edit, dict):
                continue
            old_str = edit.get("old_string")
            new_str = edit.get("new_string")
            if isinstance(old_str, str) and isinstance(new_str, str):
                diff_lines = list(
                    difflib.unified_diff(
                        old_str.splitlines(),
                        new_str.splitlines(),
                        fromfile=f"a/{Path(file_path).name} (edit #{idx})",
                        tofile=f"b/{Path(file_path).name} (edit #{idx})",
                        lineterm="",
                    )
                )
                all_diffs.extend(diff_lines)
        if all_diffs:
            return "\n".join(all_diffs)

    return None


def _render_tool_card(
    tc: ToolCall,
    full_output: bool = False,
    max_lines: int = 12,
) -> Panel:
    """Render a paired tool_use + tool_result card."""
    summary_arg = _summarize_tool_input(tc)
    dur_str = fmt_ms(tc.duration_ms)

    title = Text()
    if tc.is_error:
        title.append(" ✗ ", style="bold red")
    elif tc.has_result:
        title.append(" ✓ ", style="bold green")
    else:
        title.append(" • ", style="bold yellow")

    title.append(tc.name, style="bold yellow")
    if summary_arg:
        title.append(f"  {summary_arg}", style="bright_white")
    if tc.is_sidechain:
        title.append("  [subagent]", style="bold magenta")
    if dur_str != "-":
        title.append(f"  ({dur_str})", style="dim cyan")

    body_parts: list[RenderableType] = []

    # 1. For Bash, show the full command if multi-line or truncated in title
    if tc.name.lower() == "bash" and "command" in tc.input:
        cmd_str = str(tc.input["command"]).strip()
        if "\n" in cmd_str or len(cmd_str) > 85:
            body_parts.append(Syntax(cmd_str, "bash", theme="monokai", word_wrap=True))

    # 2. For Task (subagent), show description & prompt preview
    if tc.name.lower() == "task" and "prompt" in tc.input:
        prompt_str = str(tc.input["prompt"]).strip()
        if not full_output and len(prompt_str) > 240:
            prompt_str = prompt_str[:240] + "…"
        body_parts.append(Text(f"Subagent Prompt: {prompt_str}", style="italic magenta"))

    # 3. For Edit / MultiEdit, render inline unified diff
    diff_text = _compute_tool_diff(tc)
    if diff_text:
        diff_lines = diff_text.splitlines()
        if not full_output and len(diff_lines) > 30:
            diff_text = (
                "\n".join(diff_lines[:30])
                + f"\n... ({len(diff_lines) - 30} more diff lines, use --full to expand)"
            )
        body_parts.append(Syntax(diff_text, "diff", theme="monokai", word_wrap=True))

    # 4. For Write, show preview of written content
    elif tc.name.lower() == "write" and "content" in tc.input:
        content_str = str(tc.input["content"])
        lines = content_str.splitlines()
        preview = "\n".join(lines if full_output else lines[:max_lines])
        ext = Path(str(tc.input.get("file_path", "txt"))).suffix.lstrip(".") or "text"
        body_parts.append(Syntax(preview, ext, theme="monokai", word_wrap=True))
        if not full_output and len(lines) > max_lines:
            body_parts.append(
                Text(
                    f"... (+{len(lines) - max_lines} more lines written)",
                    style="dim",
                )
            )

    # 5. Tool Result output
    if tc.result_content:
        # Don't repeat redundant success boilerplate if we already showed a diff
        res_lines = tc.result_content.strip().splitlines()
        if not diff_text or tc.is_error or full_output:
            shown_lines = res_lines if full_output else res_lines[:max_lines]
            out_str = "\n".join(shown_lines)
            if not full_output and len(out_str) > 1200:
                out_str = out_str[:1200] + "…"
            style = "red" if tc.is_error else "dim white"
            body_parts.append(Text(out_str, style=style))
            if not full_output and len(res_lines) > max_lines:
                body_parts.append(
                    Text(
                        f"… (+{len(res_lines) - max_lines} more output lines, pass -f/--full to view all)",
                        style="dim italic",
                    )
                )

    if not body_parts:
        body_parts.append(Text("(completed with no text output)", style="dim"))

    border_style = "red" if tc.is_error else "dim yellow"
    return Panel(
        Group(*body_parts),
        title=title,
        title_align="left",
        border_style=border_style,
        box=box.ROUNDED,
        padding=(0, 1),
    )


def render_session_header(session: Session) -> Panel:
    """Render metadata summary banner at the top of `show` and `stats`."""
    grid = Table.grid(padding=(0, 3))
    grid.add_column(style="bold cyan")
    grid.add_column()
    grid.add_column(style="bold cyan")
    grid.add_column()

    date_str = (
        session.start_time.strftime("%Y-%m-%d %H:%M:%S")
        if session.start_time
        else "unknown"
    )
    dur_str = fmt_duration(session.duration_seconds)
    u = session.total_usage

    grid.add_row(
        "Session ID:",
        Text(session.session_id, style="bold white"),
        "Model:",
        Text(session.primary_model or "unknown", style="magenta"),
    )
    grid.add_row(
        "Project:",
        Text(session.cwd or session.project_name, style="green"),
        "Git Branch:",
        Text(session.git_branch or "-", style="yellow"),
    )
    grid.add_row(
        "Started:",
        f"{date_str} ({dur_str})",
        "Turns / Tools:",
        f"{session.user_prompt_count} prompts, {session.assistant_turn_count} replies, {len(session.all_tool_calls)} tool calls",
    )
    grid.add_row(
        "Tokens:",
        f"{fmt_tokens(u.total_tokens)} total (In: {fmt_tokens(u.input_tokens)}, Out: {fmt_tokens(u.output_tokens)}, Cache Hit: {u.cache_hit_rate:.1f}%)",
        "Est. Cost:",
        Text(f"${session.total_cost_usd:.4f}", style="bold green"),
    )
    if session.summaries:
        grid.add_row(
            "Summary:",
            Text(session.summaries[-1], style="italic bright_white"),
            "",
            "",
        )

    return Panel(
        grid,
        title=f"[bold bright_cyan]Claude Code Session • {session.project_name}[/]",
        subtitle=f"[dim]{session.path}[/]",
        border_style="bright_cyan",
        box=box.HEAVY,
        padding=(0, 1),
    )


def render_session_list(
    files: Iterable[Path],
    console: Console,
    limit: int = 25,
) -> None:
    """Render a table of discovered Claude Code sessions."""
    file_list = list(files)[:limit]
    if not file_list:
        console.print(
            "[yellow]No Claude Code .jsonl files found.[/yellow] "
            "Pass [bold]--dir <path>[/bold] or provide a path to a .jsonl file."
        )
        return

    table = Table(
        title=f"[bold cyan]Discovered Claude Code Sessions (showing {len(file_list)})[/]",
        box=box.SIMPLE_HEAVY,
        header_style="bold bright_cyan",
        show_lines=False,
    )
    table.add_column("#", justify="right", style="bold yellow", no_wrap=True)
    table.add_column("Session ID", style="cyan", no_wrap=True)
    table.add_column("Project", style="green")
    table.add_column("Branch", style="magenta", no_wrap=True)
    table.add_column("Started", style="dim", no_wrap=True)
    table.add_column("Dur", justify="right", no_wrap=True)
    table.add_column("Msgs", justify="right")
    table.add_column("Tools", justify="right")
    table.add_column("Tokens", justify="right", style="bright_white")
    table.add_column("Cost", justify="right", style="green")
    table.add_column("First Prompt / Summary", max_width=46, no_wrap=True)

    for idx, fpath in enumerate(file_list, start=1):
        s = parse_session(fpath)
        short_id = s.session_id[:10]
        started = s.start_time.strftime("%Y-%m-%d %H:%M") if s.start_time else "-"
        prompt_preview = " ".join(s.first_user_prompt.split())
        if len(prompt_preview) > 44:
            prompt_preview = prompt_preview[:43] + "…"

        table.add_row(
            str(idx),
            short_id,
            s.project_name,
            s.git_branch or "-",
            started,
            fmt_duration(s.duration_seconds),
            str(len(s.turns)),
            str(len(s.all_tool_calls)),
            fmt_tokens(s.total_usage.total_tokens),
            f"${s.total_cost_usd:.3f}",
            prompt_preview,
        )

    console.print(table)
    console.print(
        "[dim]Tip: Run [bold]ccview show 1[/bold] to replay session #1, "
        "[bold]ccview stats 1[/bold] for forensics, or [bold]ccview tree 1[/bold] for the UUID tree.[/dim]"
    )


def render_session_show(
    session: Session,
    console: Console,
    show_thinking: bool = False,
    full_output: bool = False,
    only_tools: bool = False,
    no_tools: bool = False,
    turn_filter: int | None = None,
) -> None:
    """Render the interactive/colored chat & tool timeline for a session."""
    console.print(render_session_header(session))
    console.print()

    for turn in session.turns:
        if turn_filter is not None and turn.index != turn_filter:
            continue
        if only_tools and not turn.tool_calls:
            continue

        ts_str = turn.timestamp.strftime("%H:%M:%S") if turn.timestamp else ""
        sidechain_badge = " [bold magenta][SUBAGENT][/]" if turn.is_sidechain else ""

        if turn.role == "user":
            prompt_text = "\n\n".join(turn.text_blocks)
            console.print(
                Panel(
                    Markdown(prompt_text),
                    title=f"[bold bright_cyan]#{turn.index} 👤 USER[/]{sidechain_badge}",
                    subtitle=f"[dim]{ts_str} • line {turn.line_numbers[0]}[/]",
                    title_align="left",
                    border_style="bright_cyan",
                    box=box.ROUNDED,
                    padding=(0, 1),
                )
            )
            console.print()

        elif turn.role == "assistant":
            parts: list[RenderableType] = []

            # 1. Thinking blocks
            if turn.thinking_blocks and not only_tools:
                for th in turn.thinking_blocks:
                    th_lines = th.splitlines()
                    if show_thinking or full_output:
                        parts.append(
                            Panel(
                                Text(th, style="italic bright_black"),
                                title="[bold magenta]💭 Thinking[/]",
                                title_align="left",
                                border_style="magenta",
                                box=box.ROUNDED,
                                padding=(0, 1),
                            )
                        )
                    else:
                        preview = " ".join(th.split())
                        if len(preview) > 120:
                            preview = preview[:120] + "…"
                        parts.append(
                            Text.from_markup(
                                f"  [magenta]💭 Thinking ({len(th_lines)} lines):[/] [dim italic]{escape(preview)}[/] [dim](pass -t to expand)[/]"
                            )
                        )

            # 2. Assistant markdown text
            if turn.text_blocks and not only_tools:
                md_text = "\n\n".join(turn.text_blocks)
                parts.append(Markdown(md_text))

            # 3. Paired tool calls + results
            if turn.tool_calls and not no_tools:
                for tc in turn.tool_calls:
                    parts.append(_render_tool_card(tc, full_output=full_output))

            if not parts:
                continue

            u = turn.usage
            cost = u.estimated_cost_usd(default_model=session.primary_model)
            usage_badge = (
                f"{fmt_tokens(u.total_tokens)} tok "
                f"(in:{fmt_tokens(u.input_tokens)} out:{fmt_tokens(u.output_tokens)} "
                f"cache:{u.cache_hit_rate:.0f}%) • ${cost:.4f}"
                if u.total_tokens > 0
                else ""
            )

            console.print(
                Panel(
                    Group(*parts),
                    title=f"[bold green]#{turn.index} 🤖 CLAUDE[/]{sidechain_badge}",
                    subtitle=f"[dim]{ts_str} • {usage_badge}[/]",
                    title_align="left",
                    border_style="green" if not turn.is_sidechain else "magenta",
                    box=box.ROUNDED,
                    padding=(0, 1),
                )
            )
            console.print()

        elif turn.role == "system" and not only_tools:
            sys_text = "\n".join(turn.text_blocks)
            console.print(
                Panel(
                    Text(sys_text, style="yellow"),
                    title=f"[bold yellow]#{turn.index} ⚙ SYSTEM[/]",
                    title_align="left",
                    border_style="dim yellow",
                )
            )
            console.print()


def render_session_stats(session: Session, console: Console) -> None:
    """Render forensic analytics: token flame/bar chart, cache breakdown, tool latency, files touched."""
    console.print(render_session_header(session))
    console.print()

    # 1. Token & Cost Breakdown Table
    u = session.total_usage
    tok_table = Table(
        title="[bold bright_cyan]1. Token & Cache Forensics[/]",
        box=box.SIMPLE_HEAVY,
        header_style="bold cyan",
    )
    tok_table.add_column("Metric", style="bold")
    tok_table.add_column("Tokens", justify="right")
    tok_table.add_column("Share of Context", justify="right")
    tok_table.add_column("Est. Cost (USD)", justify="right", style="green")

    from .models import MODEL_PRICING, get_model_tier

    tier = get_model_tier(session.primary_model)
    rates = MODEL_PRICING[tier]
    total_tok = max(1, u.total_tokens)

    tok_table.add_row(
        "Cache Read Input (Hits)",
        f"{u.cache_read_input_tokens:,}",
        f"{(u.cache_read_input_tokens / total_tok) * 100:.1f}%",
        f"${(u.cache_read_input_tokens * rates['cache_read']) / 1e6:.4f}",
    )
    tok_table.add_row(
        "Cache Creation Input (Writes)",
        f"{u.cache_creation_input_tokens:,}",
        f"{(u.cache_creation_input_tokens / total_tok) * 100:.1f}%",
        f"${(u.cache_creation_input_tokens * rates['cache_write']) / 1e6:.4f}",
    )
    tok_table.add_row(
        "Uncached Input",
        f"{u.input_tokens:,}",
        f"{(u.input_tokens / total_tok) * 100:.1f}%",
        f"${(u.input_tokens * rates['input']) / 1e6:.4f}",
    )
    tok_table.add_row(
        "Output Generation",
        f"{u.output_tokens:,}",
        f"{(u.output_tokens / total_tok) * 100:.1f}%",
        f"${(u.output_tokens * rates['output']) / 1e6:.4f}",
    )
    tok_table.add_section()
    tok_table.add_row(
        "[bold]Total[/]",
        f"[bold]{u.total_tokens:,}[/]",
        "100.0%",
        f"[bold green]${session.total_cost_usd:.4f}[/]",
    )
    console.print(tok_table)
    console.print()

    # 2. Per-Turn Token Flame / Stacked Bar Chart
    assistant_turns = [t for t in session.turns if t.role == "assistant"]
    if assistant_turns:
        max_turn_tok = max((t.usage.total_tokens for t in assistant_turns), default=1)
        max_turn_tok = max(1, max_turn_tok)
        bar_width = 32

        flame_table = Table(
            title=(
                "[bold bright_cyan]2. Per-Turn Token Flame Chart[/]  "
                "[green]█ Cache Read[/]  [yellow]▓ Cache Write[/]  [cyan]▒ Input[/]  [magenta]░ Output[/]"
            ),
            box=box.SIMPLE_HEAVY,
            header_style="bold cyan",
        )
        flame_table.add_column("Turn", justify="right", style="bold")
        flame_table.add_column("Token Distribution", no_wrap=True)
        flame_table.add_column("Total", justify="right")
        flame_table.add_column("Out", justify="right", style="magenta")
        flame_table.add_column("Cache%", justify="right", style="green")
        flame_table.add_column("Cost", justify="right", style="green")
        flame_table.add_column("Tools Called", style="yellow")

        for t in assistant_turns:
            tu = t.usage
            tot = tu.total_tokens
            scale = tot / max_turn_tok
            total_cells = max(1, int(round(scale * bar_width))) if tot > 0 else 0

            bar = Text()
            if tot > 0 and total_cells > 0:
                c_read = int(round((tu.cache_read_input_tokens / tot) * total_cells))
                c_write = int(
                    round((tu.cache_creation_input_tokens / tot) * total_cells)
                )
                c_in = int(round((tu.input_tokens / tot) * total_cells))
                c_out = max(
                    1 if tu.output_tokens > 0 else 0,
                    total_cells - (c_read + c_write + c_in),
                )
                bar.append("█" * max(0, c_read), style="green")
                bar.append("▓" * max(0, c_write), style="yellow")
                bar.append("▒" * max(0, c_in), style="cyan")
                bar.append("░" * max(0, c_out), style="magenta")
            else:
                bar.append("-", style="dim")

            tools_str = ", ".join(tc.name for tc in t.tool_calls) or "-"
            if len(tools_str) > 32:
                tools_str = tools_str[:31] + "…"
            cost = tu.estimated_cost_usd(default_model=session.primary_model)

            flame_table.add_row(
                f"#{t.index}",
                bar,
                fmt_tokens(tot),
                fmt_tokens(tu.output_tokens),
                f"{tu.cache_hit_rate:.0f}%",
                f"${cost:.4f}",
                tools_str,
            )

        console.print(flame_table)
        console.print()

    # 3. Tool Latency & Usage Leaderboard
    all_calls = session.all_tool_calls
    if all_calls:
        by_tool: dict[str, list[ToolCall]] = defaultdict(list)
        for tc in all_calls:
            by_tool[tc.name].append(tc)

        tool_table = Table(
            title="[bold bright_cyan]3. Tool Execution & Latency Leaderboard[/]",
            box=box.SIMPLE_HEAVY,
            header_style="bold cyan",
        )
        tool_table.add_column("Tool Name", style="bold yellow")
        tool_table.add_column("Calls", justify="right")
        tool_table.add_column("Errors", justify="right")
        tool_table.add_column("Avg Latency", justify="right")
        tool_table.add_column("Max Latency", justify="right")
        tool_table.add_column("Total Time", justify="right", style="cyan")

        sorted_tools = sorted(
            by_tool.items(),
            key=lambda kv: (
                len(kv[1]),
                sum(c.duration_ms or 0.0 for c in kv[1]),
            ),
            reverse=True,
        )

        for tname, calls in sorted_tools:
            errs = sum(1 for c in calls if c.is_error)
            durations = [c.duration_ms for c in calls if c.duration_ms is not None]
            avg_ms = sum(durations) / len(durations) if durations else None
            max_ms = max(durations) if durations else None
            tot_ms = sum(durations) if durations else None

            err_str = f"[bold red]{errs}[/]" if errs > 0 else "[dim]0[/]"
            tool_table.add_row(
                tname,
                str(len(calls)),
                err_str,
                fmt_ms(avg_ms),
                fmt_ms(max_ms),
                fmt_ms(tot_ms),
            )

        console.print(tool_table)
        console.print()

    # 4. Files Touched Summary
    files_ops: dict[str, set[str]] = defaultdict(set)
    for tc in all_calls:
        fp = tc.input.get("file_path") or tc.input.get("path")
        if isinstance(fp, str) and fp.strip():
            files_ops[fp.strip()].add(tc.name)

    if files_ops:
        file_table = Table(
            title="[bold bright_cyan]4. Files Touched[/]",
            box=box.SIMPLE_HEAVY,
            header_style="bold cyan",
        )
        file_table.add_column("File Path", style="green")
        file_table.add_column("Operations", style="yellow")
        for fp, ops in sorted(files_ops.items()):
            file_table.add_row(fp, ", ".join(sorted(ops)))
        console.print(file_table)


def render_session_tree(session: Session, console: Console) -> None:
    """Render the UUID -> parentUuid conversation & subagent tree."""
    console.print(render_session_header(session))
    console.print()

    root_tree = Tree(
        f"[bold bright_cyan]🌳 Conversation & Subagent Tree ({len(session.raw_nodes)} JSONL events)[/]"
    )

    children_by_parent: dict[str | None, list[RawNode]] = defaultdict(list)
    known_uuids = {n.uuid for n in session.raw_nodes}

    for node in session.raw_nodes:
        parent = node.parent_uuid if node.parent_uuid in known_uuids else None
        children_by_parent[parent].append(node)

    def _format_node_label(node: RawNode, is_branch: bool) -> str:
        short_uuid = escape(node.uuid[:8])
        safe_label = escape(node.label)
        branch_tag = "[bold yellow]⎇ BRANCH [/]" if is_branch else ""
        side_tag = "[bold magenta]\\[SUBAGENT] [/]" if node.is_sidechain else ""

        if node.node_type == "user":
            if node.is_tool_result_only:
                return f"{branch_tag}{side_tag}[dim]L{node.line_no} ({short_uuid}) ↩ {safe_label}[/]"
            return f"{branch_tag}{side_tag}[bold cyan]L{node.line_no} ({short_uuid}) 👤 User:[/] {safe_label}"
        if node.node_type == "assistant":
            think_tag = "[magenta]💭[/] " if node.has_thinking else ""
            tools_str = escape(", ".join(node.tool_names))
            tools_tag = f" [yellow]\\[{tools_str}][/]" if node.tool_names else ""
            return f"{branch_tag}{side_tag}[bold green]L{node.line_no} ({short_uuid}) 🤖 Claude:[/] {think_tag}{safe_label}{tools_tag}"
        if node.node_type == "summary":
            return f"[bold bright_white]L{node.line_no} 📝 Summary:[/] [italic]{safe_label}[/]"
        return f"[yellow]L{node.line_no} ({short_uuid}) ⚙ {escape(node.node_type)}:[/] {safe_label}"

    def _add_children(parent_uuid: str | None, tree_node: Tree, depth: int = 0) -> None:
        if depth > 250:
            tree_node.add("[dim]... (max tree depth reached)[/]")
            return
        kids = children_by_parent.get(parent_uuid, [])
        is_branch = parent_uuid is not None and len(kids) > 1
        for child in kids:
            sub = tree_node.add(_format_node_label(child, is_branch=is_branch))
            _add_children(child.uuid, sub, depth + 1)

    _add_children(None, root_tree)
    console.print(root_tree)


def render_search_results(
    query: str,
    hits: list[SearchHit],
    console: Console,
) -> None:
    """Render full-text search hits across sessions."""
    if not hits:
        console.print(f"[yellow]No matches found for query:[/] [bold]{query}[/]")
        return

    table = Table(
        title=f"[bold bright_cyan]Search Results for '{query}' ({len(hits)} hits)[/]",
        box=box.SIMPLE_HEAVY,
        header_style="bold cyan",
    )
    table.add_column("Session", style="cyan", no_wrap=True)
    table.add_column("Project", style="green", no_wrap=True)
    table.add_column("Turn", justify="right", style="yellow")
    table.add_column("Source", style="magenta", no_wrap=True)
    table.add_column("Snippet")

    for h in hits:
        snippet_txt = Text(h.snippet)
        snippet_txt.highlight_words([query], style="bold black on yellow", case_sensitive=False)
        table.add_row(
            h.session.session_id[:10],
            h.session.project_name,
            f"#{h.turn_index}",
            h.match_source,
            snippet_txt,
        )

    console.print(table)
