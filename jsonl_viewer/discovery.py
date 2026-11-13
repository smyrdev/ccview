"""Auto-discovery and search across Claude Code JSONL session logs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .models import Session
from .parser import parse_session


DEFAULT_CLAUDE_DIR = Path.home() / ".claude" / "projects"


@dataclass
class SearchHit:
    session: Session
    turn_index: int
    role: str
    match_source: str  # "prompt", "response", "thinking", "tool_input", "tool_result"
    snippet: str


def discover_jsonl_files(
    base_dir: Path | str | None = None,
    project_filter: str | None = None,
) -> list[Path]:
    """Find all .jsonl session files sorted by modification time (newest first)."""
    root = Path(base_dir).expanduser() if base_dir else DEFAULT_CLAUDE_DIR
    if not root.exists():
        return []

    if root.is_file() and root.suffix == ".jsonl":
        return [root.resolve()]

    files = list(root.rglob("*.jsonl"))
    if project_filter:
        pf_low = project_filter.lower()
        files = [
            f
            for f in files
            if pf_low in f.parent.name.lower() or pf_low in f.name.lower()
        ]

    # Sort newest first by mtime (and fallback to filename for deterministic order)
    files.sort(key=lambda p: (p.stat().st_mtime, p.name), reverse=True)
    return files


def resolve_session_target(
    target: str | None,
    base_dir: Path | str | None = None,
    project_filter: str | None = None,
) -> Path | None:
    """Resolve a target argument (file path, 1-based index, or partial UUID) to a .jsonl Path."""
    # 1. Direct file path check
    if target:
        candidate = Path(target).expanduser()
        if candidate.is_file():
            return candidate.resolve()

    files = discover_jsonl_files(base_dir=base_dir, project_filter=project_filter)
    if not files:
        return None

    # 2. Default to most recent session if target is None or "latest"
    if not target or target.lower() in ("latest", "last"):
        return files[0]

    # 3. 1-based numeric index from `list` output (e.g. "1" or "#1")
    clean_num = target.lstrip("#")
    if clean_num.isdigit():
        idx = int(clean_num)
        if 1 <= idx <= len(files):
            return files[idx - 1]

    # 4. Partial session ID or filename match
    t_low = target.lower()
    for f in files:
        if f.stem.lower().startswith(t_low) or t_low in f.stem.lower():
            return f

    # 5. Match by project folder name
    for f in files:
        if t_low in f.parent.name.lower():
            return f

    return None


def _make_snippet(text: str, query: str, radius: int = 70) -> str:
    """Extract a single-line snippet centered around the first match of query."""
    clean = " ".join(text.split())
    idx = clean.lower().find(query.lower())
    if idx == -1:
        return clean[: radius * 2]
    start = max(0, idx - radius)
    end = min(len(clean), idx + len(query) + radius)
    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(clean) else ""
    return f"{prefix}{clean[start:end]}{suffix}"


def search_sessions(
    query: str,
    files: Iterable[Path],
    max_hits: int = 50,
) -> list[SearchHit]:
    """Full-text search across user prompts, assistant responses, thinking, and tools."""
    q_low = query.lower()
    hits: list[SearchHit] = []

    for path in files:
        session = parse_session(path)
        for turn in session.turns:
            # Check text blocks
            for txt in turn.text_blocks:
                if q_low in txt.lower():
                    src = "prompt" if turn.role == "user" else "response"
                    hits.append(
                        SearchHit(
                            session=session,
                            turn_index=turn.index,
                            role=turn.role,
                            match_source=src,
                            snippet=_make_snippet(txt, query),
                        )
                    )
                    if len(hits) >= max_hits:
                        return hits

            # Check thinking blocks
            for th in turn.thinking_blocks:
                if q_low in th.lower():
                    hits.append(
                        SearchHit(
                            session=session,
                            turn_index=turn.index,
                            role=turn.role,
                            match_source="thinking",
                            snippet=_make_snippet(th, query),
                        )
                    )
                    if len(hits) >= max_hits:
                        return hits

            # Check tool calls & results
            for tc in turn.tool_calls:
                input_str = f"{tc.name} {tc.input}"
                if q_low in input_str.lower():
                    hits.append(
                        SearchHit(
                            session=session,
                            turn_index=turn.index,
                            role=turn.role,
                            match_source=f"tool:{tc.name}",
                            snippet=_make_snippet(input_str, query),
                        )
                    )
                    if len(hits) >= max_hits:
                        return hits
                if tc.result_content and q_low in tc.result_content.lower():
                    hits.append(
                        SearchHit(
                            session=session,
                            turn_index=turn.index,
                            role=turn.role,
                            match_source=f"result:{tc.name}",
                            snippet=_make_snippet(tc.result_content, query),
                        )
                    )
                    if len(hits) >= max_hits:
                        return hits

    return hits
