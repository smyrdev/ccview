"""Command-line interface for the Claude Code JSONL Viewer (ccview)."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from pathlib import Path
import sys

from rich.console import Console

from .discovery import (
    DEFAULT_CLAUDE_DIR,
    discover_jsonl_files,
    resolve_session_target,
    search_sessions,
)
from .parser import parse_session
from .renderer import (
    render_search_results,
    render_session_list,
    render_session_show,
    render_session_stats,
    render_session_tree,
)
from .tui import InteractiveTUI


def _add_common_discovery_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-d",
        "--dir",
        dest="base_dir",
        default=None,
        help=f"Directory to scan for .jsonl logs (default: {DEFAULT_CLAUDE_DIR})",
    )
    parser.add_argument(
        "-p",
        "--project",
        dest="project_filter",
        default=None,
        help="Filter sessions by project name substring",
    )
    parser.add_argument(
        "--pager",
        action="store_true",
        help="Open output in an interactive terminal pager (less -R)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ccview",
        description="Visual CLI & Forensic Viewer for Claude Code .jsonl session logs.",
    )
    _add_common_discovery_args(parser)
    parser.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="Launch the interactive full-screen Terminal UI (TUI)",
    )

    subparsers = parser.add_subparsers(dest="command", metavar="<command>")

    # 0. tui / ui
    p_tui = subparsers.add_parser(
        "tui",
        aliases=["ui"],
        help="Launch interactive full-screen Terminal UI (browse sessions, expand turns/tools, tabs)",
    )
    _add_common_discovery_args(p_tui)
    p_tui.add_argument(
        "target",
        nargs="?",
        default=None,
        help="Optional session index, partial UUID, or .jsonl path to open immediately",
    )

    # 1. list / ls
    p_list = subparsers.add_parser(
        "list",
        aliases=["ls"],
        help="List discovered Claude Code sessions sorted by recency",
    )
    _add_common_discovery_args(p_list)
    p_list.add_argument(
        "-n",
        "--limit",
        type=int,
        default=25,
        help="Max sessions to display (default: 25)",
    )

    # 2. show
    p_show = subparsers.add_parser(
        "show",
        help="Replay chat timeline, thinking blocks, paired tool calls, and diffs",
    )
    _add_common_discovery_args(p_show)
    p_show.add_argument(
        "target",
        nargs="?",
        default="latest",
        help="Session index (e.g. 1), partial UUID, project name, or .jsonl file path (default: latest)",
    )
    p_show.add_argument(
        "-t",
        "--thinking",
        action="store_true",
        help="Expand full assistant thinking blocks",
    )
    p_show.add_argument(
        "-f",
        "--full",
        action="store_true",
        help="Show full untruncated tool outputs and file diffs",
    )
    p_show.add_argument(
        "--only-tools",
        action="store_true",
        help="Only display turns that invoke tools",
    )
    p_show.add_argument(
        "--no-tools",
        action="store_true",
        help="Hide tool cards and show only user/assistant messages",
    )
    p_show.add_argument(
        "--turn",
        type=int,
        default=None,
        help="Show only a specific turn number (e.g. --turn 3)",
    )

    # 3. stats
    p_stats = subparsers.add_parser(
        "stats",
        help="Forensic analytics: token flame chart, cache hit ratio, tool latency, files touched",
    )
    _add_common_discovery_args(p_stats)
    p_stats.add_argument(
        "target",
        nargs="?",
        default="latest",
        help="Session index, partial UUID, or .jsonl file path (default: latest)",
    )

    # 4. tree
    p_tree = subparsers.add_parser(
        "tree",
        help="Visualize parent-child UUID conversation & subagent branching tree",
    )
    _add_common_discovery_args(p_tree)
    p_tree.add_argument(
        "target",
        nargs="?",
        default="latest",
        help="Session index, partial UUID, or .jsonl file path (default: latest)",
    )

    # 5. search
    p_search = subparsers.add_parser(
        "search",
        help="Full-text search across prompts, replies, thinking, and tool calls",
    )
    _add_common_discovery_args(p_search)
    p_search.add_argument("query", help="Search string to match across sessions")
    p_search.add_argument(
        "-n",
        "--max-hits",
        type=int,
        default=50,
        help="Maximum matches to return (default: 50)",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    raw_args = list(sys.argv[1:] if argv is None else argv)

    # Convenience: if first positional arg is an existing .jsonl file, default to `show <file>`
    known_subcommands = {
        "tui",
        "ui",
        "list",
        "ls",
        "show",
        "stats",
        "tree",
        "search",
        "-h",
        "--help",
    }
    if (
        raw_args
        and raw_args[0] not in known_subcommands
        and not raw_args[0].startswith("-")
        and Path(raw_args[0]).expanduser().is_file()
    ):
        raw_args = ["show", *raw_args]

    parser = build_parser()
    args = parser.parse_args(raw_args)

    console = Console()
    cmd = args.command or ("tui" if getattr(args, "interactive", False) else "list")
    if getattr(args, "interactive", False) and cmd != "tui":
        cmd = "tui"

    if cmd in ("tui", "ui"):
        app = InteractiveTUI(
            base_dir=args.base_dir,
            project_filter=args.project_filter,
            initial_target=getattr(args, "target", None),
        )
        return app.run()
    pager_ctx = console.pager(styles=True) if getattr(args, "pager", False) else nullcontext()

    with pager_ctx:
        if cmd in ("list", "ls"):
            files = discover_jsonl_files(
                base_dir=args.base_dir,
                project_filter=args.project_filter,
            )
            limit = getattr(args, "limit", 25)
            render_session_list(files, console=console, limit=limit)
            return 0

        if cmd == "search":
            files = discover_jsonl_files(
                base_dir=args.base_dir,
                project_filter=args.project_filter,
            )
            hits = search_sessions(args.query, files, max_hits=args.max_hits)
            render_search_results(args.query, hits, console=console)
            return 0

        # Resolve target session for `show`, `stats`, or `tree`
        target_path = resolve_session_target(
            getattr(args, "target", "latest"),
            base_dir=args.base_dir,
            project_filter=args.project_filter,
        )
        if not target_path or not target_path.exists():
            console.print(
                f"[bold red]Error:[/] Could not resolve session target "
                f"[yellow]{getattr(args, 'target', 'latest')!r}[/yellow].\n"
                f"Checked directory: [dim]{args.base_dir or DEFAULT_CLAUDE_DIR}[/dim]"
            )
            return 1

        session = parse_session(target_path)

        if cmd == "show":
            render_session_show(
                session,
                console=console,
                show_thinking=args.thinking,
                full_output=args.full,
                only_tools=args.only_tools,
                no_tools=args.no_tools,
                turn_filter=args.turn,
            )
            return 0

        if cmd == "stats":
            render_session_stats(session, console=console)
            return 0

        if cmd == "tree":
            render_session_tree(session, console=console)
            return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
