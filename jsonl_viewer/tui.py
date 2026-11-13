"""Interactive Terminal UI (TUI) for browsing and inspecting Claude Code JSONL sessions."""

from __future__ import annotations

import io
import os
from pathlib import Path
import select
import sys
import termios
import tty

from rich import box
from rich.console import Console, Group, RenderableType
from rich.markdown import Markdown
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .discovery import discover_jsonl_files, resolve_session_target, search_sessions
from .models import Session, Turn
from .parser import parse_session
from .renderer import (
    _render_tool_card,
    fmt_duration,
    fmt_tokens,
    render_session_header,
    render_session_stats,
    render_session_tree,
)


_ESCAPE_MAP: dict[bytes, str] = {
    b"\x1b[A": "UP",
    b"\x1bOA": "UP",
    b"\x1b[B": "DOWN",
    b"\x1bOB": "DOWN",
    b"\x1b[C": "RIGHT",
    b"\x1bOC": "RIGHT",
    b"\x1b[D": "LEFT",
    b"\x1bOD": "LEFT",
    b"\x1b[5~": "PGUP",
    b"\x1b[6~": "PGDN",
    b"\x1b[H": "HOME",
    b"\x1bOH": "HOME",
    b"\x1b[1~": "HOME",
    b"\x1b[7~": "HOME",
    b"\x1b[F": "END",
    b"\x1bOF": "END",
    b"\x1b[4~": "END",
    b"\x1b[8~": "END",
    b"\x1b[Z": "SHIFT_TAB",
}


def read_key(fd: int) -> str:
    """Read a keypress or escape sequence directly from the unbuffered OS file descriptor.

    Using os.read(fd, ...) instead of sys.stdin.read(1) prevents Python's TextIOWrapper
    from swallowing trailing escape-sequence bytes (like '[A') into its internal buffer.
    """
    data = os.read(fd, 32)
    if not data:
        return "EOF"

    if data.startswith(b"\x1b"):
        # If only '\x1b' arrived so far, wait briefly in case the rest of an escape sequence is in flight
        while len(data) < 3 and select.select([fd], [], [], 0.025)[0]:
            more = os.read(fd, 32)
            if not more:
                break
            data += more

        if data == b"\x1b":
            return "ESC"

        if data in _ESCAPE_MAP:
            return _ESCAPE_MAP[data]

        # Handle modifier arrow keys (e.g. Shift/Ctrl + Up/Down: \x1b[1;2A)
        if data.startswith((b"\x1b[", b"\x1bO")) and len(data) >= 3:
            last = data[-1:]
            if last == b"A":
                return "UP"
            if last == b"B":
                return "DOWN"
            if last == b"C":
                return "RIGHT"
            if last == b"D":
                return "LEFT"
            if last == b"H":
                return "HOME"
            if last == b"F":
                return "END"
        # Unknown multi-byte escape sequence: ignore rather than treating as ESC
        return "IGNORE"

    ch = data.decode("utf-8", errors="replace")[0]
    if ch in ("\r", "\n"):
        return "ENTER"
    if ch in ("\x7f", "\x08"):
        return "BACKSPACE"
    if ch == "\t":
        return "TAB"
    if ch == "\x03":
        return "CTRL_C"
    if ch == "\x04":
        return "CTRL_D"
    if ch == "\x15":
        return "CTRL_U"
    return ch


def render_to_ansi_lines(renderables: list[RenderableType], width: int) -> list[str]:
    """Render a list of Rich renderables into raw ANSI-formatted lines for smooth viewport scrolling."""
    buf = io.StringIO()
    cap_console = Console(
        file=buf,
        width=max(40, width),
        force_terminal=True,
        color_system="truecolor",
    )
    for item in renderables:
        cap_console.print(item)
    return buf.getvalue().splitlines()


class InteractiveTUI:
    """Interactive terminal interface for Claude Code JSONL sessions."""

    def __init__(
        self,
        base_dir: Path | str | None = None,
        project_filter: str | None = None,
        initial_target: str | None = None,
    ) -> None:
        self.base_dir = base_dir
        self.project_filter = project_filter
        self.console = Console()

        self.all_files: list[Path] = discover_jsonl_files(
            base_dir=base_dir,
            project_filter=project_filter,
        )
        self.session_cache: dict[Path, Session] = {}
        self.filtered_files: list[Path] = list(self.all_files)

        # Modes: "browser" (session list) or "session" (session detail inspector)
        self.mode: str = "browser"
        self.selected_session_idx: int = 0
        self.browser_scroll: int = 0

        # Filter/search input state in browser mode
        self.filter_input_active: bool = False
        self.filter_query: str = ""

        # Active session inspector state
        self.current_session: Session | None = None
        self.active_tab: str = "chat"  # "chat" | "stats" | "tree"
        self.selected_turn_pos: int = 0  # index into visible turns
        self.scroll_y: int = 0
        self._last_total_lines: int = 0
        self._turn_line_ranges: list[tuple[int, int]] = []

        # Per-turn and global expansion toggles in chat tab
        self.expanded_thinking_turns: set[int] = set()
        self.expanded_full_turns: set[int] = set()
        self.global_thinking: bool = False
        self.global_full: bool = False
        self.only_tools: bool = False

        # Render cache for instant scrolling on large sessions
        self._turn_render_cache: dict[tuple[int, bool, bool, bool, int], list[str]] = {}
        self._tab_render_cache: dict[tuple[str, int], list[str]] = {}

        if initial_target:
            resolved = resolve_session_target(
                initial_target,
                base_dir=base_dir,
                project_filter=project_filter,
            )
            if resolved and resolved.exists():
                if resolved not in self.all_files:
                    self.all_files.insert(0, resolved)
                    self.filtered_files = list(self.all_files)
                self._open_session(resolved)

    def _get_session(self, path: Path) -> Session:
        if path not in self.session_cache:
            self.session_cache[path] = parse_session(path)
        return self.session_cache[path]

    def _apply_browser_filter(self) -> None:
        q = self.filter_query.strip().lower()
        if not q:
            self.filtered_files = list(self.all_files)
        else:
            matched: list[Path] = []
            for f in self.all_files:
                s = self._get_session(f)
                haystack = " ".join(
                    [
                        s.session_id,
                        s.project_name,
                        s.git_branch,
                        s.first_user_prompt,
                        *s.summaries,
                    ]
                ).lower()
                if q in haystack or search_sessions(q, [f], max_hits=1):
                    matched.append(f)
            self.filtered_files = matched
        self.selected_session_idx = 0
        self.browser_scroll = 0

    def _open_session(self, path: Path) -> None:
        self.current_session = self._get_session(path)
        self.mode = "session"
        self.active_tab = "chat"
        self.selected_turn_pos = 0
        self.scroll_y = 0
        self.expanded_thinking_turns.clear()
        self.expanded_full_turns.clear()
        self._turn_render_cache.clear()
        self._tab_render_cache.clear()

    def _visible_turns(self) -> list[Turn]:
        if not self.current_session:
            return []
        if self.only_tools:
            return [t for t in self.current_session.turns if t.tool_calls]
        return list(self.current_session.turns)

    def _build_top_bar(self, width: int) -> Text:
        bar = Text()
        bar.append(" ⚡ ccview TUI ", style="bold black on bright_cyan")
        if self.mode == "browser":
            bar.append("  📂 Sessions Browser ", style="bold bright_white")
            if self.filter_query:
                bar.append(f" [Filter: '{self.filter_query}'] ", style="bold yellow")
        elif self.current_session:
            s = self.current_session
            bar.append(f"  {s.project_name} ({s.session_id[:8]})  ", style="bold green")
            tabs = [
                ("chat", "1:Chat Timeline"),
                ("stats", "2:Forensic Stats"),
                ("tree", "3:UUID Tree"),
            ]
            for key, label in tabs:
                if self.active_tab == key:
                    bar.append(f" [{label}] ", style="bold black on yellow")
                else:
                    bar.append(f"  {label}  ", style="dim white")

            turns = self._visible_turns()
            if self.active_tab == "chat" and turns:
                bar.append(
                    f"  Turn {self.selected_turn_pos + 1}/{len(turns)} • Line {self.scroll_y + 1}/{max(1, self._last_total_lines)}",
                    style="cyan",
                )
        return bar

    def _build_bottom_bar(self) -> Text:
        bar = Text()
        if self.filter_input_active:
            bar.append(" Search/Filter: ", style="bold black on yellow")
            bar.append(f" {self.filter_query}█ ", style="bold white")
            bar.append("  (Enter: apply, Esc: cancel)", style="dim")
            return bar

        if self.mode == "browser":
            keys = [
                ("↑/↓ j/k", "Select"),
                ("Enter/→", "Open Session"),
                ("/", "Filter"),
                ("c", "Clear"),
                ("q", "Quit"),
            ]
        elif self.active_tab == "chat":
            keys = [
                ("↑/↓ j/k", "Move/Scroll"),
                ("n/p Tab", "Jump Turn"),
                ("PgUp/PgDn", "Page"),
                ("t/T", "Thinking"),
                ("Enter/f/F", "Full Output"),
                ("o", "Only-Tools"),
                ("1/2/3", "Tab"),
                ("b/←", "Back"),
                ("q", "Quit"),
            ]
        else:
            keys = [
                ("↑/↓ j/k", "Scroll"),
                ("PgUp/PgDn", "Page"),
                ("g/G", "Top/Bottom"),
                ("1/2/3", "Tab"),
                ("b/←", "Back"),
                ("q", "Quit"),
            ]

        for k, desc in keys:
            bar.append(f" {k} ", style="bold black on bright_white")
            bar.append(f" {desc} ", style="cyan")
        return bar

    def _render_browser_screen(self, width: int, body_height: int) -> list[str]:
        if not self.filtered_files:
            msg = Panel(
                "[yellow]No sessions match the current directory or filter.[/yellow]\n"
                "Press [bold]c[/bold] to clear filter or [bold]q[/bold] to exit.",
                title="No Sessions",
                border_style="yellow",
            )
            return render_to_ansi_lines([msg], width)

        table = Table(
            title=f"[bold bright_cyan]Claude Code Sessions ({len(self.filtered_files)})[/]",
            box=box.SIMPLE_HEAVY,
            header_style="bold bright_cyan",
            expand=True,
        )
        table.add_column(" ", width=2, no_wrap=True)
        table.add_column("#", justify="right", width=3, no_wrap=True)
        table.add_column("Session", width=10, no_wrap=True)
        table.add_column("Project", style="green", width=18, no_wrap=True)
        table.add_column("Branch", style="magenta", width=14, no_wrap=True)
        table.add_column("Started", width=16, no_wrap=True)
        table.add_column("Dur", justify="right", width=7, no_wrap=True)
        table.add_column("Tools", justify="right", width=5, no_wrap=True)
        table.add_column("Tokens", justify="right", width=7, no_wrap=True)
        table.add_column("Cost", justify="right", width=7, no_wrap=True)
        table.add_column("First Prompt / Summary", ratio=1, no_wrap=True)

        max_rows = max(5, body_height - 10)
        if self.selected_session_idx < self.browser_scroll:
            self.browser_scroll = self.selected_session_idx
        elif self.selected_session_idx >= self.browser_scroll + max_rows:
            self.browser_scroll = self.selected_session_idx - max_rows + 1

        visible_slice = self.filtered_files[
            self.browser_scroll : self.browser_scroll + max_rows
        ]

        for offset, fpath in enumerate(visible_slice):
            real_idx = self.browser_scroll + offset
            is_sel = real_idx == self.selected_session_idx
            s = self._get_session(fpath)
            pointer = "▶" if is_sel else " "
            row_style = "bold black on bright_cyan" if is_sel else ""

            started = s.start_time.strftime("%Y-%m-%d %H:%M") if s.start_time else "-"
            prompt = " ".join(s.first_user_prompt.split())
            if len(prompt) > 55:
                prompt = prompt[:54] + "…"

            table.add_row(
                pointer,
                str(real_idx + 1),
                s.session_id[:8],
                s.project_name[:18],
                (s.git_branch or "-")[:14],
                started,
                fmt_duration(s.duration_seconds),
                str(len(s.all_tool_calls)),
                fmt_tokens(s.total_usage.total_tokens),
                f"${s.total_cost_usd:.3f}",
                prompt,
                style=row_style,
            )

        sel_session = self._get_session(self.filtered_files[self.selected_session_idx])
        preview_header = render_session_header(sel_session)
        return render_to_ansi_lines([table, preview_header], width)

    def _render_single_turn_panel(
        self,
        turn: Turn,
        is_selected: bool,
        session: Session,
    ) -> Panel:
        show_th = self.global_thinking or (turn.index in self.expanded_thinking_turns)
        show_full = self.global_full or (turn.index in self.expanded_full_turns)
        ts_str = turn.timestamp.strftime("%H:%M:%S") if turn.timestamp else ""
        sel_marker = "▶ " if is_selected else ""
        sidechain_badge = " [bold magenta]\\[SUBAGENT][/]" if turn.is_sidechain else ""

        if turn.role == "user":
            prompt_text = "\n\n".join(turn.text_blocks)
            return Panel(
                Markdown(prompt_text),
                title=f"[bold bright_cyan]{sel_marker}#{turn.index} 👤 USER[/]{sidechain_badge}",
                subtitle=f"[dim]{ts_str} • line {turn.line_numbers[0]}[/]",
                title_align="left",
                border_style="bold bright_yellow" if is_selected else "bright_cyan",
                box=box.HEAVY if is_selected else box.ROUNDED,
                padding=(0, 1),
            )

        if turn.role == "assistant":
            parts: list[RenderableType] = []
            if turn.thinking_blocks:
                for th in turn.thinking_blocks:
                    th_lines = th.splitlines()
                    if show_th:
                        parts.append(
                            Panel(
                                Text(th, style="italic bright_black"),
                                title="[bold magenta]💭 Thinking (expanded — press 't' to collapse)[/]",
                                title_align="left",
                                border_style="magenta",
                                box=box.ROUNDED,
                                padding=(0, 1),
                            )
                        )
                    else:
                        preview = " ".join(th.split())
                        if len(preview) > 105:
                            preview = preview[:105] + "…"
                        parts.append(
                            Text.from_markup(
                                f"  [magenta]💭 Thinking ({len(th_lines)} lines):[/] [dim italic]{escape(preview)}[/] [dim](press 't' to expand)[/]"
                            )
                        )

            if turn.text_blocks:
                parts.append(Markdown("\n\n".join(turn.text_blocks)))

            if turn.tool_calls:
                for tc in turn.tool_calls:
                    parts.append(_render_tool_card(tc, full_output=show_full))

            if not parts:
                parts.append(Text("(empty assistant turn)", style="dim"))

            u = turn.usage
            cost = u.estimated_cost_usd(default_model=session.primary_model)
            state_tags: list[str] = []
            if show_th and turn.thinking_blocks:
                state_tags.append("THINKING:ON")
            if show_full and turn.tool_calls:
                state_tags.append("FULL:ON")
            tag_str = f" [{' '.join(state_tags)}]" if state_tags else ""

            usage_badge = (
                f"{fmt_tokens(u.total_tokens)} tok (cache:{u.cache_hit_rate:.0f}%) • ${cost:.4f}{tag_str}"
                if u.total_tokens > 0
                else tag_str
            )

            default_border = "magenta" if turn.is_sidechain else "green"
            return Panel(
                Group(*parts),
                title=f"[bold green]{sel_marker}#{turn.index} 🤖 CLAUDE[/]{sidechain_badge}",
                subtitle=f"[dim]{ts_str} • {usage_badge}[/]",
                title_align="left",
                border_style="bold bright_yellow" if is_selected else default_border,
                box=box.HEAVY if is_selected else box.ROUNDED,
                padding=(0, 1),
            )

        sys_text = "\n".join(turn.text_blocks)
        return Panel(
            Text(sys_text, style="yellow"),
            title=f"[bold yellow]{sel_marker}#{turn.index} ⚙ SYSTEM[/]",
            title_align="left",
            border_style="bold bright_yellow" if is_selected else "dim yellow",
        )

    def _get_rendered_turn_lines(
        self, turn: Turn, is_selected: bool, session: Session, width: int
    ) -> list[str]:
        show_th = self.global_thinking or (turn.index in self.expanded_thinking_turns)
        show_full = self.global_full or (turn.index in self.expanded_full_turns)
        cache_key = (turn.index, is_selected, show_th, show_full, width)
        if cache_key not in self._turn_render_cache:
            panel = self._render_single_turn_panel(
                turn, is_selected=is_selected, session=session
            )
            self._turn_render_cache[cache_key] = render_to_ansi_lines([panel], width)
        return self._turn_render_cache[cache_key]

    def _build_session_full_lines(self, width: int) -> list[str]:
        """Build the complete list of ANSI lines for the current session tab and record turn line ranges."""
        assert self.current_session is not None
        s = self.current_session

        if self.active_tab in ("stats", "tree"):
            cache_key = (self.active_tab, width)
            if cache_key not in self._tab_render_cache:
                buf = io.StringIO()
                cap = Console(
                    file=buf,
                    width=max(40, width),
                    force_terminal=True,
                    color_system="truecolor",
                )
                if self.active_tab == "stats":
                    render_session_stats(s, cap)
                else:
                    render_session_tree(s, cap)
                self._tab_render_cache[cache_key] = buf.getvalue().splitlines()
            lines = self._tab_render_cache[cache_key]
            self._last_total_lines = len(lines)
            return lines

        turns = self._visible_turns()
        if not turns:
            lines = render_to_ansi_lines(
                [
                    render_session_header(s),
                    Text("No turns match current filter.", style="yellow"),
                ],
                width,
            )
            self._last_total_lines = len(lines)
            self._turn_line_ranges = []
            return lines

        self.selected_turn_pos = max(0, min(self.selected_turn_pos, len(turns) - 1))

        all_lines: list[str] = []
        if ("header", width) not in self._tab_render_cache:
            self._tab_render_cache[("header", width)] = render_to_ansi_lines(
                [render_session_header(s)], width
            )
        all_lines.extend(self._tab_render_cache[("header", width)])
        all_lines.append("")

        ranges: list[tuple[int, int]] = []
        for pos, turn in enumerate(turns):
            is_sel = pos == self.selected_turn_pos
            t_lines = self._get_rendered_turn_lines(turn, is_sel, s, width)
            start_l = len(all_lines)
            end_l = start_l + len(t_lines)
            ranges.append((start_l, end_l))
            all_lines.extend(t_lines)
            all_lines.append("")

        self._turn_line_ranges = ranges
        self._last_total_lines = len(all_lines)
        return all_lines

    def _sync_selected_turn_from_scroll(self, body_height: int) -> None:
        """Keep selected_turn_pos aligned with the turn visible near the top of the viewport."""
        if not self._turn_line_ranges:
            return
        focus_line = self.scroll_y + min(4, body_height // 4)
        for pos, (start_l, end_l) in enumerate(self._turn_line_ranges):
            if start_l <= focus_line <= end_l or start_l >= focus_line:
                self.selected_turn_pos = pos
                break

    def _step_down_chat(self, body_height: int) -> None:
        """Smart Down action in chat view: scroll within a tall turn or advance to the next turn."""
        turns = self._visible_turns()
        if not turns or not self._turn_line_ranges:
            return
        pos = max(0, min(self.selected_turn_pos, len(turns) - 1))
        start_l, end_l = self._turn_line_ranges[pos]

        # If the current turn extends below the viewport, scroll down through it first
        if end_l > self.scroll_y + body_height:
            self.scroll_y += 5
            return

        # Otherwise advance to the next turn and bring it into view
        if pos < len(turns) - 1:
            self.selected_turn_pos = pos + 1
            n_start, n_end = self._turn_line_ranges[self.selected_turn_pos]
            if n_end > self.scroll_y + body_height:
                if (n_end - n_start) <= body_height:
                    self.scroll_y = max(0, n_end - body_height + 2)
                else:
                    self.scroll_y = max(0, n_start - 1)
        else:
            # Already on the last turn: allow scrolling to the very bottom
            self.scroll_y += 5

    def _step_up_chat(self, body_height: int) -> None:
        """Smart Up action in chat view: scroll up within a tall turn or move to the previous turn."""
        turns = self._visible_turns()
        if not turns or not self._turn_line_ranges:
            return
        pos = max(0, min(self.selected_turn_pos, len(turns) - 1))
        start_l, _ = self._turn_line_ranges[pos]

        # If the top of the current turn is above the viewport, scroll up through it first
        if start_l < self.scroll_y:
            self.scroll_y = max(0, self.scroll_y - 5)
            return

        # Otherwise move to the previous turn and bring it into view
        if pos > 0:
            self.selected_turn_pos = pos - 1
            p_start, p_end = self._turn_line_ranges[self.selected_turn_pos]
            if self.selected_turn_pos == 0:
                self.scroll_y = 0
            elif p_start < self.scroll_y:
                if (p_end - p_start) <= body_height:
                    self.scroll_y = max(0, p_start - 1)
                else:
                    self.scroll_y = max(0, p_end - body_height + 2)
        else:
            # Already on the first turn: scroll up to show the session header at line 0
            self.scroll_y = max(0, self.scroll_y - 5)

    def _render_session_screen(
        self, width: int, body_height: int, jump_to_selected_turn: bool = False
    ) -> list[str]:
        all_lines = self._build_session_full_lines(width)

        if (
            self.active_tab == "chat"
            and jump_to_selected_turn
            and self._turn_line_ranges
            and 0 <= self.selected_turn_pos < len(self._turn_line_ranges)
        ):
            start_l, _ = self._turn_line_ranges[self.selected_turn_pos]
            if self.selected_turn_pos == 0:
                self.scroll_y = 0
            else:
                self.scroll_y = max(0, start_l - 1)

        max_scroll = max(0, len(all_lines) - body_height)
        self.scroll_y = max(0, min(self.scroll_y, max_scroll))
        return all_lines[self.scroll_y : self.scroll_y + body_height]

    def _draw(self, jump_to_selected_turn: bool = False) -> None:
        size = self.console.size
        width = size.width
        height = size.height
        body_height = max(5, height - 2)

        if self.mode == "browser":
            body_lines = self._render_browser_screen(width, body_height)
        else:
            body_lines = self._render_session_screen(
                width, body_height, jump_to_selected_turn=jump_to_selected_turn
            )

        top_ansi = render_to_ansi_lines([self._build_top_bar(width)], width)
        bot_ansi = render_to_ansi_lines([self._build_bottom_bar()], width)

        while len(body_lines) < body_height:
            body_lines.append("")

        frame_lines = [
            top_ansi[0] if top_ansi else "",
            *body_lines[:body_height],
            bot_ansi[0] if bot_ansi else "",
        ]
        output = "\x1b[H" + "\r\n".join(f"\x1b[2K{line}" for line in frame_lines)
        sys.stdout.write(output)
        sys.stdout.flush()

    def handle_key(self, key: str, body_height: int = 35) -> bool:
        """Process a single logical key event. Returns False if the TUI should exit."""
        if key in ("CTRL_C", "EOF"):
            return False
        if key == "IGNORE":
            return True

        # Filter input handling in browser mode
        if self.filter_input_active:
            if key in ("ENTER", "ESC"):
                if key == "ESC":
                    self.filter_query = ""
                self.filter_input_active = False
                self._apply_browser_filter()
            elif key == "BACKSPACE":
                self.filter_query = self.filter_query[:-1]
                self._apply_browser_filter()
            elif len(key) == 1 and key.isprintable():
                self.filter_query += key
                self._apply_browser_filter()
            return True

        if key in ("q", "Q"):
            return False

        # Mode 1: Sessions Browser
        if self.mode == "browser":
            if key in ("DOWN", "j"):
                if self.filtered_files:
                    self.selected_session_idx = min(
                        len(self.filtered_files) - 1,
                        self.selected_session_idx + 1,
                    )
            elif key in ("UP", "k"):
                if self.filtered_files:
                    self.selected_session_idx = max(0, self.selected_session_idx - 1)
            elif key in ("PGDN", "CTRL_D", " "):
                if self.filtered_files:
                    self.selected_session_idx = min(
                        len(self.filtered_files) - 1,
                        self.selected_session_idx + 10,
                    )
            elif key in ("PGUP", "CTRL_U"):
                if self.filtered_files:
                    self.selected_session_idx = max(0, self.selected_session_idx - 10)
            elif key in ("g", "HOME"):
                self.selected_session_idx = 0
            elif key in ("G", "END"):
                if self.filtered_files:
                    self.selected_session_idx = len(self.filtered_files) - 1
            elif key == "/":
                self.filter_input_active = True
            elif key == "c":
                self.filter_query = ""
                self._apply_browser_filter()
            elif key in ("ENTER", "RIGHT", "l"):
                if self.filtered_files:
                    self._open_session(self.filtered_files[self.selected_session_idx])
            return True

        # Mode 2: Session Inspector
        if key in ("b", "BACKSPACE", "ESC", "LEFT", "h"):
            self.mode = "browser"
            return True
        if key == "1":
            self.active_tab = "chat"
            self.scroll_y = 0
            return True
        if key == "2":
            self.active_tab = "stats"
            self.scroll_y = 0
            return True
        if key == "3":
            self.active_tab = "tree"
            self.scroll_y = 0
            return True

        page_step = max(5, body_height - 4)

        if self.active_tab == "chat":
            # Ensure line ranges are populated before computing scroll steps
            if not self._turn_line_ranges:
                self._build_session_full_lines(self.console.size.width)

            turns = self._visible_turns()
            cur_turn = (
                turns[self.selected_turn_pos]
                if turns and 0 <= self.selected_turn_pos < len(turns)
                else None
            )

            if key in ("DOWN", "j"):
                self._step_down_chat(body_height)
            elif key in ("UP", "k"):
                self._step_up_chat(body_height)
            elif key in ("PGDN", "CTRL_D", "d"):
                self.scroll_y += page_step
                self._sync_selected_turn_from_scroll(body_height)
            elif key in ("PGUP", "CTRL_U", "u"):
                self.scroll_y = max(0, self.scroll_y - page_step)
                self._sync_selected_turn_from_scroll(body_height)
            elif key in ("n", "]", "TAB"):
                if turns and self.selected_turn_pos < len(turns) - 1:
                    self.selected_turn_pos += 1
                    start_l, _ = self._turn_line_ranges[self.selected_turn_pos]
                    self.scroll_y = max(0, start_l - 1)
            elif key in ("p", "[", "SHIFT_TAB"):
                if turns and self.selected_turn_pos > 0:
                    self.selected_turn_pos -= 1
                    if self.selected_turn_pos == 0:
                        self.scroll_y = 0
                    else:
                        start_l, _ = self._turn_line_ranges[self.selected_turn_pos]
                        self.scroll_y = max(0, start_l - 1)
            elif key in ("g", "HOME"):
                self.selected_turn_pos = 0
                self.scroll_y = 0
            elif key in ("G", "END"):
                if turns:
                    self.selected_turn_pos = len(turns) - 1
                self.scroll_y = max(0, self._last_total_lines - body_height)
            elif key == "t" and cur_turn:
                if cur_turn.index in self.expanded_thinking_turns:
                    self.expanded_thinking_turns.remove(cur_turn.index)
                else:
                    self.expanded_thinking_turns.add(cur_turn.index)
            elif key == "T":
                self.global_thinking = not self.global_thinking
            elif key in ("f", " ", "ENTER") and cur_turn:
                if cur_turn.index in self.expanded_full_turns:
                    self.expanded_full_turns.remove(cur_turn.index)
                else:
                    self.expanded_full_turns.add(cur_turn.index)
            elif key == "F":
                self.global_full = not self.global_full
            elif key == "o":
                self.only_tools = not self.only_tools
                self.selected_turn_pos = 0
                self.scroll_y = 0
        else:
            # Stats or Tree tab scrolling
            if key in ("DOWN", "j"):
                self.scroll_y += 3
            elif key in ("UP", "k"):
                self.scroll_y = max(0, self.scroll_y - 3)
            elif key in ("PGDN", "CTRL_D", "d", " "):
                self.scroll_y += page_step
            elif key in ("PGUP", "CTRL_U", "u"):
                self.scroll_y = max(0, self.scroll_y - page_step)
            elif key in ("g", "HOME"):
                self.scroll_y = 0
            elif key in ("G", "END"):
                self.scroll_y = max(0, self._last_total_lines - body_height)

        return True

    def run(self) -> int:
        if not sys.stdin.isatty() or not sys.stdout.isatty():
            self.console.print(
                "[red]Interactive TUI requires an interactive TTY terminal.[/red]"
            )
            return 1

        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            sys.stdout.write("\x1b[?1049h\x1b[?25l\x1b[2J")
            sys.stdout.flush()
            tty.setraw(fd)

            while True:
                self._draw()
                key = read_key(fd)
                body_h = max(5, self.console.size.height - 2)
                if not self.handle_key(key, body_height=body_h):
                    break
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            sys.stdout.write("\x1b[?25h\x1b[?1049l")
            sys.stdout.flush()

        return 0
