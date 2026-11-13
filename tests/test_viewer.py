"""Unit tests for the Claude Code JSONL Viewer."""

from __future__ import annotations

import os
from pathlib import Path
import unittest

from jsonl_viewer.cli import main
from jsonl_viewer.discovery import discover_jsonl_files, resolve_session_target, search_sessions
from jsonl_viewer.parser import parse_session
from jsonl_viewer.tui import InteractiveTUI, read_key


EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"
SAMPLE_FILE = EXAMPLES_DIR / "sample-claude-session.jsonl"


class TestClaudeJsonlViewer(unittest.TestCase):
    def test_parse_sample_session(self) -> None:
        session = parse_session(SAMPLE_FILE)
        self.assertEqual(session.session_id, "8f4b2c91-3e1a-4c72-9d01-abcdef123456")
        self.assertEqual(session.git_branch, "fix/jwt-expiry")
        self.assertEqual(session.project_name, "Work/api-server")
        self.assertEqual(session.user_prompt_count, 1)
        # msg_01A (merged from 2 lines), msg_01B, msg_sub_1, msg_01C, msg_01D => 5 assistant turns
        self.assertEqual(session.assistant_turn_count, 5)
        self.assertEqual(len(session.all_tool_calls), 5)

        # Verify tool_use + tool_result pairing and latency calculation
        calls_by_name = {tc.name: tc for tc in session.all_tool_calls}
        self.assertTrue(calls_by_name["Read"].has_result)
        self.assertAlmostEqual(calls_by_name["Read"].duration_ms or 0, 180.0, places=1)
        self.assertTrue(calls_by_name["Edit"].has_result)
        self.assertIn("ExpiredSignatureError", calls_by_name["Edit"].input["new_string"])

        # Verify token deduplication across multi-line msg_01A
        # msg_01A (420) + msg_01B (310) + msg_sub_1 (190) + msg_01C (280) + msg_01D (190) = 1390
        self.assertEqual(session.total_usage.input_tokens, 1390)
        self.assertGreater(session.total_cost_usd, 0.0)

    def test_discovery_and_search(self) -> None:
        files = discover_jsonl_files(EXAMPLES_DIR)
        self.assertEqual(len(files), 1)

        resolved = resolve_session_target("1", base_dir=EXAMPLES_DIR)
        self.assertEqual(resolved, SAMPLE_FILE.resolve())

        hits = search_sessions("ExpiredSignatureError", files)
        self.assertGreaterEqual(len(hits), 1)

    def test_cli_commands_exit_zero(self) -> None:
        self.assertEqual(main(["list", "-d", str(EXAMPLES_DIR)]), 0)
        self.assertEqual(main(["show", "1", "-d", str(EXAMPLES_DIR), "-t", "-f"]), 0)
        self.assertEqual(main(["stats", "1", "-d", str(EXAMPLES_DIR)]), 0)
        self.assertEqual(main(["tree", "1", "-d", str(EXAMPLES_DIR)]), 0)
        self.assertEqual(main(["search", "ExpiredSignatureError", "-d", str(EXAMPLES_DIR)]), 0)

    def test_interactive_tui_state_and_rendering(self) -> None:
        tui = InteractiveTUI(base_dir=EXAMPLES_DIR)
        self.assertEqual(tui.mode, "browser")
        browser_lines = tui._render_browser_screen(width=100, body_height=30)
        self.assertTrue(any("8f4b2c91" in line for line in browser_lines))

        # Filter sessions live
        tui.filter_query = "ExpiredSignatureError"
        tui._apply_browser_filter()
        self.assertEqual(len(tui.filtered_files), 1)

        # Open session and test tabs & per-turn toggles
        tui._open_session(tui.filtered_files[0])
        self.assertEqual(tui.mode, "session")
        self.assertEqual(tui.active_tab, "chat")
        self.assertEqual(tui.selected_turn_pos, 0)

        # Render initial frame with small viewport height (20 lines) to test scrolling & turn stepping
        tui._render_session_screen(width=100, body_height=20)
        self.assertEqual(tui.selected_turn_pos, 0)

        # Pressing DOWN moves selection from turn #1 (pos 0) to turn #2 (pos 1) and scrolls viewport
        tui.handle_key("DOWN", body_height=20)
        self.assertEqual(tui.selected_turn_pos, 1)
        tui._render_session_screen(width=100, body_height=20)
        self.assertGreater(tui.scroll_y, 0)

        # Pressing UP moves back to turn #1 (pos 0) and scrolls back to 0
        tui.handle_key("UP", body_height=20)
        self.assertEqual(tui.selected_turn_pos, 0)
        self.assertEqual(tui.scroll_y, 0)

        # Toggle thinking on turn #2
        tui.expanded_thinking_turns.add(2)
        chat_lines = tui._render_session_screen(width=100, body_height=40)
        self.assertTrue(any("Thinking (expanded" in line for line in chat_lines))

        # Switch to stats and tree tabs and test DOWN/UP scrolling
        tui.handle_key("2", body_height=20)
        self.assertEqual(tui.active_tab, "stats")
        stats_lines = tui._render_session_screen(width=100, body_height=20)
        self.assertTrue(any("Token & Cache Forensics" in line for line in stats_lines))
        tui.handle_key("DOWN", body_height=20)
        self.assertEqual(tui.scroll_y, 3)
        tui.handle_key("UP", body_height=20)
        self.assertEqual(tui.scroll_y, 0)

        tui.handle_key("3", body_height=20)
        self.assertEqual(tui.active_tab, "tree")
        tree_lines = tui._render_session_screen(width=100, body_height=40)
        self.assertTrue(any("Conversation & Subagent Tree" in line for line in tree_lines))

    def test_unbuffered_escape_sequence_reading(self) -> None:
        r_fd, w_fd = os.pipe()
        try:
            os.write(w_fd, b"\x1b[B")
            self.assertEqual(read_key(r_fd), "DOWN")

            os.write(w_fd, b"\x1b[A")
            self.assertEqual(read_key(r_fd), "UP")

            os.write(w_fd, b"\x1bOB")
            self.assertEqual(read_key(r_fd), "DOWN")

            os.write(w_fd, b"\x1b[6~")
            self.assertEqual(read_key(r_fd), "PGDN")

            os.write(w_fd, b"\x1b")
            self.assertEqual(read_key(r_fd), "ESC")
        finally:
            os.close(r_fd)
            os.close(w_fd)


if __name__ == "__main__":
    unittest.main()
