"""Regression tests for text decoding of ffmpeg and Claude output on machines with a legacy code page.

On Windows, Python decodes subprocess text with the system code page (cp1252) unless told otherwise,
but ffmpeg prints episode metadata (titles, artists) as UTF-8. A character such as "A-acute" has the
bytes C3 81, and 0x81 is undefined in cp1252, so reading ffmpeg's output raised UnicodeDecodeError in
a reader thread, left ``stderr`` as None, and failed the whole cut. These tests reproduce that
condition on any platform by forcing a C locale with no UTF-8 mode.
"""

from __future__ import annotations

import ast
import os
import shutil
import stat
import subprocess
import sys
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from siphon import cutter

SRC = Path(cutter.__file__).parent
LEGACY_ENV = {"LC_ALL": "C", "LANG": "C", "PYTHONUTF8": "0", "PYTHONCOERCECLOCALE": "0"}

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
                                  reason="ffmpeg not available")


def _run_legacy(script: str, extra_env: dict | None = None) -> subprocess.CompletedProcess:
    """Run ``script`` in a fresh Python that decodes subprocess text with a legacy encoding."""
    env = {**os.environ, **LEGACY_ENV, "PYTHONPATH": os.pathsep.join(p for p in sys.path if p), **(extra_env or {})}
    return subprocess.run([sys.executable, "-c", textwrap.dedent(script)], env=env, capture_output=True, timeout=300)


@needs_ffmpeg
class TestNonAsciiMetadataUnderALegacyEncoding:
    SCRIPT = '''
        import os, subprocess, sys, tempfile
        from siphon import cutter

        tmp = tempfile.mkdtemp()
        meta = os.path.join(tmp, "meta.txt")
        with open(meta, "w", encoding="utf-8") as f:
            f.write(";FFMETADATA1\\ntitle=Pivot \\u00c1 Caf\\u00e9 \\u2013 Episode\\nartist=Vox Media \\u00c1\\n")
        path = os.path.join(tmp, "ep.mp3")
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "sine=frequency=440:duration=40",
                        "-i", meta, "-map_metadata", "1", "-ac", "2", "-c:a", "libmp3lame", path], check=True)

        assert isinstance(cutter.find_nearest_silence(path, 20.0), float)
        out = os.path.join(tmp, "out.mp3")
        report = {}
        cutter.cut_segments(path, [{"start": 10, "end": 18}], output_path=out,
                            snap_to_silence=True, chime=True, report=report)
        assert report["chime_count"] == 1
        cutter.extract_audio(path, os.path.join(tmp, "a.wav"))
        print("OK", round(cutter.get_duration(out), 1))
    '''

    def test_cutting_snapping_chiming_and_extracting_all_work(self):
        result = _run_legacy(self.SCRIPT)
        assert result.returncode == 0, result.stderr.decode("utf-8", "replace")[-1500:]
        assert result.stdout.decode().startswith("OK")
        assert b"Exception in thread" not in result.stderr          # the old failure showed up as a reader-thread error

    def test_this_really_is_a_legacy_environment(self):
        result = _run_legacy("import locale; print(locale.getencoding())")
        assert result.stdout.decode().strip() in ("ANSI_X3.4-1968", "US-ASCII", "ascii", "cp1252", "646")


class TestOutputIsDecodedAsUtf8:
    """Mocked checks that every call asks for UTF-8 with replacement, never the system default."""

    @pytest.mark.parametrize("call", [
        lambda: cutter.get_duration("/f.mp3"),
        lambda: cutter.validate_file("/f.mp3") if os.path.exists("/f.mp3") else None,
        lambda: cutter.find_nearest_silence("/f.mp3", 10.0),
        lambda: cutter.extract_audio("/f.mp3", "/o.wav"),
        lambda: cutter.has_real_video_stream("/f.mp3"),
        lambda: cutter.audio_stream_info("/f.mp3"),
    ])
    def test_cutter_calls_decode_as_utf8(self, call):
        with patch("siphon.cutter.subprocess.run", return_value=MagicMock(returncode=0, stdout="1.0", stderr="")) as run, \
             patch("siphon.cutter.os.path.exists", return_value=True):
            try:
                call()
            except Exception:
                pass
        for invocation in run.call_args_list:
            assert invocation.kwargs.get("encoding") == "utf-8" and invocation.kwargs.get("errors") == "replace"
            assert not invocation.kwargs.get("text")

    def test_silence_detection_survives_missing_output(self):
        with patch("siphon.cutter.subprocess.run", return_value=MagicMock(returncode=0, stderr=None)):
            assert cutter.find_nearest_silence("/f.mp3", 42.0) == 42.0

    def test_the_concat_list_is_written_as_utf8_even_for_unusual_paths(self, tmp_path):
        weird = tmp_path / "Café Á"
        weird.mkdir()
        seen = {}

        class FakeTempDir:
            def __enter__(self):
                return str(weird)

            def __exit__(self, *args):
                return False

        def fake_run(cmd, *args, **kwargs):
            if "-f" in cmd and cmd[cmd.index("-f") + 1] == "concat":
                seen["bytes"] = Path(cmd[cmd.index("-i") + 1]).read_bytes()
            return MagicMock(returncode=0, stderr="", stdout="")

        with patch("shutil.move"), patch("siphon.cutter.validate_file", return_value=True), \
             patch("siphon.cutter.get_duration", return_value=600.0), \
             patch("siphon.cutter.tempfile.TemporaryDirectory", FakeTempDir), \
             patch("siphon.cutter.subprocess.run", side_effect=fake_run):
            cutter.cut_segments("/tmp/in.mp3", [{"start": 100, "end": 130}], output_path="/tmp/out.mp3")

        assert "Café Á".encode("utf-8") in seen["bytes"]


class TestNoLocaleDependentSubprocessText:
    """A guard against the problem coming back anywhere in Siphon."""

    def test_no_subprocess_call_decodes_with_the_system_encoding(self):
        offenders = []
        for path in sorted(SRC.rglob("*.py")):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                        and isinstance(node.func.value, ast.Name) and node.func.value.id == "subprocess"
                        and node.func.attr in ("run", "Popen", "check_output")):
                    continue
                keywords = {k.arg: k.value for k in node.keywords}
                text_mode = any(
                    isinstance(keywords.get(name), ast.Constant) and keywords[name].value is True
                    for name in ("text", "universal_newlines")
                )
                if text_mode and "encoding" not in keywords:
                    offenders.append(f"{path.relative_to(SRC)}:{node.lineno}")
        assert offenders == [], f"subprocess output is decoded with the system code page at: {offenders}"


@pytest.mark.skipif(sys.platform == "win32", reason="uses a shell script as a stand-in for the claude CLI")
class TestClaudeOutputUnderALegacyEncoding:
    def test_non_ascii_labels_from_claude_are_read_intact(self, tmp_path):
        fake = tmp_path / "claude"
        # prints UTF-8 bytes for "Caf-e-acute A-acute"; the second byte of A-acute is 0x81, undefined in cp1252
        fake.write_bytes(b"#!/bin/sh\ncat > /dev/null\n"
                         b"printf '{\"structured_output\": {\"segments\": [{\"start\": 1, \"end\": 30, "
                         b"\"label\": \"Caf\\303\\251 \\303\\201 sponsor\", \"confidence\": 0.9}]}}'\n")
        fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
        script = '''
            import json
            from siphon.ad_detect import detect_ads
            result = detect_ads("some transcript", "find ads", segments=[{"start": 0, "end": 5, "text": "hi"}])
            print(json.dumps(result["segments"][0]["label"], ensure_ascii=True))
        '''
        result = _run_legacy(script, {"PATH": f"{tmp_path}{os.pathsep}{os.environ['PATH']}"})
        assert result.returncode == 0, result.stderr.decode("utf-8", "replace")[-1500:]
        assert result.stdout.decode().strip() == '"Caf\\u00e9 \\u00c1 sponsor"'
