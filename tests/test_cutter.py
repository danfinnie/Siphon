"""Tests for siphon.cutter — ffmpeg-based segment cutting."""

from __future__ import annotations

import os
import subprocess
from unittest.mock import patch, MagicMock

import pytest

from siphon.cutter import (
    cut_segments, extract_audio, find_nearest_silence, get_duration, has_real_video_stream,
)


def _has_ffmpeg() -> bool:
    """Check if ffmpeg is available on PATH."""
    try:
        result = subprocess.run(["ffmpeg", "-version"], capture_output=True, timeout=5)
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


needs_ffmpeg = pytest.mark.skipif(not _has_ffmpeg(), reason="ffmpeg not available")


# ------------------------------------------------------------------ #
# Unit tests with mocked subprocess
# ------------------------------------------------------------------ #

class TestExtractAudioMocked:
    @patch("siphon.cutter.subprocess.run")
    def test_calls_ffmpeg(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stderr="")
        extract_audio("/input.mp4", "/output.wav")
        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "ffmpeg"
        assert "/input.mp4" in cmd
        assert "/output.wav" in cmd

    @patch("siphon.cutter.subprocess.run")
    def test_raises_on_failure(self, mock_run):
        mock_run.return_value = MagicMock(returncode=1, stderr="error")
        with pytest.raises(RuntimeError, match="extraction failed"):
            extract_audio("/input.mp4", "/output.wav")


class TestGetDurationMocked:
    @patch("siphon.cutter.subprocess.run")
    def test_parses_duration(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="123.456\n", stderr="")
        duration = get_duration("/file.mp3")
        assert duration == pytest.approx(123.456)


class TestHasRealVideoStreamMocked:
    @patch("siphon.cutter.subprocess.run")
    def test_real_h264_video(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="h264,27000\n", stderr="")
        assert has_real_video_stream("/file.mp4") is True

    @patch("siphon.cutter.subprocess.run")
    def test_vp9_video_unknown_frame_count(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="vp9,N/A\n", stderr="")
        assert has_real_video_stream("/file.mp4") is True

    @patch("siphon.cutter.subprocess.run")
    def test_png_thumbnail_only(self, mock_run):
        # The matt-mcmuscles shape: PNG "video" with 7 frames
        mock_run.return_value = MagicMock(returncode=0, stdout="png,7\n", stderr="")
        assert has_real_video_stream("/file.mp4") is False

    @patch("siphon.cutter.subprocess.run")
    def test_mjpeg_thumbnail_only(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="mjpeg,1\n", stderr="")
        assert has_real_video_stream("/file.mp4") is False

    @patch("siphon.cutter.subprocess.run")
    def test_no_video_stream_at_all(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        assert has_real_video_stream("/file.mp3") is False

    @patch("siphon.cutter.subprocess.run")
    def test_png_plus_real_video(self, mock_run):
        # Real video with an attached thumbnail — must detect the real one
        mock_run.return_value = MagicMock(returncode=0, stdout="png,1\nh264,27000\n", stderr="")
        assert has_real_video_stream("/file.mp4") is True

    @patch("siphon.cutter.subprocess.run")
    def test_ffprobe_failure(self, mock_run):
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="error")
        assert has_real_video_stream("/file.mp4") is False


class TestCutSegmentsMocked:
    def test_no_segments_returns_input(self):
        result = cut_segments("/file.mp3", [])
        assert result == "/file.mp3"

    @patch("shutil.move")
    @patch("siphon.cutter.validate_file")
    @patch("siphon.cutter.get_duration")
    @patch("siphon.cutter.subprocess.run")
    def test_calls_ffmpeg_for_segments(self, mock_run, mock_duration, mock_validate, mock_move):
        mock_duration.return_value = 600.0
        mock_run.return_value = MagicMock(returncode=0, stderr="")
        mock_validate.return_value = True

        result = cut_segments(
            "/tmp/test.mp3",
            [{"start": 10.0, "end": 40.0}],
            output_path="/tmp/output.mp3",
        )

        assert result == "/tmp/output.mp3"
        # Should have called ffmpeg multiple times (segment extraction + concat)
        assert mock_run.call_count >= 2
        # Should have validated the output
        mock_validate.assert_called_once()


class TestCutSegmentsMerging:
    """Ads that touch or nearly touch are removed as one piece; no sliver is kept between them."""

    @staticmethod
    def _keep_ranges(segments, total=600.0, **kwargs):
        with patch("shutil.move"), \
             patch("siphon.cutter.validate_file", return_value=True), \
             patch("siphon.cutter.get_duration", return_value=total), \
             patch("siphon.cutter.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stderr="")
            cut_segments("/tmp/test.mp3", segments, output_path="/tmp/out.mp3", **kwargs)
        ranges = []
        for call in mock_run.call_args_list:
            cmd = call.args[0]
            if "-ss" in cmd:
                ranges.append((float(cmd[cmd.index("-ss") + 1]), float(cmd[cmd.index("-to") + 1])))
        return ranges

    def test_back_to_back_segments_leave_one_gap_in_the_audio(self):
        ranges = self._keep_ranges([{"start": 0, "end": 30}, {"start": 30, "end": 60}])
        assert ranges == [(60.0, 600.0)]

    def test_sliver_between_close_segments_is_not_kept(self):
        ranges = self._keep_ranges([{"start": 100, "end": 130}, {"start": 130.6, "end": 160}])
        assert ranges == [(0.0, 100.0), (160.0, 600.0)]

    def test_distant_segments_keep_the_content_between_them(self):
        ranges = self._keep_ranges([{"start": 100, "end": 130}, {"start": 200, "end": 230}])
        assert ranges == [(0.0, 100.0), (130.0, 200.0), (230.0, 600.0)]

    def test_merge_gap_can_be_disabled(self):
        ranges = self._keep_ranges(
            [{"start": 100, "end": 130}, {"start": 130.5, "end": 160}], merge_gap=0.0,
        )
        assert (130.0, 130.5) in ranges


SILENCE_LOG = """
[silencedetect @ 0x1] silence_start: 1.2
[silencedetect @ 0x1] silence_end: 2.0 | silence_duration: 0.8
[silencedetect @ 0x1] silence_start: 5.5
[silencedetect @ 0x1] silence_end: 5.9 | silence_duration: 0.4
"""


class TestFindNearestSilenceMocked:
    @patch("siphon.cutter.subprocess.run")
    def test_snaps_to_middle_of_closest_silence(self, mock_run):
        # window starts at 100 - 3 = 97; silences are at 97+1.6=98.6 and 97+5.7=102.7
        mock_run.return_value = MagicMock(returncode=0, stderr=SILENCE_LOG)
        assert find_nearest_silence("/f.mp3", 100.0, window=3.0) == pytest.approx(98.6)

    @patch("siphon.cutter.subprocess.run")
    def test_no_silence_returns_target(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stderr="size=N/A time=00:00:06.00")
        assert find_nearest_silence("/f.mp3", 100.0) == 100.0

    @patch("siphon.cutter.subprocess.run")
    def test_silence_running_past_the_window_counts(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stderr="[silencedetect @ 0x1] silence_start: 4.0\n")
        # slice is 97..103 (6s), silence 4.0..6.0 -> middle 5.0 -> 102.0
        assert find_nearest_silence("/f.mp3", 100.0, window=3.0) == pytest.approx(102.0)

    @patch("siphon.cutter.subprocess.run", side_effect=OSError("ffmpeg missing"))
    def test_ffmpeg_failure_returns_target(self, mock_run):
        assert find_nearest_silence("/f.mp3", 100.0) == 100.0

    @patch("siphon.cutter.subprocess.run")
    def test_slice_is_clamped_at_the_start_of_the_file(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stderr="")
        find_nearest_silence("/f.mp3", 1.0, window=3.0)
        cmd = mock_run.call_args.args[0]
        assert cmd[cmd.index("-ss") + 1] == "0.0"


class TestCutSegmentsSnapping:
    @staticmethod
    def _keep_ranges(segments, total, snap_map=None, **kwargs):
        snap_map = snap_map or {}
        with patch("shutil.move"), \
             patch("siphon.cutter.validate_file", return_value=True), \
             patch("siphon.cutter.get_duration", return_value=total), \
             patch("siphon.cutter.find_nearest_silence", side_effect=lambda f, t, w: snap_map.get(t, t)), \
             patch("siphon.cutter.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stderr="")
            cut_segments("/tmp/test.mp3", segments, output_path="/tmp/out.mp3", **kwargs)
        out = []
        for call in mock_run.call_args_list:
            cmd = call.args[0]
            if "-ss" in cmd:
                out.append((float(cmd[cmd.index("-ss") + 1]), float(cmd[cmd.index("-to") + 1])))
        return out

    def test_off_by_default(self):
        ranges = self._keep_ranges([{"start": 100, "end": 130}], 600, snap_map={100: 99, 130: 131})
        assert ranges == [(0.0, 100.0), (130.0, 600.0)]

    def test_cut_points_move_to_the_silence(self):
        ranges = self._keep_ranges(
            [{"start": 100, "end": 130}], 600, snap_map={100: 99.4, 130: 131.2}, snap_to_silence=True,
        )
        assert ranges == [(0.0, 99.4), (131.2, 600.0)]

    def test_cut_near_the_start_is_extended_to_the_start(self):
        ranges = self._keep_ranges([{"start": 1.5, "end": 30}], 600, snap_map={30: 30.5}, snap_to_silence=True)
        assert ranges == [(30.5, 600.0)]

    def test_cut_near_the_end_is_extended_to_the_end(self):
        ranges = self._keep_ranges([{"start": 500, "end": 598.5}], 600, snap_map={500: 499.5}, snap_to_silence=True)
        assert ranges == [(0.0, 499.5)]

    def test_snapped_cuts_that_collide_are_merged(self):
        # snapping pulls these two ads together so they overlap; no sliver may remain
        ranges = self._keep_ranges(
            [{"start": 100, "end": 130}, {"start": 134, "end": 160}], 600,
            snap_map={130: 135, 134: 133}, snap_to_silence=True,
        )
        assert ranges == [(0.0, 100.0), (160.0, 600.0)]


class TestCutSegmentsApplied:
    """cut_segments reports what it really removed, for the Episodes page."""

    @staticmethod
    def _run(segments, applied, total=600.0, snap_map=None, **kwargs):
        snap_map = snap_map or {}
        with patch("shutil.move"), \
             patch("siphon.cutter.validate_file", return_value=True), \
             patch("siphon.cutter.get_duration", return_value=total), \
             patch("siphon.cutter.find_nearest_silence", side_effect=lambda f, t, w: snap_map.get(t, t)), \
             patch("siphon.cutter.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stderr="")
            cut_segments("/tmp/test.mp3", segments, output_path="/tmp/out.mp3", applied=applied, **kwargs)

    def test_reports_merged_ranges(self):
        applied = []
        self._run([{"start": 100, "end": 130}, {"start": 130.5, "end": 160}, {"start": 300, "end": 330}], applied)
        assert applied == [{"start": 100, "end": 160}, {"start": 300, "end": 330}]

    def test_reports_snapped_ranges(self):
        applied = []
        self._run([{"start": 100, "end": 130}], applied, snap_map={100: 99.4, 130: 131.2}, snap_to_silence=True)
        assert applied == [{"start": 99.4, "end": 131.2}]

    def test_left_alone_when_there_is_nothing_to_cut(self):
        applied = [{"start": 1, "end": 2}]
        assert cut_segments("/tmp/test.mp3", [], applied=applied) == "/tmp/test.mp3"
        assert applied == [{"start": 1, "end": 2}]

    def test_optional(self):
        self._run([{"start": 100, "end": 130}], None)


# ------------------------------------------------------------------ #
# Integration tests with real ffmpeg
# ------------------------------------------------------------------ #

@needs_ffmpeg
class TestCutterIntegration:
    def _create_test_audio(self, path: str, duration: float = 10.0) -> None:
        """Create a silent WAV test file."""
        subprocess.run([
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", f"anullsrc=r=44100:cl=mono",
            "-t", str(duration),
            path,
        ], capture_output=True, check=True)

    def test_extract_audio_from_wav(self, tmp_path):
        input_file = str(tmp_path / "input.wav")
        output_file = str(tmp_path / "output.wav")
        self._create_test_audio(input_file, 5.0)

        extract_audio(input_file, output_file)
        assert os.path.exists(output_file)
        assert os.path.getsize(output_file) > 0

    def test_get_duration(self, tmp_path):
        test_file = str(tmp_path / "test.wav")
        self._create_test_audio(test_file, 5.0)

        duration = get_duration(test_file)
        assert 4.5 <= duration <= 5.5  # Allow some tolerance

    def test_cut_segments_removes_middle(self, tmp_path):
        test_file = str(tmp_path / "test.wav")
        output_file = str(tmp_path / "output.wav")
        self._create_test_audio(test_file, 10.0)

        original_duration = get_duration(test_file)

        # Cut out 3 seconds from the middle
        cut_segments(
            test_file,
            [{"start": 3.0, "end": 6.0}],
            output_path=output_file,
        )

        new_duration = get_duration(output_file)
        # Should be roughly 7 seconds (10 - 3)
        assert 6.0 <= new_duration <= 8.0

    def test_cut_segments_overwrites_in_place(self, tmp_path):
        test_file = str(tmp_path / "test.wav")
        self._create_test_audio(test_file, 10.0)

        cut_segments(
            test_file,
            [{"start": 2.0, "end": 5.0}],
        )

        # Original file should now be shorter
        new_duration = get_duration(test_file)
        assert 6.0 <= new_duration <= 8.0

    def test_back_to_back_ads_are_removed_as_one_chunk(self, tmp_path):
        test_file = str(tmp_path / "test.wav")
        output_file = str(tmp_path / "output.wav")
        self._create_test_audio(test_file, 20.0)

        # two back-to-back ads, 2s-6s and 6s-10s, as the pipeline would pass them
        cut_segments(
            test_file,
            [{"start": 2.0, "end": 6.0}, {"start": 6.0, "end": 10.0}],
            output_path=output_file,
        )

        # 20s minus the whole 8s block
        assert 11.0 <= get_duration(output_file) <= 13.0

    def test_find_nearest_silence_finds_a_real_pause(self, tmp_path):
        test_file = str(tmp_path / "tone_with_gap.wav")
        # 20s of tone with a silent gap from 8s to 9s
        subprocess.run([
            "ffmpeg", "-y", "-f", "lavfi",
            "-i", "aevalsrc=if(between(t\\,8\\,9)\\,0\\,0.5*sin(2*PI*440*t)):s=16000:d=20",
            test_file,
        ], capture_output=True, check=True)

        snapped = find_nearest_silence(test_file, 8.8, window=3.0)
        assert 8.3 <= snapped <= 8.7  # middle of the 8-9s gap

        # no silence within the window: the target is returned unchanged
        assert find_nearest_silence(test_file, 15.0, window=2.0) == 15.0

    def test_cut_with_snapping_lands_in_the_pause(self, tmp_path):
        test_file = str(tmp_path / "tone_with_gap.wav")
        output_file = str(tmp_path / "out.wav")
        subprocess.run([
            "ffmpeg", "-y", "-f", "lavfi",
            "-i", "aevalsrc=if(between(t\\,8\\,9)\\,0\\,0.5*sin(2*PI*440*t)):s=16000:d=20",
            test_file,
        ], capture_output=True, check=True)

        # ad "starts" at 8.8 (inside the pause); snapping moves the cut to ~8.5
        cut_segments(
            test_file, [{"start": 8.8, "end": 14.0}], output_path=output_file,
            snap_to_silence=True, snap_window=3.0,
        )
        # kept: 0 to ~8.5 plus 14 to 20 (~14.5s), instead of 8.8 + 6 = 14.8s unsnapped
        assert 14.0 <= get_duration(output_file) <= 15.0
