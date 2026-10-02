"""Tests for siphon.cutter — ffmpeg-based segment cutting."""

from __future__ import annotations

import math
import os
import struct
import subprocess
from unittest.mock import patch, MagicMock

import pytest

from siphon.cutter import (
    audio_stream_info, cut_segments, extract_audio, find_nearest_silence, get_duration,
    has_real_video_stream, make_chime,
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


MP3_INFO = {"codec": "mp3", "sample_rate": 44100, "channels": 2, "has_video": False}


class TestChimeInsertionMocked:
    """Where the chimes go in the list of pieces that ffmpeg joins."""

    @staticmethod
    def _run(segments, total=600.0, info=MP3_INFO, chime_length=1.7, **kwargs):
        lists, commands = [], []

        def fake_run(cmd, *args, **kw):
            commands.append(cmd)
            if "-f" in cmd and cmd[cmd.index("-f") + 1] == "concat":
                with open(cmd[cmd.index("-i") + 1]) as f:
                    lists.append([os.path.basename(line.split("'")[1]) for line in f.read().splitlines()])
            return MagicMock(returncode=0, stderr="", stdout="")

        report = {}
        with patch("shutil.move"), \
             patch("siphon.cutter.validate_file", return_value=True), \
             patch("siphon.cutter.get_duration", return_value=total), \
             patch("siphon.cutter.audio_stream_info", return_value=info), \
             patch("siphon.cutter.make_chime", return_value=chime_length) as mock_chime, \
             patch("siphon.cutter.subprocess.run", side_effect=fake_run):
            cut_segments("/tmp/test.mp3", segments, output_path="/tmp/out.mp3", report=report, **kwargs)
        return {"order": lists[0], "report": report, "commands": commands, "make_chime": mock_chime}

    @staticmethod
    def _shape(order):
        return ["chime" if name.startswith("chime") else "piece" for name in order]

    def test_chime_between_pieces_at_a_middle_cut(self):
        out = self._run([{"start": 100, "end": 130}], chime=True)
        assert self._shape(out["order"]) == ["piece", "chime", "piece"]
        assert out["report"] == {"chime_seconds": 1.7, "chime_count": 1}

    def test_one_chime_per_cut(self):
        out = self._run([{"start": 100, "end": 130}, {"start": 300, "end": 330}], chime=True)
        assert self._shape(out["order"]) == ["piece", "chime", "piece", "chime", "piece"]
        assert out["report"]["chime_count"] == 2

    def test_ads_cut_together_get_a_single_chime(self):
        # back-to-back and near-touching ads merge into one cut, so one chime
        out = self._run([{"start": 100, "end": 130}, {"start": 130, "end": 160}, {"start": 160.5, "end": 190}], chime=True)
        assert self._shape(out["order"]) == ["piece", "chime", "piece"]
        assert out["report"]["chime_count"] == 1

    def test_chime_at_the_start_when_the_episode_opens_with_an_ad(self):
        out = self._run([{"start": 0, "end": 30}], chime=True)
        assert self._shape(out["order"]) == ["chime", "piece"]

    def test_chime_at_the_end_when_the_episode_closes_with_an_ad(self):
        out = self._run([{"start": 570, "end": 600}], chime=True)
        assert self._shape(out["order"]) == ["piece", "chime"]

    def test_chimes_at_both_ends_and_the_middle(self):
        out = self._run([{"start": 0, "end": 20}, {"start": 300, "end": 330}, {"start": 580, "end": 600}], chime=True)
        assert self._shape(out["order"]) == ["chime", "piece", "chime", "piece", "chime"]

    def test_same_chime_file_is_reused(self):
        out = self._run([{"start": 100, "end": 130}, {"start": 300, "end": 330}], chime=True)
        chimes = [n for n in out["order"] if n.startswith("chime")]
        assert len(chimes) == 2 and len(set(chimes)) == 1
        out["make_chime"].assert_called_once()

    def test_off_by_default(self):
        out = self._run([{"start": 100, "end": 130}])
        assert self._shape(out["order"]) == ["piece", "piece"]
        assert out["report"] == {}
        out["make_chime"].assert_not_called()

    def test_volume_is_passed_on(self):
        out = self._run([{"start": 100, "end": 130}], chime=True, chime_volume=0.3)
        assert out["make_chime"].call_args.args[2] == 0.3

    @pytest.mark.parametrize("info", [
        None,
        {"codec": "pcm_s16le", "sample_rate": 16000, "channels": 1, "has_video": False},
        {"codec": "aac", "sample_rate": 44100, "channels": 2, "has_video": True},     # a video file
        {"codec": "mp3", "sample_rate": 0, "channels": 2, "has_video": False},
    ])
    def test_unsupported_files_are_cut_without_a_chime(self, info):
        out = self._run([{"start": 100, "end": 130}], info=info, chime=True)
        assert self._shape(out["order"]) == ["piece", "piece"]
        assert out["report"] == {}
        out["make_chime"].assert_not_called()

    def test_a_failed_chime_does_not_fail_the_cut(self):
        out = self._run([{"start": 100, "end": 130}], chime=True, chime_length=None)
        assert self._shape(out["order"]) == ["piece", "piece"]
        assert out["report"] == {}

    def test_only_the_audio_stream_is_kept_when_chiming(self):
        out = self._run([{"start": 100, "end": 130}], chime=True)
        extractions = [c for c in out["commands"] if "-ss" in c]
        assert extractions and all(c[c.index("-map") + 1] == "0:a:0" for c in extractions)

    def test_streams_are_left_alone_without_a_chime(self):
        out = self._run([{"start": 100, "end": 130}])
        assert not any("-map" in c for c in out["commands"] if "-ss" in c)


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


def _tone_power(path, start, length, freq):
    """Power of one frequency in a slice of the file (Goertzel); decodes with ffmpeg."""
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", str(start), "-t", str(length), "-i", path,
         "-ac", "1", "-ar", "8000", "-f", "s16le", "-"], capture_output=True, check=True).stdout
    xs = [v / 32768 for v in struct.unpack("<%dh" % (len(raw) // 2), raw)]
    n = len(xs)
    k = round(n * freq / 8000)
    coeff = 2 * math.cos(2 * math.pi * k / n)
    s1 = s2 = 0.0
    for x in xs:
        s1, s2 = x + coeff * s1 - s2, s1
    return (s1 * s1 + s2 * s2 - coeff * s1 * s2) / n / n


@needs_ffmpeg
class TestChimeWithRealFfmpeg:
    TWO_ADS_BACK_TO_BACK_AND_ONE_MORE = [
        {"start": 10, "end": 15}, {"start": 15, "end": 18}, {"start": 25, "end": 28},
    ]

    @staticmethod
    def _tone(tmp_path, name, args, rate=44100, channels=2, seconds=40, cover=False):
        path = str(tmp_path / name)
        plain = str(tmp_path / f"plain_{name}") if cover else path
        subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i", f"sine=frequency=440:sample_rate={rate}:duration={seconds}",
                        "-ac", str(channels), *args, plain], capture_output=True, check=True)
        if cover:
            png = str(tmp_path / "cover.png")
            subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=red:s=300x300:d=1", "-frames:v", "1", png],
                           capture_output=True, check=True)
            subprocess.run(["ffmpeg", "-y", "-i", plain, "-i", png, "-map", "0:0", "-map", "1:0", "-c", "copy",
                            "-id3v2_version", "3", path], capture_output=True, check=True)
        return path

    def _cut(self, tmp_path, path, segments, **kwargs):
        out, report = str(tmp_path / ("out_" + os.path.basename(path))), {}
        cut_segments(path, segments, output_path=out, chime=True, report=report, **kwargs)
        return out, report

    @staticmethod
    def _chime_level(out, splice):
        """Power of the chime's two notes in the first half second after a splice."""
        return _tone_power(out, splice + 0.2, 0.5, 880) + _tone_power(out, splice + 0.2, 0.5, 1318.5)

    def test_chime_is_audible_at_each_splice_and_ads_in_a_row_get_one(self, tmp_path):
        path = self._tone(tmp_path, "ep.mp3", ["-c:a", "libmp3lame", "-b:a", "128k"])
        out, report = self._cut(tmp_path, path, self.TWO_ADS_BACK_TO_BACK_AND_ONE_MORE)

        chime = report["chime_seconds"]
        assert report["chime_count"] == 2                        # 10-18 is one cut, 25-28 another
        assert get_duration(out) == pytest.approx(40 - 8 - 3 + 2 * chime, abs=0.3)

        episode_tone = _tone_power(out, 4, 0.5, 440)             # before the first cut: just the episode
        assert episode_tone > 5e-4
        assert self._chime_level(out, 4) < episode_tone / 100    # and no chime there
        # first splice, at 10s: the episode tone gives way to the chime...
        assert _tone_power(out, 10.2, 0.5, 440) < episode_tone / 50
        assert self._chime_level(out, 10) > episode_tone / 10
        # ...and the episode carries on after it
        assert _tone_power(out, 10 + chime + 1, 0.5, 440) > episode_tone / 2
        # second splice: 7s of episode (18s-25s) after the first chime
        second = 10 + chime + (25 - 18)
        assert self._chime_level(out, second) > episode_tone / 10
        assert _tone_power(out, second + 0.2, 0.5, 440) < episode_tone / 50

    def test_chime_at_the_very_start(self, tmp_path):
        path = self._tone(tmp_path, "ep.mp3", ["-c:a", "libmp3lame", "-b:a", "128k"])
        out, report = self._cut(tmp_path, path, [{"start": 0, "end": 6}])
        assert report["chime_count"] == 1
        episode_tone = _tone_power(out, report["chime_seconds"] + 2, 0.5, 440)
        assert self._chime_level(out, 0) > episode_tone / 10
        assert get_duration(out) == pytest.approx(34 + report["chime_seconds"], abs=0.3)

    @pytest.mark.parametrize("rate,channels,bitrate", [(22050, 1, "64k"), (48000, 2, "128k")])
    def test_other_mp3_sample_rates_and_channels(self, tmp_path, rate, channels, bitrate):
        path = self._tone(tmp_path, "ep.mp3", ["-c:a", "libmp3lame", "-b:a", bitrate], rate=rate, channels=channels)
        out, report = self._cut(tmp_path, path, [{"start": 12, "end": 20}])
        info = audio_stream_info(out)
        assert (info["sample_rate"], info["channels"]) == (rate, channels)   # the chime matched the episode
        assert get_duration(out) == pytest.approx(32 + report["chime_seconds"], abs=0.3)

    def test_aac_in_m4a(self, tmp_path):
        path = self._tone(tmp_path, "ep.m4a", ["-c:a", "aac", "-b:a", "96k"])
        out, report = self._cut(tmp_path, path, self.TWO_ADS_BACK_TO_BACK_AND_ONE_MORE)
        assert report["chime_count"] == 2
        assert audio_stream_info(out)["codec"] == "aac"
        assert get_duration(out) == pytest.approx(29 + 2 * report["chime_seconds"], abs=0.4)

    def test_mp3_with_cover_art_still_cuts_and_chimes(self, tmp_path):
        path = self._tone(tmp_path, "cover.mp3", ["-c:a", "libmp3lame", "-b:a", "128k"], cover=True)
        out, report = self._cut(tmp_path, path, [{"start": 10, "end": 18}])
        assert report["chime_count"] == 1
        pictures = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v", "-show_entries", "stream=codec_name",
                                   "-of", "csv=p=0", out], capture_output=True, text=True).stdout.split()
        assert pictures == []                                    # embedded art is left out when chiming

    def test_wav_is_cut_without_a_chime(self, tmp_path):
        path = self._tone(tmp_path, "ep.wav", ["-c:a", "pcm_s16le"], rate=16000, channels=1)
        out, report = self._cut(tmp_path, path, [{"start": 10, "end": 18}])
        assert report == {}
        assert get_duration(out) == pytest.approx(32, abs=0.2)

    def test_make_chime_matches_the_requested_format(self, tmp_path):
        path = str(tmp_path / "chime.mp3")
        length = make_chime(path, {"codec": "mp3", "sample_rate": 32000, "channels": 1}, volume=0.5)
        assert 1.6 <= length <= 1.9
        info = audio_stream_info(path)
        assert (info["codec"], info["sample_rate"], info["channels"]) == ("mp3", 32000, 1)

    def test_chime_volume_is_clamped_so_it_cannot_clip(self, tmp_path):
        quiet, loud = str(tmp_path / "q.mp3"), str(tmp_path / "l.mp3")
        info = {"codec": "mp3", "sample_rate": 44100, "channels": 1}
        assert make_chime(quiet, info, volume=-3) is not None
        assert make_chime(loud, info, volume=50) is not None      # no error; 50 is treated as 0.9
        assert _tone_power(loud, 0.4, 0.5, 880) > _tone_power(quiet, 0.4, 0.5, 880)
