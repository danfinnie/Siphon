"""Tests for siphon.inspector and siphon.episode_files — the logic behind the Episodes page."""

from __future__ import annotations

import json
import os

import pytest

from siphon import inspector as ins
from siphon.episode_files import original_path, remove_extras, transcript_path


def _seg(start, end, label="x", confidence=0.9, **extra):
    return {"start": start, "end": end, "label": label, "confidence": confidence, **extra}


AUDIT = {
    "segments": [
        _seg(10, 40, "Acme", 0.95), _seg(40, 70, "Globex", 0.9),
        _seg(200, 230, "maybe-an-ad", 0.6), _seg(300, 304, "tiny", 0.9),
    ],
    "high_confidence": [_seg(10, 70, "Acme + Globex", 0.95, merged_from=2)],
    "marginal": [_seg(200, 230, "maybe-an-ad", 0.6)],
    "applied_cuts": [{"start": 9.4, "end": 70.6}],
    "original_duration": 600.0,
    "final_duration": 538.8,
}


class TestParseAudit:
    @pytest.mark.parametrize("raw", [None, "", "not json", "[1, 2]", "null", "{}"])
    def test_unusable_input_gives_empty_audit(self, raw):
        assert ins.parse_audit(raw) == {}

    def test_valid_json(self):
        assert ins.parse_audit(json.dumps(AUDIT))["original_duration"] == 600.0


class TestRemovedRanges:
    def test_prefers_applied_cuts_over_detected_segments(self):
        assert ins.removed_ranges(AUDIT) == [(9.4, 70.6)]

    def test_falls_back_to_high_confidence_for_older_episodes(self):
        audit = {"high_confidence": [_seg(10, 70)]}
        assert ins.removed_ranges(audit) == [(10.0, 70.0)]

    def test_sorts_and_merges_overlaps(self):
        audit = {"applied_cuts": [{"start": 50, "end": 90}, {"start": 0, "end": 60}, {"start": 200, "end": 210}]}
        assert ins.removed_ranges(audit) == [(0.0, 90.0), (200.0, 210.0)]

    def test_ignores_malformed_entries(self):
        audit = {"applied_cuts": [{"start": "a"}, {"start": 5, "end": 5}, "junk", {"start": 1, "end": 4}]}
        assert ins.removed_ranges(audit) == [(1.0, 4.0)]

    def test_empty(self):
        assert ins.removed_ranges({}) == []


class TestBuildCutList:
    def test_label_confidence_and_parts_come_from_the_detected_chunk(self):
        (cut,) = ins.build_cut_list(AUDIT)
        assert cut["label"] == "Acme + Globex"
        assert cut["confidence"] == 0.95
        assert cut["parts"] == 1
        assert (cut["start"], cut["end"]) == (9.4, 70.6)
        assert cut["duration"] == pytest.approx(61.2)

    def test_clean_at_accounts_for_earlier_cuts(self):
        audit = {"applied_cuts": [{"start": 10, "end": 40}, {"start": 100, "end": 130}, {"start": 500, "end": 560}]}
        cuts = ins.build_cut_list(audit)
        assert [c["clean_at"] for c in cuts] == [10, 70, 440]
        assert [c["index"] for c in cuts] == [1, 2, 3]

    def test_cut_without_a_matching_label_still_listed(self):
        (cut,) = ins.build_cut_list({"applied_cuts": [{"start": 5, "end": 25}]})
        assert cut["label"] == "cut"
        assert cut["confidence"] is None

    def test_several_detected_chunks_in_one_cut_are_joined(self):
        audit = {
            "applied_cuts": [{"start": 0, "end": 100}],
            "high_confidence": [_seg(0, 40, "A", 0.8), _seg(40, 100, "B", 0.9), _seg(500, 520, "elsewhere")],
        }
        (cut,) = ins.build_cut_list(audit)
        assert cut["label"] == "A + B"
        assert cut["parts"] == 2
        assert cut["confidence"] == 0.9


class TestClassifySegments:
    LINES = [
        {"start": 0, "end": 5, "text": "Welcome to the show"},
        {"start": 8, "end": 10.2, "text": "and now a word"},       # 0.8s of 2.2s inside the cut -> edge
        {"start": 20, "end": 25, "text": "This episode is brought to you by Acme"},
        {"start": 70, "end": 72, "text": "back to the show"},       # 0.6s of 2s inside the cut -> edge
        {"start": 100, "end": 105, "text": "regular content"},
        {"start": 200, "end": 205, "text": "check out my other thing"},
        {"start": 300, "end": 302, "text": "tiny flagged line"},
    ]

    def _by_start(self):
        return {r["start"]: r for r in ins.classify_segments(self.LINES, AUDIT)}

    def test_statuses(self):
        got = {start: row["status"] for start, row in self._by_start().items()}
        assert got == {0: "content", 8: "edge", 20: "cut", 70: "edge", 100: "content", 200: "possible", 300: "detected"}

    def test_cut_lines_carry_the_cut_label_and_number(self):
        row = self._by_start()[20]
        assert row["label"] == "Acme + Globex"
        assert row["cut_index"] == 1

    def test_flag_labels(self):
        rows = self._by_start()
        assert rows[200]["label"] == "maybe-an-ad"
        assert rows[300]["label"] == "tiny"
        assert rows[0]["label"] == ""

    def test_no_audit_means_everything_is_content(self):
        rows = ins.classify_segments(self.LINES, {})
        assert {r["status"] for r in rows} == {"content"}

    def test_unusable_lines_are_skipped(self):
        rows = ins.classify_segments([{"start": "x", "end": 1}, {"start": 5, "end": 5}, {"start": 1, "end": 3, "text": "ok"}], {})
        assert [r["text"] for r in rows] == ["ok"]


class TestFlaggedNotCut:
    def test_lists_uncut_segments_with_reasons(self):
        rows = ins.flagged_not_cut(AUDIT)
        assert [(r["label"], r["reason"]) for r in rows] == [
            ("maybe-an-ad", "confidence below the cut threshold"),
            ("tiny", "shorter than the minimum length"),
        ]

    def test_cut_segments_are_not_listed(self):
        assert all(r["label"] not in ("Acme", "Globex") for r in ins.flagged_not_cut(AUDIT))

    def test_low_confidence(self):
        audit = {"segments": [_seg(0, 60, "weak", 0.3)]}
        assert ins.flagged_not_cut(audit)[0]["reason"] == "confidence too low"

    def test_empty(self):
        assert ins.flagged_not_cut({}) == []


class TestSummarize:
    def test_uses_recorded_durations(self):
        cuts = ins.build_cut_list(AUDIT)
        s = ins.summarize(AUDIT, {}, cuts)
        assert s["original_duration"] == 600.0
        assert s["final_duration"] == 538.8
        assert s["removed"] == pytest.approx(61.2)
        assert s["percent"] == pytest.approx(10.2)
        assert s["cut_count"] == 1

    def test_derives_missing_durations_from_the_cuts(self):
        audit = {"applied_cuts": [{"start": 0, "end": 60}], "final_duration": 540.0}
        s = ins.summarize(audit, {}, ins.build_cut_list(audit))
        assert s["original_duration"] == 600.0

    def test_nothing_known(self):
        s = ins.summarize({}, {}, [])
        assert s["original_duration"] is None and s["percent"] is None and s["cut_count"] == 0


class TestHelpers:
    @pytest.mark.parametrize("seconds,expected", [
        (0, "0:00"), (59.4, "0:59"), (61, "1:01"), (3599.6, "1:00:00"), (3725, "1:02:05"), (None, "-"), ("x", "-"), (-5, "0:00"),
    ])
    def test_format_clock(self, seconds, expected):
        assert ins.format_clock(seconds) == expected

    def test_clean_time_map(self):
        cuts = ins.build_cut_list({"applied_cuts": [{"start": 10, "end": 40}]})
        assert ins.clean_time_map(cuts) == [{"start": 10, "end": 40, "clean_at": 10, "chime": 0.0}]

    def test_load_transcript_segments(self, tmp_path):
        good = tmp_path / "t.json"
        good.write_text(json.dumps({"segments": [{"start": 0, "end": 1, "text": "hi"}]}))
        assert ins.load_transcript_segments(str(good))[0]["text"] == "hi"
        assert ins.load_transcript_segments(str(tmp_path / "missing.json")) is None
        bad = tmp_path / "bad.json"
        bad.write_text("not json")
        assert ins.load_transcript_segments(str(bad)) is None
        empty = tmp_path / "empty.json"
        empty.write_text(json.dumps({"segments": []}))
        assert ins.load_transcript_segments(str(empty)) is None


class TestEpisodeFiles:
    def test_original_path_sits_beside_the_episode(self):
        assert original_path("/m/feed/abc.mp3") == "/m/feed/abc.orig.mp3"
        assert original_path("/m/feed/some.name.mp4") == "/m/feed/some.name.orig.mp4"

    def test_transcript_path_sanitises_the_id(self):
        path = transcript_path("/m", "feed", "https://x.com/ep/1?a=b")
        assert path == os.path.join("/m", "feed", "https___x_com_ep_1_a_b_transcript.json")

    def test_remove_extras_deletes_original_and_transcripts(self, tmp_path):
        feed_dir = tmp_path / "feed"
        feed_dir.mkdir()
        episode_file = feed_dir / "ep1.mp3"
        episode_file.write_bytes(b"clean")
        kept = [feed_dir / "ep1.orig.mp3", feed_dir / "ep1_transcript.json"]
        for f in kept:
            f.write_bytes(b"x")
        remove_extras(str(tmp_path), {"feed_name": "feed", "video_id": "ep1", "file_path": str(episode_file)})
        assert not any(f.exists() for f in kept)
        assert episode_file.exists()  # remove_extras never touches the episode itself

    def test_remove_extras_tolerates_missing_files(self, tmp_path):
        remove_extras(str(tmp_path), {"feed_name": "feed", "video_id": "none", "file_path": None})


class TestChimes:
    AUDIT = {"applied_cuts": [{"start": 10, "end": 40}, {"start": 100, "end": 130}, {"start": 500, "end": 560}], "chime_seconds": 1.7}

    def test_chime_seconds(self):
        assert ins.chime_seconds(self.AUDIT) == 1.7
        assert ins.chime_seconds({}) == 0.0
        assert ins.chime_seconds({"chime_seconds": "junk"}) == 0.0
        assert ins.chime_seconds({"chime_seconds": -4}) == 0.0

    def test_each_earlier_cut_leaves_a_chime_in_the_cleaned_timeline(self):
        cuts = ins.build_cut_list(self.AUDIT)
        # without chimes: [10, 70, 440]; each chime inserted at an earlier cut pushes later splices out by 1.7s
        assert [round(c["clean_at"], 2) for c in cuts] == [10.0, 71.7, 443.4]
        assert all(c["chime"] == 1.7 for c in cuts)

    def test_no_chimes_means_the_old_timeline(self):
        audit = {k: v for k, v in self.AUDIT.items() if k != "chime_seconds"}
        assert [c["clean_at"] for c in ins.build_cut_list(audit)] == [10, 70, 440]

    def test_time_map_carries_the_chime_length(self):
        mapped = ins.clean_time_map(ins.build_cut_list(self.AUDIT))
        assert [m["chime"] for m in mapped] == [1.7, 1.7, 1.7]

    def test_cleaned_length_is_consistent_with_the_map(self):
        # original 600s, 120s of ads removed, 3 chimes added
        cuts = ins.build_cut_list(self.AUDIT)
        removed = sum(c["duration"] for c in cuts)
        assert removed == 120
        last = cuts[-1]
        # the splice of the last cut, plus its chime, plus the rest of the episode, is the cleaned length
        assert last["clean_at"] + last["chime"] + (600 - 560) == pytest.approx(600 - removed + 3 * 1.7)
