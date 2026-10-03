"""Tests for the Reprocess button: restoring the original audio and re-queuing an episode."""

from __future__ import annotations

import json
import os
from unittest.mock import patch

import httpx
import pytest

from siphon.episode_files import original_path, restore_original, transcript_path
from siphon.pipeline import process_claude
from siphon.reprocess import ReprocessError, check_reprocess, reprocess_episode
from tests.test_episodes_page import AUDIT, TRANSCRIPT, app, client, config, seed  # noqa: F401  (fixtures)

NO_CUTS = {"segments": [], "high_confidence": [], "marginal": []}


def _post(client, mode, feed="show", id="ep1", **kwargs):
    return client.post("/ui/episode/reprocess", data={"feed": feed, "id": id, "mode": mode}, **kwargs)


def _row(app, id="ep1"):
    return app.state.db.get_episode(id, "show")


class TestRestoreOriginal:
    def test_replaces_the_episode_file_and_keeps_the_original(self, tmp_path):
        path = str(tmp_path / "ep.mp3")
        open(path, "wb").write(b"CLEANED")
        open(original_path(path), "wb").write(b"ORIGINAL")
        assert restore_original(path) is True
        assert open(path, "rb").read() == b"ORIGINAL"
        assert open(original_path(path), "rb").read() == b"ORIGINAL"
        assert not os.path.exists(path + ".restoring")

    def test_nothing_to_restore_when_the_file_is_the_original(self, tmp_path):
        path = str(tmp_path / "ep.mp3")
        open(path, "wb").write(b"ONLY")
        assert restore_original(path) is False
        assert open(path, "rb").read() == b"ONLY"

    def test_failure_leaves_no_staging_file_and_the_episode_untouched(self, tmp_path):
        path = str(tmp_path / "ep.mp3")
        open(path, "wb").write(b"CLEANED")
        open(original_path(path), "wb").write(b"ORIGINAL")
        with patch("siphon.episode_files.os.replace", side_effect=PermissionError("in use")):
            with pytest.raises(PermissionError):
                restore_original(path)
        assert open(path, "rb").read() == b"CLEANED"
        assert not os.path.exists(path + ".restoring")


class TestRedoAdDetection:
    async def test_restores_original_and_queues_claude(self, app, config, client):
        seed(app, config)
        resp = await _post(client, "detect")
        assert resp.status_code == 303
        assert resp.headers["location"].startswith("/ui/episode?feed=show&id=ep1&queued=detect")

        row = _row(app)
        assert row["status"] == "pending_claude"
        assert row["llm_trim_status"] == "pending"
        assert row["llm_cuts_applied"] == 0
        assert row["llm_segments_json"] is None
        assert row["file_size"] == len(b"ORIGINALAUDIO")
        assert open(row["file_path"], "rb").read() == b"ORIGINALAUDIO"          # restored
        assert open(original_path(row["file_path"]), "rb").read() == b"ORIGINALAUDIO"  # still kept
        assert os.path.exists(transcript_path(config.storage.download_dir, "show", "ep1"))  # reused

    async def test_htmx_gets_an_hx_redirect(self, app, config, client):
        seed(app, config)
        resp = await _post(client, "detect", headers={"HX-Request": "true"})
        assert "queued=detect" in resp.headers["hx-redirect"]

    async def test_no_cuts_and_no_kept_original_is_fine(self, app, config, client):
        file_path = seed(app, config, audit=NO_CUTS, original=False)
        assert (await _post(client, "detect")).status_code == 303
        assert _row(app)["status"] == "pending_claude"
        assert open(file_path, "rb").read() == b"CLEANAUDIO"   # already the original; left alone

    async def test_queued_notice_is_shown_after_the_redirect(self, app, config, client):
        seed(app, config)
        await _post(client, "detect")
        page = await client.get("/ui/episode", params={"feed": "show", "id": "ep1", "queued": "detect"})
        assert "Queued for ad detection again" in page.text
        assert "being processed: Waiting for ad detection (Claude)" in page.text
        assert "location.reload()" in page.text                  # refreshes itself while queued


class TestRetranscribe:
    async def test_discards_the_transcript_and_queues_whisper(self, app, config, client):
        seed(app, config)
        assert (await _post(client, "transcribe")).status_code == 303
        row = _row(app)
        assert row["status"] == "pending_whisper"
        assert open(row["file_path"], "rb").read() == b"ORIGINALAUDIO"
        assert not os.path.exists(transcript_path(config.storage.download_dir, "show", "ep1"))

    async def test_works_without_a_kept_transcript(self, app, config, client):
        seed(app, config, transcript=None)
        assert (await _post(client, "transcribe")).status_code == 303
        assert _row(app)["status"] == "pending_whisper"


class TestRefusals:
    async def _refused(self, app, client, mode, expected_text):
        before = dict(_row(app))
        resp = await _post(client, mode)
        assert resp.status_code == 400
        assert expected_text in resp.text
        after = _row(app)
        assert (after["status"], after["llm_segments_json"], after["llm_cuts_applied"]) == (
            before["status"], before["llm_segments_json"], before["llm_cuts_applied"])
        return after

    async def test_detect_needs_the_transcript(self, app, config, client):
        seed(app, config, transcript=None)
        await self._refused(app, client, "detect", "transcript was not kept")

    async def test_needs_the_original_when_ads_were_cut(self, app, config, client):
        file_path = seed(app, config, original=False)
        await self._refused(app, client, "detect", "original audio was not kept")
        await self._refused(app, client, "transcribe", "original audio was not kept")
        assert open(file_path, "rb").read() == b"CLEANAUDIO"

    async def test_not_while_the_episode_is_being_processed(self, app, config, client):
        seed(app, config)
        app.state.db.update_episode_status("ep1", "show", "pending_claude")
        await self._refused(app, client, "detect", "still being processed")

    async def test_not_when_ad_removal_is_off_for_the_feed(self, app, config, client):
        seed(app, config)
        config.feeds[0].llm_trim = False
        await self._refused(app, client, "detect", "llm_trim")

    async def test_not_when_the_feed_is_gone(self, app, config, client):
        seed(app, config)
        config.feeds.clear()
        await self._refused(app, client, "detect", "no longer in the config")

    async def test_unknown_mode(self, app, config, client):
        seed(app, config)
        await self._refused(app, client, "explode", "Unknown")

    async def test_failed_restore_changes_nothing(self, app, config, client):
        file_path = seed(app, config)
        with patch("siphon.episode_files.os.replace", side_effect=PermissionError("file in use")):
            resp = await _post(client, "detect")
        assert resp.status_code == 400 and "Could not restore" in resp.text
        assert _row(app)["status"] == "done"
        assert open(file_path, "rb").read() == b"CLEANAUDIO"

    async def test_unknown_episode_is_404(self, client):
        assert (await _post(client, "detect", id="nope")).status_code == 404

    async def test_remote_clients_are_refused(self, app, config):
        seed(app, config)
        transport = httpx.ASGITransport(app=app, client=("203.0.113.9", 4321))
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as remote:
            assert (await _post(remote, "detect")).status_code == 403
        assert _row(app)["status"] == "done"


class TestPageButtons:
    async def test_enabled_when_everything_was_kept(self, app, config, client):
        seed(app, config)
        text = (await client.get("/ui/episode", params={"feed": "show", "id": "ep1"})).text
        assert "Redo ad detection" in text and "Re-transcribe and redo" in text
        assert "disabled" not in text.split("Process again")[1].split("What was cut")[0]

    async def test_disabled_with_a_reason_when_not_possible(self, app, config, client):
        seed(app, config, original=False, transcript=None)
        text = (await client.get("/ui/episode", params={"feed": "show", "id": "ep1"})).text
        card = text.split("Process again")[1].split("What was cut")[0]
        assert card.count("disabled") == 2
        assert "original audio was not kept" in card

    async def test_no_auto_refresh_when_idle(self, app, config, client):
        seed(app, config)
        text = (await client.get("/ui/episode", params={"feed": "show", "id": "ep1"})).text
        assert "location.reload()" not in text and "being processed" not in text


class TestEndToEnd:
    async def test_the_claude_worker_cuts_the_restored_original_again(self, app, config, client):
        file_path = seed(app, config)
        queued = await _post(client, "detect")
        assert queued.status_code == 303, queued.text[-600:]
        seen = {}

        def fake_cut(path, segments, **kwargs):
            seen["audio_at_cut_time"] = open(path, "rb").read()
            open(path, "wb").write(b"RECUT")
            kwargs["applied"][:] = [{"start": 100.0, "end": 160.0}]
            return path

        new_ads = {"segments": [{"start": 100.0, "end": 160.0, "type": "ad", "label": "new-sponsor", "confidence": 0.95}]}
        with patch("siphon.ad_detect.detect_ads", return_value=new_ads), patch("siphon.cutter.cut_segments", side_effect=fake_cut):
            await process_claude(config, app.state.db)

        row = _row(app)
        assert row["status"] == "done" and row["llm_trim_status"] == "done"
        assert seen["audio_at_cut_time"] == b"ORIGINALAUDIO"              # cut from the original, not the old cut
        assert open(file_path, "rb").read() == b"RECUT"
        assert open(original_path(file_path), "rb").read() == b"ORIGINALAUDIO"
        audit = json.loads(row["llm_segments_json"])
        assert audit["high_confidence"][0]["label"] == "new-sponsor"
        assert audit["applied_cuts"] == [{"start": 100.0, "end": 160.0}]
