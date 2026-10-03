"""Tests for the pages that offer downloading an episode again, and the list of every episode."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

from siphon.config import FeedConfig
from tests.test_episodes_page import app, client, config, seed  # noqa: F401  (fixtures)


def _page(client, id="ep1", **extra):
    return client.get("/ui/episode", params={"feed": "show", "id": id, **extra})


class TestPages:
    async def test_third_button_on_a_finished_episode(self, app, config, client):
        seed(app, config)
        text = (await _page(client)).text
        assert "Re-download and redo everything" in text
        card = text.split("Process again")[1].split("What was cut")[0]
        assert "disabled" not in card

    async def test_third_button_is_disabled_for_youtube_feeds(self, app, config, client):
        seed(app, config)
        config.feeds[0] = FeedConfig(name="show", url="https://www.youtube.com/@x", type="youtube", llm_trim=True)
        card = (await _page(client)).text.split("Process again")[1].split("What was cut")[0]
        assert "only available for podcast feeds" in card

    async def test_filtered_episode_offers_download_and_process(self, app, config, client):
        app.state.db.insert_episode(video_id="f1", feed_name="show", title="Filtered ep", upload_date="20260929",
                                    status="filtered", filter_reason="too_short")
        text = (await _page(client, id="f1")).text
        assert "Download and process" in text and "<strong>filtered</strong> (too_short)" in text
        assert "being processed" not in text
        assert "is filtered, so there is no processed copy" in text      # the first two buttons explain why not
        assert "This episode has no processed audio" in text             # not "processed before originals were kept"
        assert "has not been processed" in text
        assert "processed before" not in text

    async def test_waiting_banner_and_auto_refresh(self, app, config, client):
        seed(app, config)
        app.state.db.update_episode_status("ep1", "show", "done", redownload_requested=1)
        text = (await _page(client)).text
        assert "Waiting to be downloaded again" in text and "stays in your feed" in text
        assert "location.reload()" in text

    async def test_failed_attempt_is_shown(self, app, config, client):
        seed(app, config)
        app.state.db.update_episode_status("ep1", "show", "done", error="The feed no longer lists this episode. Your existing copy was left as it was.")
        text = (await _page(client)).text
        assert "flash-error" in text and "left as it was" in text

    async def test_post_queues_the_download_and_starts_the_worker(self, app, config, client):
        seed(app, config)
        with patch("siphon.pipeline.process_downloads", new=AsyncMock()) as worker:
            resp = await client.post("/ui/episode/reprocess", data={"feed": "show", "id": "ep1", "mode": "redownload"})
            await asyncio.sleep(0.05)
        assert resp.status_code == 303 and "queued=redownload" in resp.headers["location"]
        assert app.state.db.get_episode("ep1", "show")["redownload_requested"] == 1
        worker.assert_awaited_once()

    async def test_post_is_refused_when_already_queued(self, app, config, client):
        seed(app, config)
        app.state.db.update_episode_status("ep1", "show", "done", redownload_requested=1)
        with patch("siphon.pipeline.process_downloads", new=AsyncMock()) as worker:
            resp = await client.post("/ui/episode/reprocess", data={"feed": "show", "id": "ep1", "mode": "redownload"})
            await asyncio.sleep(0.05)
        assert resp.status_code == 400 and "already waiting" in resp.text
        worker.assert_not_awaited()

    async def test_queued_notice_after_the_redirect(self, app, config, client):
        seed(app, config)
        assert "Queued to be downloaded again" in (await _page(client, queued="redownload")).text


class TestEpisodeList:
    async def _seed_mixed(self, app, config):
        seed(app, config)
        db = app.state.db
        db.insert_episode(video_id="f1", feed_name="show", title="A filtered one", upload_date="20260929",
                          status="filtered", filter_reason="too_short")
        db.insert_episode(video_id="p1", feed_name="show", title="A pruned one", upload_date="20260915", status="pruned")

    async def test_default_view_only_lists_processed_episodes(self, app, config, client):
        await self._seed_mixed(app, config)
        text = (await client.get("/ui/episodes")).text
        assert "The Big Episode" in text and "A filtered one" not in text and "A pruned one" not in text
        assert 'name="show" value="all"' in text and "checked" not in text.split('name="show"')[1].split(">")[0]

    async def test_every_episode_view_lists_all_statuses_with_dates(self, app, config, client):
        await self._seed_mixed(app, config)
        text = (await client.get("/ui/episodes", params={"show": "all"})).text
        for title in ("The Big Episode", "A filtered one", "A pruned one"):
            assert title in text
        assert "Published" in text and "2026-09-29" in text and "2026-09-15" in text
        assert ">filtered<" in text and ">pruned<" in text
        assert "<th>Status</th>" in text
        assert 'id="episode-search"' in text and "checked" in text.split('name="show"')[1].split(">")[0]

    async def test_every_episode_view_respects_the_feed_filter(self, app, config, client):
        await self._seed_mixed(app, config)
        text = (await client.get("/ui/episodes", params={"show": "all", "feed": "other"})).text
        assert "A filtered one" not in text and "No episodes found" in text

    async def test_links_lead_to_the_episode_pages(self, app, config, client):
        await self._seed_mixed(app, config)
        text = (await client.get("/ui/episodes", params={"show": "all"})).text
        assert "/ui/episode?feed=show&amp;id=f1" in text
