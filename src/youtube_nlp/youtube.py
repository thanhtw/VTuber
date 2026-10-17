"""Public video metadata and paginated top-level comments via YouTube Data API v3."""
from dataclasses import replace
import logging
import time
import requests

log = logging.getLogger(__name__)


class YouTubeError(RuntimeError):
    def __init__(self, status, reason):
        self.status, self.reason = status, reason
        super().__init__(f"YouTube API error ({status}): {reason}")


class YouTubeClient:
    def __init__(self, api_key):
        self.api_key = api_key
        self.session = requests.Session()

    def close(self):
        self.session.close()

    def _get(self, resource, **params):
        for attempt in range(1, 4):
            log.info("Requesting YouTube %s (attempt %d/3; connect timeout 5s, read timeout 20s)",
                     resource, attempt)
            try:
                response = self.session.get(
                    f"https://www.googleapis.com/youtube/v3/{resource}",
                    params={**params, "key": self.api_key}, timeout=(5, 20))
            except (requests.Timeout, requests.ConnectionError):
                if attempt == 3:
                    raise
                log.warning("YouTube %s connection failed or timed out; retrying", resource)
            else:
                if response.status_code not in {429, 500, 502, 503, 504} or attempt == 3:
                    break
                log.warning("YouTube %s returned HTTP %s; retrying", resource, response.status_code)
                response.close()
            time.sleep(attempt)
        if not response.ok:
            try:
                errors = response.json().get("error", {}).get("errors", [])
                reason = errors[0].get("reason", "unknown") if errors else "unknown"
            except (ValueError, KeyError, TypeError):
                reason = "invalidResponse"
            raise YouTubeError(response.status_code, reason)
        return response.json()

    def collect(self, config):
        videos_by_id, comments_by_id = {}, {}
        self._collected_comment_videos = set()
        for keyword in config.search_keywords:
            videos, comments = self._collect_keyword(replace(config, keyword=keyword))
            for video in videos:
                row = videos_by_id.setdefault(video["video_id"], {**video, "search_keywords": []})
                row["search_keywords"].append(keyword)
            for comment in comments:
                comments_by_id.setdefault(comment["comment_id"], comment)
        for comment in comments_by_id.values():
            comment["search_keywords"] = videos_by_id[comment["video_id"]]["search_keywords"]
        return list(videos_by_id.values()), list(comments_by_id.values())

    def _collect_keyword(self, config):
        log.info("Searching for %r (up to %d videos)", config.keyword, config.max_videos)
        ids, token = [], None
        while len(ids) < config.max_videos:
            params = dict(part="id", type="video", q=config.keyword,
                          maxResults=min(50, config.max_videos - len(ids)))
            if config.language:
                params["relevanceLanguage"] = config.language
            if token:
                params["pageToken"] = token
            page = self._get("search", **params)
            for item in page.get("items", []):
                video_id = item["id"]["videoId"]
                if video_id not in ids:
                    ids.append(video_id)
            token = page.get("nextPageToken")
            if not token:
                break
        videos, comments = [], []
        for offset in range(0, len(ids), 50):
            page = self._get("videos", part="snippet,statistics,contentDetails",
                             id=",".join(ids[offset:offset + 50]))
            for item in page.get("items", []):
                snippet = item["snippet"]
                videos.append({"video_id": item["id"], "title": snippet["title"],
                               "description": snippet.get("description", ""),
                               "tags": snippet.get("tags", []),
                               "channel_title": snippet.get("channelTitle", ""),
                               "published_at": snippet.get("publishedAt", ""),
                               "statistics": item.get("statistics", {}),
                               "duration": item.get("contentDetails", {}).get("duration", "")})
        for video in videos:
            if video["video_id"] in self._collected_comment_videos:
                continue
            self._collected_comment_videos.add(video["video_id"])
            collected, seen, token = [], set(), None
            while len(collected) < config.max_comments:
                params = dict(part="snippet", videoId=video["video_id"], textFormat="plainText",
                              order="time", maxResults=min(100, config.max_comments - len(collected)))
                if token:
                    params["pageToken"] = token
                try:
                    page = self._get("commentThreads", **params)
                except YouTubeError as exc:
                    if exc.reason in {"commentsDisabled", "videoNotFound"}:
                        log.warning("Skipping comments for %s: %s", video["video_id"], exc.reason)
                        break
                    raise
                for thread in page.get("items", []):
                    comment = thread["snippet"]["topLevelComment"]
                    if comment["id"] in seen:
                        continue
                    seen.add(comment["id"])
                    snippet = comment["snippet"]
                    collected.append({"comment_id": comment["id"], "video_id": video["video_id"],
                                      "text": snippet.get("textOriginal", snippet.get("textDisplay", "")),
                                      "like_count": snippet.get("likeCount", 0),
                                      "published_at": snippet.get("publishedAt", "")})
                token = page.get("nextPageToken")
                if not token:
                    break
            comments.extend(collected)
            log.info("Collected %d comments for %s", len(collected), video["video_id"])
        return videos, comments
