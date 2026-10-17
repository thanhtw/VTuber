"""Registered discovery, researcher screening, channel uploads, and complete reply pagination."""
import json
import copy
import logging
import uuid
from pathlib import Path
from ..youtube import YouTubeClient, YouTubeError
from .storage import save, read_json, utcnow, csv_rows
from .screening import update_template, decision, FORMATS, report, require_teaching_evidence

log = logging.getLogger(__name__)


class RegisteredClient(YouTubeClient):
    def __init__(self, key, root):
        super().__init__(key)
        self.root = Path(root)
        self.run_id = uuid.uuid4().hex
        self.context = {}

    def _get(self, resource, **params):
        call_id = uuid.uuid4().hex
        entry = {"run_id": self.run_id, "call_id": call_id, "time": utcnow(),
                 "resource": resource, "context": dict(self.context), "parameters": {k: v for k, v in params.items() if k != "key"}}
        try:
            data = super()._get(resource, **params)
            entry.update(status="ok", returned_items=len(data.get("items", [])),
                         next_page_token=data.get("nextPageToken"),
                         ids=[x.get("id") for x in data.get("items", [])])
            retained = copy.deepcopy(data)
            if resource in {"comments", "commentThreads"}:
                def redact(value):
                    if isinstance(value, dict):
                        for key in list(value):
                            if key.startswith('author'):
                                del value[key]
                            else:
                                redact(value[key])
                    elif isinstance(value, list):
                        for item in value:
                            redact(item)
                redact(retained)
            save(self.root / "api_responses" / f"{call_id}.json", retained)
            return data
        except BaseException as exc:
            entry.update(status="failed", error_type=type(exc).__name__)
            if isinstance(exc, YouTubeError):
                entry.update(http_status=exc.status, reason=exc.reason)
            raise
        finally:
            path = self.root / "query_registry.jsonl"
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
            path.chmod(0o600)


def pages(client, resource, **params):
    token = None
    seen = set()
    while True:
        page = client._get(resource, **params, **({"pageToken": token} if token else {}))
        yield page
        token = page.get("nextPageToken")
        if not token:
            break
        if token in seen:
            raise ValueError(f"Repeated page token from {resource}; collection is incomplete.")
        seen.add(token)


def channel_uploads(client, channel):
    playlist = channel.get("contentDetails", {}).get("relatedPlaylists", {}).get("uploads")
    if not playlist:
        return []
    result = []
    for page in pages(client, "playlistItems", part="snippet,contentDetails", playlistId=playlist, maxResults=50):
        for item in page.get("items", []):
            snippet = item.get("snippet", {})
            video_id = item.get("contentDetails", {}).get("videoId")
            if video_id:
                result.append({"id": video_id, "title": snippet.get("title", ""),
                               "channel_id": channel["id"], "source": "verified_channel_uploads"})
    return result


def discover(study, client):
    study.prepare()
    videos = {v["id"]: v for v in read_json(study.root / "candidates.json", [])}
    channels = {c["id"]: c for c in read_json(study.root / "channel_candidates.json", [])}
    for query in study.get("queries", []):
        client.context = {"query_id": query.get("id", query["q"]), "stage": "discovery"}
        kind = query.get("type", "video")
        windows = study.get("year_windows") or [None]
        for region in query.get("regions", [""]):
            for order in query.get("orders", ["relevance"]):
                for year in windows:
                    params = {"part": "snippet", "type": kind, "q": query["q"], "maxResults": 50, "order": order}
                    if region:
                        params["regionCode"] = region
                    if query.get("language"):
                        params["relevanceLanguage"] = query["language"]
                    if year is not None:
                        params.update(publishedAfter=f"{int(year)}-01-01T00:00:00Z", publishedBefore=f"{int(year)+1}-01-01T00:00:00Z")
                    for page_number, page in enumerate(pages(client, "search", **params), 1):
                        for item in page.get("items", []):
                            snippet = item.get("snippet", {})
                            channel_id = item["id"].get("channelId") if kind == "channel" else snippet.get("channelId")
                            if channel_id:
                                channels.setdefault(channel_id, {"id": channel_id, "title": snippet.get("channelTitle", snippet.get("title", "")) if kind == "video" else snippet.get("title", "")})
                            if kind == "video":
                                identifier = item["id"]["videoId"]
                                row = videos.setdefault(identifier, {"id": identifier, "title": snippet.get("title", ""), "channel_id": channel_id, "queries": []})
                                if query.get("id", query["q"]) not in row["queries"]:
                                    row["queries"].append(query.get("id", query["q"]))
                        save(study.root / "candidates.json", list(videos.values()))
                        save(study.root / "channel_candidates.json", list(channels.values()))
                        limit = study.get("max_search_pages_per_query", 0)
                        if limit and page_number >= limit:
                            log.warning("Search page cap reached; this query is a bounded sample, not exhaustive retrieval.")
                            break
    client.context = {"stage": "verified_channel_uploads"}
    verified = [r["id"] for r in csv_rows(study.root / "channel_screening.csv") if decision(r) == "include"]
    for offset in range(0, len(verified), 50):
        metadata = client._get("channels", part="snippet,statistics,contentDetails", id=",".join(verified[offset:offset+50]))
        for channel in metadata.get("items", []):
            channels[channel["id"]] = {"id": channel["id"], "title": channel["snippet"]["title"], "metadata": channel}
            for video in channel_uploads(client, channel):
                videos.setdefault(video["id"], {**video, "queries": []})
    save(study.root / "candidates.json", list(videos.values()))
    save(study.root / "channel_candidates.json", list(channels.values()))
    update_template(study.root / "screening.csv", videos.values())
    update_template(study.root / "channel_screening.csv", channels.values())
    log.info("Screening templates: %d videos and %d channels", len(videos), len(channels))


def parse_comment(item, video, parent_id=None):
    snippet = item["snippet"]
    # Do not retain author identities. Equality is computed before discarding them.
    return {"comment_id": item["id"], "video_id": video["id"], "channel_id": video["snippet"]["channelId"],
            "parent_id": parent_id, "is_reply": parent_id is not None,
            "is_creator": snippet.get("authorChannelId", {}).get("value") == video["snippet"]["channelId"],
            "text": snippet.get("textOriginal", snippet.get("textDisplay", "")),
            "published_at": snippet.get("publishedAt"), "updated_at": snippet.get("updatedAt"),
            "like_count": snippet.get("likeCount", 0)}


def collect_comments(client, video, checkpoint):
    rows, issues = {}, []
    try:
        for page in pages(client, "commentThreads", part="snippet", videoId=video["id"],
                          textFormat="plainText", order="time", maxResults=100):
            for thread in page.get("items", []):
                snippet = thread["snippet"]
                top = snippet["topLevelComment"]
                rows[top["id"]] = parse_comment(top, video)
                expected = snippet.get("totalReplyCount", 0)
                if expected:
                    found = 0
                    try:
                        for reply_page in pages(client, "comments", part="snippet", parentId=top["id"], textFormat="plainText", maxResults=100):
                            for reply in reply_page.get("items", []):
                                if reply["id"] not in rows:
                                    found += 1
                                rows[reply["id"]] = parse_comment(reply, video, top["id"])
                    except YouTubeError as exc:
                        if exc.reason not in {"commentNotFound", "forbidden"}:
                            raise
                        issues.append({"parent_id": top["id"], "reason": exc.reason})
                    if found != expected:
                        issues.append({"parent_id": top["id"], "expected_replies": expected, "retrieved_replies": found,
                                       "reason": "reply_count_changed_or_inaccessible"})
            save(checkpoint, {"complete": False, "comments": list(rows.values()), "issues": issues})
    except YouTubeError as exc:
        if exc.reason not in {"commentsDisabled", "videoNotFound", "forbidden"}:
            raise
        issues.append({"reason": exc.reason})
    except BaseException:
        save(checkpoint, {"complete": False, "comments": list(rows.values()), "issues": issues})
        raise
    result = {"complete": True, "collected_at": utcnow(), "comments": list(rows.values()), "issues": issues}
    save(checkpoint, result)
    return result


def collect(study, client):
    study.prepare()
    client.context = {"stage": "screened_collection"}
    channel_rows = csv_rows(study.root / "channel_screening.csv")
    verified = {r["id"] for r in channel_rows if decision(r) == "include"}
    eligible = [r for r in csv_rows(study.root / "screening.csv") if decision(r) == "include"]
    if not eligible:
        raise ValueError("No eligible videos. Complete two independent decisions in screening.csv first.")
    if any(r.get("channel_id") not in verified for r in eligible):
        raise ValueError("Every included video must belong to an independently verified channel in channel_screening.csv.")
    if any(r.get("format") not in FORMATS for r in eligible):
        raise ValueError("Assign each eligible video a format: lecture, worked_example, livestream_lesson, or short.")
    for row in eligible:
        require_teaching_evidence(row)
    channels, videos, comments, coverage = [], [], [], []
    ids = sorted(verified)
    for offset in range(0, len(ids), 50):
        channels.extend(client._get("channels", part="snippet,statistics,contentDetails", id=",".join(ids[offset:offset+50])).get("items", []))
    for row in eligible:
        checkpoint = study.root / "comment_checkpoints" / f"{row['id']}.json"
        result = client._get("videos", part="snippet,statistics,contentDetails,liveStreamingDetails", id=row["id"])
        if not result.get("items"):
            coverage.append({"video_id": row["id"], "reason": "video_inaccessible"})
            continue
        video = result["items"][0]
        if video["snippet"]["channelId"] != row["channel_id"]:
            raise ValueError(f"Screening channel ID disagrees with API metadata for {row['id']}")
        cached = read_json(checkpoint, {})
        batch = cached if cached.get("complete") else collect_comments(client, video, checkpoint)
        video.update(format=row["format"], collected_at=utcnow(), comments_collected_at=batch["collected_at"],
                     teaching_eligibility_evidence={key: row[key] for key in ("avatar_evidence", "instruction_evidence", "instruction_timestamps")})
        videos.append(video)
        comments.extend(batch["comments"])
        coverage.extend({"video_id": row["id"], **issue} for issue in batch["issues"])
        log.info("Collected %d comments/replies for %s", len(batch["comments"]), row["id"])
        save(study.root / "dataset.json", {"schema_version": 1, "collection_complete": False, "collected_at": utcnow(), "channels": channels,
                                           "videos": videos, "comments": comments, "coverage_issues": coverage})
    if videos:
        save(study.root / "dataset.json", {"schema_version": 1, "collection_complete": True,
             "collected_at": utcnow(), "channels": channels, "videos": videos, "comments": comments,
             "coverage_issues": coverage, "eligible_screened_videos": len(eligible)})
    save(study.root / "screening_summary.json", {"videos": report(study.root / "screening.csv"),
                                                "channels": report(study.root / "channel_screening.csv")})
    if not videos:
        raise ValueError("No accessible eligible videos were returned; no analysis dataset was produced.")
