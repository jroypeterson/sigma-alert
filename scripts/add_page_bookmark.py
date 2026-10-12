"""Bookmark the Weekly Price Summary page in #stock-price-alerts.

JP 2026-10-12: "Lets pin this as a bookmark in the stock-price alert channel".
A channel bookmark sits in the header bar, so the page is one tap away without
hunting for Friday's post. The URL is stable (the artifact is republished in
place), so this only needs re-running if the URL ever changes.

Idempotent by title: an existing bookmark is edited, never added twice. Uses the
same bot token as post_overview.py (needs bookmarks:read + bookmarks:write).

    python scripts/add_page_bookmark.py [--dry-run]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from post_overview import CHANNEL_ID, _bot_token  # noqa: E402
from weekly_summary import WEEKLY_PAGE_URL  # noqa: E402

TITLE = "Weekly price summary"
EMOJI = ":bar_chart:"


def _api(method: str, token: str, params: dict, post: bool = False) -> dict:
    headers = {"Authorization": f"Bearer {token}"}
    r = (requests.post(f"https://slack.com/api/{method}", json=params, headers=headers, timeout=30)
         if post else
         requests.get(f"https://slack.com/api/{method}", params=params, headers=headers, timeout=30))
    body = r.json()
    if not body.get("ok"):
        raise RuntimeError(f"{method} failed: {body.get('error')}")
    return body


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    token = _bot_token()
    if not token:
        print("[ERROR] no SLACK_BOT_TOKEN")
        return 1
    existing = _api("bookmarks.list", token, {"channel_id": CHANNEL_ID}).get("bookmarks", [])
    match = next((b for b in existing if b.get("title") == TITLE), None)
    if a.dry_run:
        print(f"[dry-run] would {'edit' if match else 'add'} {TITLE!r} -> {WEEKLY_PAGE_URL}")
        return 0
    if match:
        _api("bookmarks.edit", token, {"channel_id": CHANNEL_ID, "bookmark_id": match["id"],
                                       "title": TITLE, "link": WEEKLY_PAGE_URL}, post=True)
        print(f"updated {TITLE!r} -> {WEEKLY_PAGE_URL}")
    else:
        _api("bookmarks.add", token, {"channel_id": CHANNEL_ID, "title": TITLE, "type": "link",
                                      "link": WEEKLY_PAGE_URL, "emoji": EMOJI}, post=True)
        print(f"added {TITLE!r} -> {WEEKLY_PAGE_URL}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
