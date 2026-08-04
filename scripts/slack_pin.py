"""Retire a superseded pinned card and pin the new one.

VENDORED ON PURPOSE, not imported from `_shared/`. These lanes run in GitHub Actions
where `<workspace>/_shared/` is not checked out, so a sys.path shim would work on the
laptop and go silently inert in CI -- which for a pin means "the stale card stays up
and nobody notices". Canonical copy + rationale:
`portfolio_daily/scripts/post_pm_overview.py`.

Requires ClaudeBot scopes `pins:read` + `pins:write` (added 2026-08-04, board #258).

THE SAFETY PROPERTY THAT MATTERS: `retire_own_pins` only ever unpins a message the BOT
wrote whose text matches the card being replaced. A pin a human put up, or another
bot's, is left alone -- "tidy up the old one" must never quietly become "removed
something someone chose to keep".
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request


_PINS_CACHE: dict = {}


def _call(token: str, method: str, payload: dict | None = None, get: bool = False,
          _attempt: int = 0):
    """Slack call that HONOURS 429 Retry-After.

    Measured 2026-08-04: `pins.list` throttles after about five rapid calls and answers
    `429 Retry-After: 30`. That matters more than it looks -- `pin_is_current` resolves
    any read failure to "refresh", so an unhandled 429 does not merely fail, it silently
    turns into a repost of a card that had not changed. A sweep across 20 channels was
    reposting most of them for no reason at all.
    """
    if get:
        url = f"https://slack.com/api/{method}?" + urllib.parse.urlencode(payload or {})
        req = urllib.request.Request(
            url, headers={"Authorization": f"Bearer {token}"}, method="GET")
    else:
        req = urllib.request.Request(
            f"https://slack.com/api/{method}",
            data=json.dumps(payload or {}).encode(),
            headers={"Authorization": f"Bearer {token}",
                     "Content-Type": "application/json; charset=utf-8"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        if e.code == 429 and _attempt < 3:
            wait = int(e.headers.get("Retry-After") or 30) + 1
            print(f"[pin] rate limited on {method}; waiting {wait}s", file=sys.stderr)
            time.sleep(wait)
            return _call(token, method, payload, get, _attempt + 1)
        raise


def _invalidate(channel: str) -> None:
    _PINS_CACHE.pop(channel, None)


def retire_own_pins(token: str, channel: str, fallback_text: str,
                    marker: str | None = None) -> int:
    """Unpin this card's previous copies. Returns how many were retired.

    Call BEFORE posting the replacement: if the post then fails, the channel is left
    with no pin rather than two pinned cards that disagree with each other.

    PASS A `marker`. Matching on `fallback_text` alone was the first design and it is
    subtly broken: the moment a card's TITLE changes -- which is exactly what happens
    when a generator is improved -- the old copy no longer matches, is never retired,
    and quietly accumulates. Observed live on 2026-08-04: #13f ended up with a June and
    an August card pinned together, and #macro-and-markets with three.

    A `marker` is a stable string the generator always embeds (its own module path is
    ideal -- it never changes when the wording does). Any BOT-authored pinned message
    containing it is ours and is retired. Human pins never contain it, so the safety
    property that mattered is preserved.
    """
    try:
        listing = _PINS_CACHE.get(channel) or _call(
            token, "pins.list", {"channel": channel}, get=True)
    except Exception as e:  # noqa: BLE001 - tidying must never block publishing
        print(f"[pin] pins.list failed ({e}); not retiring anything", file=sys.stderr)
        return 0
    n = 0
    for item in listing.get("items", []):
        msg = item.get("message") or {}
        if not msg.get("bot_id"):
            continue          # a human's pin - never touch it
        if marker:
            # Search the ENTITY-NORMALISED blob: Slack stores "&" as "&amp;", so a
            # marker containing an ampersand ("Macro & Markets Monitor") would never
            # match its own card. Cost us a duplicate pin before it was spotted.
            blob = json.dumps(msg.get("blocks") or [], ensure_ascii=False)
            blob = (blob.replace("&amp;", "&").replace("&lt;", "<")
                        .replace("&gt;", ">"))
            body = (msg.get("text") or "").replace("&amp;", "&")
            if marker not in blob and marker not in body:
                continue      # another lane's card in the same channel
        elif (msg.get("text") or "") != fallback_text:
            continue
        res = _call(token, "pins.remove",
                    {"channel": channel, "timestamp": msg.get("ts")})
        if res.get("ok"):
            n += 1
            _invalidate(channel)
        else:
            print(f"[pin] could not unpin {msg.get('ts')}: {res.get('error')}",
                  file=sys.stderr)
    return n


def pin(token: str, channel: str, ts: str) -> bool:
    """Pin a message. NON-FATAL on failure: the card has already landed in the
    channel, and failing the run here would discard work that succeeded."""
    try:
        res = _call(token, "pins.add", {"channel": channel, "timestamp": ts})
    except Exception as e:  # noqa: BLE001
        print(f"[pin] pins.add raised ({e}) - pin by hand", file=sys.stderr)
        return False
    _invalidate(channel)
    if res.get("ok") or res.get("error") == "already_pinned":
        return True
    print(f"[pin] pins.add failed: {res.get('error')} - pin by hand", file=sys.stderr)
    return False

def stamp(blocks: list) -> list:
    """Append a short content hash to the card, so freshness is a lookup not a guess.

    WHY THIS REPLACED TEXT COMPARISON. Slack does not store what you post, and the list
    of rewrites is longer than it looks: channel names become mentions, `&` becomes
    `&amp;`, literal emoji become shortcodes, keycaps collapse, bare URLs auto-link,
    newlines in the fallback text become spaces, and every block gains a `block_id`.
    Each one was found by a card reposting itself forever, and after fixing six of them
    a seventh appeared. Reversing a transformation you do not control is the wrong
    shape of problem.

    So the card carries its own identity instead: a hash of the blocks WE built, which
    Slack stores verbatim because it is lowercase hex. Comparing it is exact, and it is
    immune to every present and future rewrite.
    """
    body = json.dumps(blocks, sort_keys=True, ensure_ascii=False)
    h = hashlib.sha256(body.encode("utf-8")).hexdigest()[:12]
    out = [dict(b) for b in blocks]
    tag = f"card:{h}"
    for b in reversed(out):
        if b.get("type") == "context" and b.get("elements"):
            els = [dict(e) for e in b["elements"]]
            els[-1] = dict(els[-1])
            els[-1]["text"] = f"{els[-1].get('text', '')}  ·  {tag}"
            b["elements"] = els
            return out
    out.append({"type": "context",
                "elements": [{"type": "mrkdwn", "text": tag}]})
    return out


def _stamped_hash(blocks: list) -> str | None:
    for b in reversed(blocks or []):
        if (b or {}).get("type") == "context":
            for e in reversed(b.get("elements") or []):
                m = re.search(r"card:([0-9a-f]{12})", (e or {}).get("text", "") or "")
                if m:
                    return m.group(1)
    return None


def pin_is_current(token: str, channel: str, fallback_text: str,
                   blocks: list) -> bool:
    """True when the pinned card carries the same content hash we would post.

    `blocks` may be stamped or unstamped -- both work, so callers cannot get it wrong.
    Any doubt resolves to False (refresh): a spurious repost is cosmetic, whereas
    wrongly concluding "still current" leaves a card asserting an old threshold, which
    is the entire failure this exists to prevent.
    """
    want = _stamped_hash(blocks) or _stamped_hash(stamp(blocks))
    if not want:
        return False
    try:
        if channel in _PINS_CACHE:
            listing = _PINS_CACHE[channel]
        else:
            listing = _call(token, "pins.list", {"channel": channel}, get=True)
            _PINS_CACHE[channel] = listing
    except Exception as e:  # noqa: BLE001
        print(f"[pin] pins.list failed ({e}); assuming a refresh is needed",
              file=sys.stderr)
        return False
    for item in listing.get("items", []):
        msg = item.get("message") or {}
        if not msg.get("bot_id"):
            continue
        if _stamped_hash(msg.get("blocks")) == want:
            return True
    return False
