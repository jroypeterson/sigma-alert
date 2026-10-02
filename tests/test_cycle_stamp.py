"""Per-cycle stamp for return_map.html (board #445).

Three cycles overwrite one HTML, so its mtime cannot say WHICH cycle ran. Each
mode records its own key in `readable/return_map.cycles.json`; the fleet's
artifact-freshness check reads `<mode>.ok_at`. These pin: one mode never
touches another's key, ok_at advances only on an OK exit, a corrupt file does
not raise, and main()'s real exits stamp what actually happened.
"""
import json
import sys
from datetime import datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import return_map as rm  # noqa: E402
import sigma_screener as ss  # noqa: E402

ET = rm.ET
T1 = datetime(2026, 10, 1, 9, 52, tzinfo=ET)
T2 = datetime(2026, 10, 1, 16, 49, tzinfo=ET)


def _read(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


class TestRecordCycle:
    def test_written_sets_ok_at_and_exit(self, tmp_path):
        p = tmp_path / "c.json"
        assert rm.record_cycle("open", "written", p, now=T1, ref_date="2026-10-01")
        e = _read(p)["open"]
        assert e["ok_at"] == T1.isoformat(timespec="seconds")
        assert e["last_exit"] == "written" and e["ref_date"] == "2026-10-01"
        assert e["artifact"]

    def test_one_mode_never_touches_another(self, tmp_path):
        p = tmp_path / "c.json"
        rm.record_cycle("open", "written", p, now=T1)
        rm.record_cycle("close", "written", p, now=T2)
        d = _read(p)
        assert d["open"]["ok_at"] == T1.isoformat(timespec="seconds")
        assert d["close"]["ok_at"] == T2.isoformat(timespec="seconds")
        assert "midday" not in d

    def test_a_failed_exit_keeps_the_previous_ok_at(self, tmp_path):
        p = tmp_path / "c.json"
        rm.record_cycle("close", "written", p, now=T1)
        rm.record_cycle("close", "publish-gate", p, now=T2)
        e = _read(p)["close"]
        assert e["ok_at"] == T1.isoformat(timespec="seconds")
        assert e["last_exit"] == "publish-gate"
        assert e["last_exit_at"] == T2.isoformat(timespec="seconds")

    def test_market_closed_advances_ok_at(self, tmp_path):
        p = tmp_path / "c.json"
        rm.record_cycle("close", "market-closed", p, now=T2)
        assert _read(p)["close"]["ok_at"] == T2.isoformat(timespec="seconds")

    @pytest.mark.parametrize("kind", ["publish-gate", "coverage-floor", "write-failed",
                                      "duplicate", "started"])
    def test_failure_exits_never_set_ok_at(self, tmp_path, kind):
        p = tmp_path / "c.json"
        rm.record_cycle("midday", kind, p, now=T2)
        assert "ok_at" not in _read(p)["midday"]

    def test_corrupt_file_is_replaced_not_raised(self, tmp_path):
        p = tmp_path / "c.json"
        p.write_text("{not json", encoding="utf-8")
        assert rm.record_cycle("open", "written", p, now=T1)
        assert _read(p)["open"]["last_exit"] == "written"
        assert list(tmp_path.glob("*.tmp")) == []

    def test_unwritable_path_returns_false_never_raises(self, tmp_path):
        blocker = tmp_path / "file"
        blocker.write_text("x")
        assert rm.record_cycle("open", "written", blocker / "c.json", now=T1) is False

    def test_default_now_carries_an_offset(self, tmp_path):
        # The stamp is compared against an ET task start time; it must carry an offset.
        p = tmp_path / "c.json"
        rm.record_cycle("open", "written", p)
        assert datetime.fromisoformat(_read(p)["open"]["ok_at"]).tzinfo is not None


class TestGateExitKind:
    def test_holiday_is_market_closed_when_the_prior_session_was_covered(self, monkeypatch):
        # Thanksgiving 2026; the open map was written Wednesday 11-25.
        rm.record_cycle("open", "written", rm.CYCLES_PATH,
                        now=datetime(2026, 11, 25, 9, 52, tzinfo=ET))
        monkeypatch.setattr(ss, "now_et", lambda: datetime(2026, 11, 26, 9, 41, tzinfo=ET))
        assert ss.gate_exit_kind("publish-gate", "open") == "market-closed"

    def test_holiday_after_a_MISSED_session_is_not_blessed(self, monkeypatch):
        # Codex R2: last written Tuesday 11-24, Wednesday missed, then the holiday.
        rm.record_cycle("open", "written", rm.CYCLES_PATH,
                        now=datetime(2026, 11, 24, 9, 52, tzinfo=ET))
        monkeypatch.setattr(ss, "now_et", lambda: datetime(2026, 11, 26, 9, 41, tzinfo=ET))
        assert ss.gate_exit_kind("publish-gate", "open") == "publish-gate"
        assert ss.gate_exit_kind("publish-gate", "midday") == "publish-gate"  # no stamp

    def test_weekend_catch_up_is_NOT_market_closed(self, monkeypatch):
        # Codex R1: a Friday cycle run on Saturday by StartWhenAvailable must not
        # bless the missed Friday.
        monkeypatch.setattr(ss, "now_et", lambda: datetime(2026, 10, 3, 9, 41, tzinfo=ET))
        assert ss.gate_exit_kind("publish-gate", "open") == "publish-gate"

    def test_session_day_keeps_the_gate_kind(self, monkeypatch):
        monkeypatch.setattr(ss, "now_et", lambda: datetime(2026, 10, 1, 16, 30, tzinfo=ET))
        assert ss.gate_exit_kind("publish-gate", "open") == "publish-gate"


class TestDryRunRedirectsTheStamp:
    def test_cycles_path_moves_into_the_state_dir(self, monkeypatch, tmp_path):
        for name in ("DRY_RUN", "CACHE_PATH", "SKIP_LOG_PATH", "MISSING_METADATA_PATH"):
            monkeypatch.setattr(ss, name, getattr(ss, name))
        monkeypatch.setattr(rm, "SNAPSHOT_PATH", rm.SNAPSHOT_PATH)
        monkeypatch.setattr(rm, "HTML_PATH", rm.HTML_PATH)
        d = ss._enter_dry_run(tmp_path / "state")
        assert Path(rm.CYCLES_PATH).resolve().parent == d.resolve()


class TestStartedStamp:
    def test_started_never_sets_ok_at_and_keeps_the_previous_one(self, tmp_path):
        p = tmp_path / "c.json"
        rm.record_cycle("open", "written", p, now=T1)
        rm.record_cycle("open", "started", p, now=T2)
        e = _read(p)["open"]
        assert e["last_exit"] == "started"
        assert e["ok_at"] == T1.isoformat(timespec="seconds")


class TestTransientReadErrorNeverWipes:
    def test_a_locked_file_is_left_alone_and_returns_false(self, tmp_path, monkeypatch):
        p = tmp_path / "c.json"
        rm.record_cycle("close", "written", p, now=T1)
        before = p.read_text(encoding="utf-8")
        real = Path.read_text

        def _locked(self, *a, **k):
            if self == p:
                raise PermissionError(32, "sharing violation")
            return real(self, *a, **k)
        monkeypatch.setattr(Path, "read_text", _locked)
        monkeypatch.setattr("time.sleep", lambda s: None)
        assert rm.record_cycle("open", "written", p, now=T2) is False
        monkeypatch.undo()
        assert p.read_text(encoding="utf-8") == before


class TestHolidayUsesTheCoveredSession:
    def test_a_map_written_from_the_previous_days_bars_does_not_cover_today(self, monkeypatch):
        # Codex R3: written 2026-04-02 from 04-01 bars; Good Friday 04-03 must
        # not be blessed, because no 04-02 session map exists.
        rm.record_cycle("midday", "written", rm.CYCLES_PATH,
                        now=datetime(2026, 4, 2, 12, 50, tzinfo=ET), ref_date="2026-04-01")
        monkeypatch.setattr(ss, "now_et", lambda: datetime(2026, 4, 3, 12, 36, tzinfo=ET))
        assert ss.gate_exit_kind("publish-gate", "midday") == "publish-gate"

    def test_a_map_covering_the_previous_session_does(self, monkeypatch):
        rm.record_cycle("midday", "written", rm.CYCLES_PATH,
                        now=datetime(2026, 4, 2, 12, 50, tzinfo=ET), ref_date="2026-04-02")
        monkeypatch.setattr(ss, "now_et", lambda: datetime(2026, 4, 3, 12, 36, tzinfo=ET))
        assert ss.gate_exit_kind("publish-gate", "midday") == "market-closed"
