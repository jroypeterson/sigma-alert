"""One-off preview: posts a sample Slack message with dummy alerts
so you can eyeball the new subcategory formatting. Requires SLACK_WEBHOOK
in the environment. Not wired into any workflow — manual use only.

Run:  SLACK_WEBHOOK=https://hooks.slack.com/... python scripts/preview_slack.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sigma_screener import format_slack_message, send_slack


def _alert(ticker, name, sector, z, ret, price, tier="2sigma"):
    return {
        "ticker": ticker,
        "name": name,
        "sector": sector,
        "z_score": z,
        "return_pct": ret,
        "price": price,
        "direction": "up" if ret > 0 else "down",
        "three_sigma": abs(z) >= 3.0,
        "tier": tier,
    }


dummy_alerts = [
    # 2-sigma tier — spread across all four subcategories, some multi-category
    _alert("UNH", "UnitedHealth Group Inc", "Healthcare Services", 3.21, 7.84, 512.30),
    _alert("HCA", "HCA Healthcare Inc", "Healthcare Services", -2.55, -4.12, 301.45),
    _alert("ISRG", "Intuitive Surgical Inc", "MedTech", 2.78, 5.62, 452.10),
    _alert("MDT", "Medtronic plc", "MedTech", -2.11, -3.40, 83.25),
    _alert("NVO", "Novo Nordisk A/S", "PA", 2.44, 4.91, 128.77),
    _alert("AAPL", "Apple Inc", "Tech", 2.35, 3.20, 189.40),   # S&P 500 only
    _alert("NVDA", "NVIDIA Corporation", "Tech", -2.05, -5.10, 842.15),  # S&P 500 only
    # 1-sigma tier — sector-restricted, no S&P overlap needed
    _alert("TDOC", "Teladoc Health Inc", "Healthcare Services", 1.45, 2.30, 9.85, tier="1sigma"),
    _alert("DXCM", "DexCom Inc", "MedTech", -1.22, -1.80, 72.40, tier="1sigma"),
    _alert("LLY", "Eli Lilly and Company", "PA", 1.67, 2.55, 789.10, tier="1sigma"),
]

dummy_hi_lo = [
    {"ticker": "UNH", "name": "UnitedHealth Group Inc", "sector": "Healthcare Services", "type": "high", "price": 512.30},
    {"ticker": "NVDA", "name": "NVIDIA Corporation", "sector": "Tech", "type": "low", "price": 842.15},
]

# S&P 500 membership for dummy data
sp500 = {"UNH", "HCA", "ISRG", "MDT", "AAPL", "NVDA", "LLY"}

stats = {"screened": 693, "skipped": 0, "stale": 0, "ref_date": "2026-04-10"}
payload = format_slack_message(
    dummy_alerts, "close", 693, stats, dummy_hi_lo, sp500
)
send_slack(payload)
