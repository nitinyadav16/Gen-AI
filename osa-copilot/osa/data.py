"""Load, scope and join the weekly feeds (FR-1), and check them (FR-2).

Two guardrails from section 10 are enforced here, before anything reaches
the model:

* Retailer scoping: every feed is filtered to the manager's retailer the
  moment it is read, so another retailer's rows never exist in memory for
  that session ("Cross-retailer data leak").
* Context contract: people's names, free-text notes and commercial terms are
  dropped on load (section 7.2), so they cannot reach the evidence pack or a
  prompt ("Blame on individuals", "Prompt injection through data").
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import pandas as pd

from osa.config import DATA_DIR, Rules, rules_for

KEYS = ["store_id", "sku", "week"]
STORE_FEEDS = {
    "osa": ["osa_pct"],
    "pos": ["units"],
    "store_inventory": ["on_hand"],
    "shipments": ["ordered_units", "shipped_units"],
    "audits": ["on_shelf", "facings", "planogram_facings", "ranged"],
}
DC_COLUMNS = ["dc_on_hand", "dc_ordered_units", "dc_received_units"]

# Section 7.2: never loaded into the copilot's working data.
EXCLUDED_COLUMNS = {"store_manager_name", "rep_name", "notes", "margin_pct"}


@dataclass
class FeedStatus:
    feed: str
    owner: str
    required: bool
    last_refreshed: str
    stale: bool


@dataclass
class Dataset:
    retailer: str
    rules: Rules
    as_of: str
    weeks: list[str]
    latest_week: str
    panel: pd.DataFrame               # one row per store-SKU-week, every signal joined
    stores: pd.DataFrame
    skus: pd.DataFrame
    playbook: dict
    feeds: dict[str, FeedStatus]
    data_issues: pd.DataFrame         # item-level gaps for the latest week
    regions: list[str] = field(default_factory=list)


def available_retailers(data_dir: Path = DATA_DIR) -> list[str]:
    manifest = json.loads((data_dir / "feed_manifest.json").read_text(encoding="utf-8"))
    return sorted(manifest["retailers"])


def _read(path: Path, retailer: str) -> pd.DataFrame:
    df = pd.read_csv(path, keep_default_na=True)
    if "retailer" in df.columns:
        df = df[df["retailer"] == retailer]          # scope first, before anything else
    return df.drop(columns=[c for c in df.columns if c in EXCLUDED_COLUMNS]).reset_index(drop=True)


def load_dataset(retailer: str, data_dir: Path = DATA_DIR, as_of: str | None = None) -> Dataset:
    manifest = json.loads((data_dir / "feed_manifest.json").read_text(encoding="utf-8"))
    if retailer not in manifest["retailers"]:
        raise ValueError(f"Unknown retailer scope: {retailer!r}")
    rules = rules_for(retailer)
    as_of = as_of or manifest["as_of"]

    feeds: dict[str, FeedStatus] = {}
    for name, spec in manifest["retailers"][retailer].items():
        age = (date.fromisoformat(as_of) - date.fromisoformat(spec["last_refreshed"])).days
        feeds[name] = FeedStatus(name, spec["owner"], spec["required"], spec["last_refreshed"],
                                 stale=age > rules.stale_after_days)

    stores = _read(data_dir / "stores.csv", retailer)
    skus = _read(data_dir / "skus.csv", retailer)
    raw = {name: _read(data_dir / f"{name}.csv", retailer) for name in STORE_FEEDS}
    dc = _read(data_dir / "dc.csv", retailer)
    promo = _read(data_dir / "promo.csv", retailer)
    playbook = json.loads((data_dir / "playbook.json").read_text(encoding="utf-8"))

    weeks = sorted(raw["osa"]["week"].unique())
    latest = manifest.get("latest_week", weeks[-1])

    # The full store x SKU x week grid, so a missing feed row shows up as a gap, not as zero.
    grid = (stores[["store_id", "region", "dc_id"]]
            .merge(skus[["sku", "sku_name", "category", "price"]], how="cross")
            .merge(pd.DataFrame({"week": weeks}), how="cross"))
    panel = grid
    for name, cols in STORE_FEEDS.items():
        panel = panel.merge(raw[name][KEYS + cols], on=KEYS, how="left")
    panel = panel.merge(dc[["dc_id", "sku", "week"] + DC_COLUMNS], on=["dc_id", "sku", "week"], how="left")
    promo = promo.rename(columns={"expected_uplift_pct": "promo_uplift_pct"})
    panel = panel.merge(promo[["region", "sku", "week", "promo_uplift_pct"]],
                        on=["region", "sku", "week"], how="left")
    panel["promo_active"] = panel["promo_uplift_pct"].notna()
    panel.insert(0, "retailer", retailer)
    panel = panel.sort_values(["store_id", "sku", "week"]).reset_index(drop=True)

    data_issues = _completeness(panel, latest, feeds)
    return Dataset(retailer, rules, as_of, weeks, latest, panel, stores, skus, playbook, feeds,
                   data_issues, regions=sorted(stores["region"].unique()))


def _completeness(panel: pd.DataFrame, latest: str, feeds: dict[str, FeedStatus]) -> pd.DataFrame:
    """FR-2: a required feed that is stale, or missing for an item's latest week, is a data issue."""
    last = panel[panel["week"] == latest]
    feed_cols = {**{k: v for k, v in STORE_FEEDS.items()}, "dc": DC_COLUMNS}
    rows = []
    stale = [f for f, s in feeds.items() if s.required and s.stale]
    for r in last.itertuples(index=False):
        problems = [(f, "feed more than one week stale") for f in stale]
        for f, cols in feed_cols.items():
            if feeds.get(f) and feeds[f].required and any(pd.isna(getattr(r, c)) for c in cols):
                problems.append((f, f"no {f} data for week {latest}"))
        for f, why in problems:
            rows.append({"item_key": item_key(r.store_id, r.sku), "store_id": r.store_id, "sku": r.sku,
                         "sku_name": r.sku_name, "region": r.region, "feed": f, "issue": why,
                         "owner": feeds[f].owner})
    return pd.DataFrame(rows, columns=["item_key", "store_id", "sku", "sku_name", "region", "feed",
                                       "issue", "owner"])


def item_key(store_id: str, sku: str) -> str:
    return f"{store_id}|{sku}"
