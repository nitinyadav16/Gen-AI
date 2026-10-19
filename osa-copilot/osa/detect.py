"""At-risk detection (FR-3) and the lost-sales ranked worklist (FR-4).

No LLM here: flags are rules, lost sales is arithmetic (section 3, O1 and O4).

A week counts as "affected" when OSA is below target, or has dropped more than
`drop_points` against the mean of the 4 weeks before it. Walking back from the
latest week gives the number of weeks affected; the 4-week baseline is the 4
weeks before the first affected week, so the decline never dilutes its own
baseline.

    lost sales = baseline weekly units x price x weeks affected x (1 - OSA)

where OSA is the mean OSA over the affected weeks and price is shelf price.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

from osa.data import Dataset, item_key


@dataclass
class FlaggedItem:
    key: str
    rank: int
    retailer: str
    region: str
    store_id: str
    dc_id: str
    sku: str
    sku_name: str
    category: str
    latest_week: str
    osa_latest: float
    osa_baseline: float
    osa_target: float
    osa_avg_affected: float
    drop_points: float
    weeks_affected: int
    flag_reason: str
    baseline_units: float
    price: float
    lost_sales: float
    baseline_weeks: list[str]
    affected_weeks: list[str]

    def formula(self) -> str:
        return (f"{self.baseline_units:g} units/wk x ${self.price:.2f} x {self.weeks_affected} wk "
                f"x (1 - {self.osa_avg_affected:g}%) = ${self.lost_sales:,.2f}")

    def to_dict(self) -> dict:
        return asdict(self)


def _affected_run(osa: list[float], target: float, drop: float, window: int) -> int:
    n = 0
    for i in range(len(osa) - 1, -1, -1):
        prior = osa[max(0, i - window):i]
        dropped = len(prior) == window and (sum(prior) / window - osa[i]) > drop
        if osa[i] < target or dropped:
            n += 1
        else:
            break
    return n


def build_worklist(ds: Dataset) -> list[FlaggedItem]:
    r = ds.rules
    excluded = set(ds.data_issues["item_key"])          # FR-2: data issues never get ranked
    items: list[FlaggedItem] = []
    for (store_id, sku), g in ds.panel.groupby(["store_id", "sku"], sort=True):
        key = item_key(store_id, sku)
        if key in excluded:
            continue
        g = g[g["osa_pct"].notna()]
        if len(g) < r.min_history_weeks or g["week"].iloc[-1] != ds.latest_week:
            continue
        osa = g["osa_pct"].tolist()
        k = _affected_run(osa, r.osa_target, r.drop_points, r.baseline_weeks)
        if k == 0 or len(osa) - k < r.baseline_weeks:           # not flagged, or not enough history
            continue
        base = g.iloc[len(osa) - k - r.baseline_weeks: len(osa) - k]
        hit = g.iloc[len(osa) - k:]
        last = g.iloc[-1]
        osa_base = round(base["osa_pct"].mean(), 1)
        osa_hit = round(hit["osa_pct"].mean(), 1)
        units = round(base["units"].mean(), 1)
        price = float(last["price"])
        drop = round(osa_base - last["osa_pct"], 1)
        reasons = []
        if last["osa_pct"] < r.osa_target:
            reasons.append(f"OSA {last['osa_pct']:g}% below target {r.osa_target:g}%")
        if drop > r.drop_points:
            reasons.append(f"down {drop:g} pts vs 4-week baseline {osa_base:g}%")
        items.append(FlaggedItem(
            key=key, rank=0, retailer=ds.retailer, region=last["region"], store_id=store_id,
            dc_id=last["dc_id"], sku=sku, sku_name=last["sku_name"], category=last["category"],
            latest_week=ds.latest_week, osa_latest=float(last["osa_pct"]), osa_baseline=osa_base,
            osa_target=r.osa_target, osa_avg_affected=osa_hit, drop_points=drop, weeks_affected=k,
            flag_reason="; ".join(reasons) or f"OSA below target for {k} weeks",
            baseline_units=units, price=price,
            lost_sales=round(units * price * k * (1 - osa_hit / 100), 2),
            baseline_weeks=base["week"].tolist(), affected_weeks=hit["week"].tolist(),
        ))
    items.sort(key=lambda x: (-x.lost_sales, x.key))
    for i, it in enumerate(items, 1):
        it.rank = i
    return items


def worklist_frame(items: list[FlaggedItem]) -> pd.DataFrame:
    return pd.DataFrame([{
        "rank": i.rank, "key": i.key, "region": i.region, "store": i.store_id, "sku": i.sku,
        "product": i.sku_name, "osa_latest": i.osa_latest, "osa_baseline": i.osa_baseline,
        "weeks_affected": i.weeks_affected, "baseline_units": i.baseline_units, "price": i.price,
        "est_lost_sales": i.lost_sales, "flag_reason": i.flag_reason,
    } for i in items])
