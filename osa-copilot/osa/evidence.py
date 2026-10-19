"""Evidence pack per flagged store-SKU (FR-5).

Code, not the model, computes every number. The pack holds:

* signals   - numbered facts (S1, S2, ...) with a value and, where it makes
              sense, the 4-week baseline it is compared to. The model cites
              these by ID and value; the validator checks each citation.
* checks    - one deterministic rule check per cause in the cause list,
              "strong" (every condition met), "weak" (the key condition plus
              at least one other) or "not met".
* conflicts - signal combinations that contradict each other. Any conflict
              sends the item to "Needs investigation" (section 9.2).

Only structured numeric and categorical fields go in. Free text never does.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import pandas as pd

from osa import causes as C
from osa.data import Dataset
from osa.detect import FlaggedItem


@dataclass
class Signal:
    id: str
    name: str
    value: float | int | str
    baseline: float | int | str | None = None
    unit: str = ""
    basis: str = "4-week baseline"       # what `baseline` means for this signal

    def text(self) -> str:
        v = fmt(self.value, self.unit)
        if self.baseline is None:
            return f"{self.name}: {v}"
        return f"{self.name}: {v} vs {self.basis} {fmt(self.baseline, self.unit)}"


@dataclass
class Check:
    cause: str
    result: str                      # strong | weak | not met
    met: list[str]
    not_met: list[str]
    signal_ids: list[str]


@dataclass
class EvidencePack:
    item: FlaggedItem
    signals: list[Signal]
    checks: list[Check]
    conflicts: list[str] = field(default_factory=list)

    def signal(self, sid: str) -> Signal | None:
        return next((s for s in self.signals if s.id == sid), None)

    def check(self, cause: str) -> Check | None:
        return next((c for c in self.checks if c.cause == cause), None)

    @property
    def any_check_met(self) -> bool:
        return any(c.result != "not met" for c in self.checks)

    def for_model(self) -> dict:
        """The exact JSON the model sees: structured fields only."""
        it = self.item
        return {
            "item": {"region": it.region, "store_id": it.store_id, "sku": it.sku, "product": it.sku_name,
                     "category": it.category, "latest_week": it.latest_week, "osa_target_pct": it.osa_target,
                     "weeks_affected": it.weeks_affected},
            "signals": [{k: v for k, v in asdict(s).items()
                         if v not in (None, "") and not (k == "basis" and s.baseline is None)}
                        for s in self.signals],
            "rule_checks": [{"cause": c.cause, "result": c.result, "conditions_met": c.met,
                             "signals": c.signal_ids} for c in self.checks],
            "conflicts": self.conflicts,
        }

    def to_dict(self) -> dict:
        return {"item": self.item.to_dict(), "signals": [asdict(s) for s in self.signals],
                "checks": [asdict(c) for c in self.checks], "conflicts": self.conflicts}


def fmt(v, unit: str = "") -> str:
    if isinstance(v, str):
        return v
    s = f"{v:,.0f}" if float(v).is_integer() else f"{v:,.1f}"
    return f"{s}%" if unit == "%" else f"{s} {unit}".rstrip()


def _r(x: float) -> float | int:
    x = round(float(x), 1)
    return int(x) if x.is_integer() else x


def _ratio(a: float, b: float) -> float:
    return a / b if b else (0.0 if a == 0 else float("inf"))


def _fill(shipped: float, ordered: float) -> float | str:
    return _r(100 * shipped / ordered) if ordered else "no orders"


def build_pack(ds: Dataset, item: FlaggedItem) -> EvidencePack:
    g = ds.panel[(ds.panel.store_id == item.store_id) & (ds.panel.sku == item.sku)].set_index("week")
    b = g.loc[item.baseline_weeks]
    a = g.loc[item.affected_weeks]
    last = g.loc[item.latest_week]
    before = g.loc[item.baseline_weeks[-1]]

    audited = pd.notna(last["on_shelf"])
    promo_weeks = a[a["promo_active"]]
    sig = [
        Signal("S1", "OSA latest week", _r(last.osa_pct), _r(b.osa_pct.mean()), "%"),
        Signal("S2", "Weeks affected", int(item.weeks_affected), None, "weeks"),
        Signal("S3", "POS units latest week", _r(last.units), _r(b.units.mean()), "units"),
        Signal("S4", "Peak POS units in affected weeks", _r(a.units.max()), _r(b.units.mean()), "units"),
        Signal("S5", "Store on-hand latest week", _r(last.on_hand), _r(b.on_hand.mean()), "units"),
        Signal("S6", "Store on-hand week before decline", _r(before.on_hand), None, "units"),
        Signal("S7", "Store orders per week in affected weeks", _r(a.ordered_units.mean()),
               _r(b.ordered_units.mean()), "units"),
        Signal("S8", "Store order fill rate latest week", _fill(last.shipped_units, last.ordered_units),
               _fill(b.shipped_units.sum(), b.ordered_units.sum()), "%"),
        Signal("S9", "DC on-hand latest week", _r(last.dc_on_hand), _r(b.dc_on_hand.mean()), "units"),
        Signal("S10", "Manufacturer fill rate to DC latest week",
               _fill(last.dc_received_units, last.dc_ordered_units),
               _fill(b.dc_received_units.sum(), b.dc_ordered_units.sum()), "%"),
        Signal("S11", "Audit: SKU on shelf", last.on_shelf if audited else "not audited"),
        Signal("S12", "Audit: facings on shelf", _r(last.facings) if audited else "not audited",
               _r(last.planogram_facings) if audited else None, "facings", "planogram"),
        Signal("S13", "Audit: SKU ranged on planogram", last.ranged if audited else "not audited"),
        Signal("S14", "Promotion active in affected weeks", "Y" if len(promo_weeks) else "N"),
    ]
    if len(promo_weeks):
        sig.append(Signal("S15", "Promotion expected uplift", _r(promo_weeks.promo_uplift_pct.max()), None, "%"))
    s = {x.id: x for x in sig}

    # ratios the checks use (never shown to the model as new numbers)
    r_osa_drop = s["S1"].baseline - s["S1"].value
    r_pos = _ratio(s["S3"].value, s["S3"].baseline)
    r_peak = _ratio(s["S4"].value, s["S4"].baseline)
    r_oh = _ratio(s["S5"].value, s["S5"].baseline)
    r_orders = _ratio(s["S7"].value, s["S7"].baseline)
    r_dc = _ratio(s["S9"].value, s["S9"].baseline)
    store_fill = s["S8"].value if not isinstance(s["S8"].value, str) else None
    mfr_fill = s["S10"].value if not isinstance(s["S10"].value, str) else 100.0
    on_shelf, ranged = s["S11"].value, s["S13"].value
    facings, plan = s["S12"].value, s["S12"].baseline
    short_facings = audited and plan not in (None, 0) and facings < plan

    def check(cause, key, others, ids) -> Check:
        conds = [key] + others
        met = [d for d, ok in conds if ok]
        not_met = [d for d, ok in conds if not ok]
        result = "strong" if not not_met else ("weak" if key[1] and len(met) >= 2 else "not met")
        return Check(cause, result, met, not_met, ids)

    checks = [
        check(C.DC_SHORTAGE, ("DC on-hand below 35% of baseline", r_dc < 0.35), [
            ("store order fill rate below 70%", store_fill is not None and store_fill < 70),
            ("manufacturer fill rate to DC at least 90%", mfr_fill >= 90),
        ], ["S9", "S8", "S10"]),
        check(C.ORDER_CUT, ("manufacturer fill rate to DC below 75%", mfr_fill < 75), [
            ("store order fill rate below 80%", store_fill is not None and store_fill < 80),
            ("DC on-hand below 85% of baseline", r_dc < 0.85),
        ], ["S10", "S8", "S9"]),
        check(C.STORE_ORDERING_MISS, ("store orders below 30% of baseline", r_orders < 0.3), [
            ("store on-hand below 30% of baseline", r_oh < 0.3),
            ("DC on-hand at least 60% of baseline", r_dc >= 0.6),
            ("SKU still ranged", ranged == "Y"),
        ], ["S7", "S5", "S9", "S13"]),
        check(C.PHANTOM_INVENTORY, ("store on-hand at least 50% of baseline while POS below 30% of baseline",
                                    r_oh >= 0.5 and r_pos < 0.3), [
            ("store orders below 30% of baseline", r_orders < 0.3),
            ("store on-hand unchanged since the week before the decline", s["S5"].value == s["S6"].value),
            ("audit found the SKU off shelf", on_shelf == "N"),
        ], ["S5", "S3", "S7", "S6", "S11"]),
        check(C.SHELF_EXECUTION_GAP, ("audit found the SKU off shelf or below planogram facings",
                                      bool(audited and ranged == "Y" and (on_shelf == "N" or short_facings))), [
            ("store on-hand at least 90% of baseline", r_oh >= 0.9),
            ("deliveries continuing (fill rate at least 80%, orders at least 50% of baseline)",
             store_fill is not None and store_fill >= 80 and r_orders >= 0.5),
        ], ["S11", "S12", "S5", "S8", "S7"]),
        check(C.DELISTING, ("audit shows SKU not ranged or zero planogram facings",
                            bool(audited and (ranged == "N" or plan == 0))), [
            ("store orders below 30% of baseline", r_orders < 0.3),
            ("audit found the SKU off shelf", on_shelf == "N"),
        ], ["S13", "S12", "S7", "S11"]),
        check(C.PROMO_DEMAND_SPIKE, ("promotion active in affected weeks", s["S14"].value == "Y"), [
            ("peak POS units at least 1.5x baseline", r_peak >= 1.5),
            ("store orders not raised above 1.2x baseline", r_orders <= 1.2),
            ("store on-hand below 30% of baseline", r_oh < 0.3),
        ], ["S14", "S4", "S7", "S5"] + (["S15"] if "S15" in s else [])),
    ]

    conflicts = []
    if r_osa_drop >= 15 and r_pos >= 0.9:
        conflicts.append("POS units held at 90% of baseline or more although OSA fell 15+ points (S1, S3)")
    if audited and on_shelf == "Y" and plan and facings >= plan and s["S1"].value < 70:
        conflicts.append("Audit found full planogram facings on shelf although OSA is below 70% (S1, S11, S12)")
    if s["S5"].value == 0 and r_pos >= 0.9:
        conflicts.append("System on-hand is zero while sales held at baseline (S5, S3)")

    return EvidencePack(item, sig, checks, conflicts)
