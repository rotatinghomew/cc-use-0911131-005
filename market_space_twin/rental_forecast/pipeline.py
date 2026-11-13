"""招商需求预测流水线。

职责链：输入校验 → 起报时可见性裁剪 → 空间版本（拆合）→ 真实起报时间回测
→ 选模型并冻结 → 分区/总计预测 → 在同一冻结模型上应用场景 → 落账。

防泄漏约定：
- 训练只用 known_at <= 起报日的行；闭区周（rentable_days=0）是无供给而非
  无需求，从需求训练中剔除并记录。
- 展会标志只在 fair_known_at <= 起报日时可用，否则按训练期基准率边际化。
- 回测折点的检验周全部 <= 正式起报日，选模型看不见正式预测周。
- 新区（拆合产生、无历史）只允许借力父区需求率并加宽区间，或拒绝预测。
"""
from __future__ import annotations

import json
import math
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from market_space_twin.simulation.twin_store import DomainError, fingerprint, utc_now
from market_space_twin.rental_forecast.models import (
    BENCHMARK_NAME,
    CANDIDATE_ORDER,
    TrainPoint,
    fit_model,
)
from market_space_twin.rental_forecast.panel import PanelInput, Scenario, load_input
from market_space_twin.rental_forecast.store import ForecastStore

CODE_VERSION = "1"
HORIZON_WEEKS = 4
INTERVAL_LEVEL = 0.8
INTERVAL_Q_LOW = 0.1
INTERVAL_Q_HIGH = 0.9
MIN_TRAIN_WEEKS = 10
N_FOLDS = 5
MIN_FOLDS = 2
DEFAULT_RENTABLE_DAYS = 7.0
PLAN_SCENARIO_ID = "base"

SUPPORT_SUPPORTED = "supported"
SUPPORT_SPACE_CHANGED = "supported_space_changed"
SUPPORT_BORROWED = "borrowed_parent_low_support"
SUPPORT_REFUSED = "refused_no_history"
INTERVAL_WIDENING = {
    SUPPORT_SUPPORTED: 1.0,
    SUPPORT_SPACE_CHANGED: 1.25,
    SUPPORT_BORROWED: 1.5,
}


def _r3(value: float | None) -> float | None:
    return None if value is None else round(float(value), 3)


def _r2(value: float | None) -> float | None:
    return None if value is None else round(float(value), 2)


def _round_params(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, dict):
        return {k: _round_params(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_round_params(v) for v in value]
    return value


def _quantile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = q * (len(sorted_values) - 1)
    low = int(math.floor(pos))
    high = int(math.ceil(pos))
    if low == high:
        return sorted_values[low]
    frac = pos - low
    return sorted_values[low] * (1 - frac) + sorted_values[high] * frac


class Pipeline:
    def __init__(self, data: PanelInput):
        self.data = data
        self.weeks = data.weeks
        self.week_index = {week: i for i, week in enumerate(self.weeks)}
        self.origin = data.forecast_origin
        self.rows_by_zone_week = {(row.zone_id, row.week): row for row in data.rows}

    # ---------- 基础视图 ----------

    def index_of(self, week: date) -> int:
        if week in self.week_index:
            return self.week_index[week]
        last = self.weeks[-1]
        return self.week_index[last] + round((week - last).days / 7)

    def horizon_weeks(self) -> list[date]:
        future = [week for week in self.weeks if week > self.origin]
        horizon = list(future[:HORIZON_WEEKS])
        while len(horizon) < HORIZON_WEEKS:
            base = horizon[-1] if horizon else max(list(self.weeks) + [self.origin])
            horizon.append(base + timedelta(days=7))
        return horizon

    def training_points(self, origin: date) -> tuple[list[TrainPoint], list[dict], list[dict]]:
        """起报日可见的训练点；返回 (点, 闭区剔除, 展会未确认剔除)。"""
        points: list[TrainPoint] = []
        closed: list[dict] = []
        unconfirmed: list[dict] = []
        for row in self.data.rows:
            if row.known_at > origin:
                continue
            if not row.is_open:
                closed.append({"zone_id": row.zone_id, "week": row.week.isoformat(), "reason": "rentable_days=0 闭区，无供给不等于无需求"})
                continue
            if row.fair_known_at > origin:
                unconfirmed.append({"zone_id": row.zone_id, "week": row.week.isoformat(), "reason": "展会资料起报时未确认"})
                continue
            points.append(TrainPoint(row.zone_id, self.index_of(row.week), row.occupancy_rate, row.fair))
        return points, closed, unconfirmed

    def fair_visibility(self, weeks: list[date], origin: date) -> dict[date, bool | None]:
        """每个目标周在起报时是否已确认展会状态（展会为场馆周级事件）。"""
        visibility: dict[date, bool | None] = {}
        for week in weeks:
            flags = [
                row.fair
                for row in self.data.rows
                if row.week == week and row.fair_known_at <= origin
            ]
            visibility[week] = flags[0] if flags else None
        return visibility

    def rent_map(self, origin: date) -> dict[str, float]:
        rent: dict[str, float] = {}
        for row in sorted(self.data.rows, key=lambda r: r.week):
            if row.known_at <= origin:
                rent[row.zone_id] = row.rent_per_area_day
        return rent

    def rentable_days_for(self, zone_id: str, week: date) -> tuple[float, bool]:
        row = self.rows_by_zone_week.get((zone_id, week))
        if row is not None:
            return row.rentable_days, False
        return DEFAULT_RENTABLE_DAYS, True

    # ---------- 空间版本 ----------

    def space_version(self, train_zones: set[str]) -> dict[str, Any]:
        """按起报时已确认的拆合事件推导预测用空间版本。"""
        area: dict[str, float] = {}
        for row in sorted(self.data.rows, key=lambda r: r.week):
            if row.known_at <= self.origin:
                area[row.zone_id] = row.effective_area
        applied: list[dict] = []
        pending: list[dict] = []
        borrow: dict[str, dict] = {}
        changed: dict[str, dict] = {}
        for event in sorted(self.data.topology, key=lambda e: (e.effective_at, e.child, e.parent)):
            if event.known_at > self.origin:
                pending.append({"event": event.event_id(), "reason": "起报时未确认，未应用"})
                continue
            if event.effective_at > self.origin:
                pending.append({"event": event.event_id(), "reason": "起报日后才生效，未应用"})
                continue
            parent_area = area.get(event.parent)
            if parent_area is None:
                pending.append({"event": event.event_id(), "reason": "父区在起报时无已知面积，未应用"})
                continue
            share_area = parent_area * event.area_share
            area[event.parent] = parent_area - share_area
            area[event.child] = area.get(event.child, 0.0) + share_area
            applied.append({
                "event": event.event_id(),
                "parent": event.parent,
                "child": event.child,
                "area_share": event.area_share,
                "parent_area_before": round(parent_area, 6),
                "parent_area_after": round(area[event.parent], 6),
                "child_area": round(area[event.child], 6),
            })
            changed[event.parent] = {"before": round(parent_area, 6), "after": round(area[event.parent], 6)}
            if event.child not in train_zones:
                borrow[event.child] = {"parent": event.parent, "area_share": event.area_share}
        zones = sorted(area)
        definition = {
            "zones": {zone: round(area[zone], 6) for zone in zones},
            "applied_events": [item["event"] for item in applied],
            "borrow": borrow,
        }
        return {
            "space_version_id": "sv-" + fingerprint(definition)[:16],
            "zones": definition["zones"],
            "applied_events": applied,
            "pending_events": pending,
            "borrow": borrow,
            "changed_zones": changed,
            "definition": definition,
        }

    # ---------- 回测 ----------

    def select_fold_origins(self) -> list[int]:
        idx_origin = max(i for i, week in enumerate(self.weeks) if week <= self.origin)
        usable: list[int] = []
        for i in range(len(self.weeks)):
            if i + HORIZON_WEEKS > idx_origin:
                break  # 检验周必须全部 <= 正式起报日
            points, _, _ = self.training_points(self.weeks[i])
            per_zone: dict[str, int] = {}
            for point in points:
                per_zone[point.zone_id] = per_zone.get(point.zone_id, 0) + 1
            # 只要求"当时已有可训练历史的区"满足最小周数；
            # 完全没有开放周的区会在预测时被拒绝，不应阻塞回测。
            if per_zone and all(count >= MIN_TRAIN_WEEKS for count in per_zone.values()):
                usable.append(i)
        if len(usable) <= N_FOLDS:
            return usable
        step = (len(usable) - 1) / (N_FOLDS - 1)
        return sorted({usable[round(k * step)] for k in range(N_FOLDS)})

    def backtest(self, fold_indices: list[int]) -> tuple[list[dict], dict[str, dict]]:
        accumulators = {
            name: {"abs": [], "sq": [], "residuals": [], "n": 0} for name in CANDIDATE_ORDER
        }
        folds_out: list[dict] = []
        for fold_no, i in enumerate(fold_indices, start=1):
            origin_w = self.weeks[i]
            points, closed, unconfirmed = self.training_points(origin_w)
            fitted = {name: fit_model(name, points) for name in CANDIDATE_ORDER}
            test_weeks = self.weeks[i + 1: i + 1 + HORIZON_WEEKS]
            test_rows = [
                row for row in self.data.rows if row.week in set(test_weeks) and row.is_open
            ]
            fold_metrics: dict[str, dict] = {}
            for name, model in fitted.items():
                abs_err: list[float] = []
                sq_err: list[float] = []
                for row in test_rows:
                    fair = row.fair if row.fair_known_at <= origin_w else None
                    rate = max(0.0, model.predict_rate(row.zone_id, self.index_of(row.week), fair))
                    prediction = rate * row.capacity
                    err = prediction - row.occupied_area_days
                    abs_err.append(abs(err))
                    sq_err.append(err * err)
                    acc = accumulators[name]
                    acc["abs"].append(abs(err))
                    acc["sq"].append(err * err)
                    acc["residuals"].append(err / row.capacity)
                    acc["n"] += 1
                fold_metrics[name] = {
                    "mae": round(sum(abs_err) / len(abs_err), 3) if abs_err else None,
                    "n": len(abs_err),
                }
            train_weeks = sorted({self.weeks[p.week_index] if p.week_index < len(self.weeks) else self.weeks[-1] for p in points})
            folds_out.append({
                "fold_id": f"fold-{fold_no}",
                "origin": origin_w.isoformat(),
                "train": {
                    "rows": len(points),
                    "zones": len({p.zone_id for p in points}),
                    "first_week": train_weeks[0].isoformat() if train_weeks else None,
                    "last_week": train_weeks[-1].isoformat() if train_weeks else None,
                },
                "test_weeks": [w.isoformat() for w in test_weeks],
                "excluded_closed_weeks": closed,
                "excluded_fair_unconfirmed_rows": unconfirmed,
                "metrics": fold_metrics,
            })
        summary: dict[str, dict] = {}
        for name, acc in accumulators.items():
            n = acc["n"]
            if n == 0:
                summary[name] = {"mae": None, "rmse": None, "n": 0}
                continue
            residuals = sorted(acc["residuals"])
            q_low = _quantile(residuals, INTERVAL_Q_LOW)
            q_high = _quantile(residuals, INTERVAL_Q_HIGH)
            coverage = sum(1 for r in residuals if q_low <= r <= q_high) / n
            summary[name] = {
                "mae": round(sum(acc["abs"]) / n, 3),
                "rmse": round(math.sqrt(sum(acc["sq"]) / n), 3),
                "n": n,
                "residual_q_low": round(q_low, 6),
                "residual_q_high": round(q_high, 6),
                "interval_coverage": round(coverage, 3),
            }
        return folds_out, summary

    # ---------- 预测 ----------

    def support_status(self, zone: str, version: dict, train_zones: set[str]) -> str:
        if zone in train_zones:
            return SUPPORT_SPACE_CHANGED if zone in version["changed_zones"] else SUPPORT_SUPPORTED
        if zone in version["borrow"]:
            return SUPPORT_BORROWED
        return SUPPORT_REFUSED

    def base_rows(self, frozen, version: dict, horizon: list[date], q_low: float, q_high: float) -> list[dict]:
        rent = self.rent_map(self.origin)
        visibility = self.fair_visibility(horizon, self.origin)
        rows: list[dict] = []
        for zone in sorted(version["zones"]):
            status = self.support_status(zone, version, self._train_zones)
            if status == SUPPORT_REFUSED:
                for week in horizon:
                    rows.append({
                        "zone_id": zone,
                        "week": week.isoformat(),
                        "demand_area_days": None,
                        "revenue": None,
                        "interval": None,
                        "support_status": status,
                        "capacity_cap": None,
                        "rent_per_area_day": None,
                        "fair_status": None,
                        "drivers": ["无自身历史且无父区可借力，拒绝预测，不输出伪造置信度"],
                    })
                continue
            model_zone = version["borrow"][zone]["parent"] if status == SUPPORT_BORROWED else zone
            zone_rent = rent.get(zone, rent.get(model_zone))
            widening = INTERVAL_WIDENING[status]
            for week in horizon:
                rentable_days, defaulted = self.rentable_days_for(zone, week)
                capacity = version["zones"][zone] * rentable_days
                fair = visibility.get(week)
                rate = max(0.0, frozen.predict_rate(model_zone, self.index_of(week), fair))
                demand = rate * capacity
                low = max(0.0, (rate + q_low * widening) * capacity)
                high = max(0.0, (rate + q_high * widening) * capacity)
                drivers: list[str] = []
                if status == SUPPORT_SPACE_CHANGED:
                    change = version["changed_zones"][zone]
                    drivers.append(f"拆合后面积 {change['before']}→{change['after']} 平方米，需求率沿用自身历史")
                if status == SUPPORT_BORROWED:
                    parent = version["borrow"][zone]["parent"]
                    drivers.append(f"无自身历史，借力父区 {parent} 的单位面积需求率（area_share={version['borrow'][zone]['area_share']}）")
                if defaulted:
                    drivers.append(f"无 panel 记录，可出租天数按 {DEFAULT_RENTABLE_DAYS:.0f} 天假设")
                if fair is True:
                    fair_status = "confirmed_fair"
                    drivers.append("展会已确认，需求率含展会上调")
                elif fair is False:
                    fair_status = "confirmed_no_fair"
                    drivers.append("已确认无展会")
                else:
                    fair_status = "unconfirmed"
                    if "fair_base_rate" in frozen.parameters:
                        drivers.append(
                            f"展会起报时未确认，按训练期基准率 {frozen.parameters['fair_base_rate']:.2f} 边际化"
                        )
                    else:
                        drivers.append("展会起报时未确认；所选模型不依赖展会特征，按历史同相位规律外推")
                rows.append({
                    "zone_id": zone,
                    "week": week.isoformat(),
                    "demand_area_days": _r3(demand),
                    "interval": {"lower": _r3(low), "upper": _r3(high), "level": INTERVAL_LEVEL},
                    "support_status": status,
                    "capacity_cap": _r3(capacity),
                    "rent_per_area_day": zone_rent,
                    "fair_status": fair_status,
                    "model_zone": model_zone,
                    "drivers": drivers,
                })
        return rows

    # ---------- 情景 ----------

    def apply_scenario(self, scenario: Scenario, base_rows: list[dict]) -> dict[str, Any]:
        out_rows: list[dict] = []
        for row in base_rows:
            demand = row["demand_area_days"]
            if demand is None:
                out_rows.append({
                    "zone_id": row["zone_id"],
                    "week": row["week"],
                    "scenario_id": scenario.scenario_id,
                    "demand_area_days": None,
                    "served_area_days": None,
                    "capacity_cap": None,
                    "truncated_area_days": None,
                    "revenue": None,
                    "interval": None,
                    "served_interval": None,
                    "revenue_interval": None,
                    "support_status": row["support_status"],
                    "drivers": row["drivers"],
                })
                continue
            cap = row["capacity_cap"] * scenario.area_multiplier
            served = min(demand, cap)
            truncated = max(0.0, demand - cap)
            rent = row["rent_per_area_day"] * scenario.rent_multiplier
            served_low = min(row["interval"]["lower"], cap)
            served_high = min(row["interval"]["upper"], cap)
            drivers = list(row["drivers"])
            if truncated > 0:
                drivers.append(f"需求超过方案上限，截断 {_r3(truncated)} 面积日")
            if scenario.rent_multiplier != 1:
                drivers.append(f"租金×{scenario.rent_multiplier} 仅机械换算，无因果证据")
            out_rows.append({
                "zone_id": row["zone_id"],
                "week": row["week"],
                "scenario_id": scenario.scenario_id,
                "demand_area_days": _r3(demand),
                "served_area_days": _r3(served),
                "capacity_cap": _r3(cap),
                "truncated_area_days": _r3(truncated),
                "revenue": _r2(served * rent),
                "interval": row["interval"],
                "served_interval": {
                    "lower": _r3(served_low),
                    "upper": _r3(served_high),
                    "level": INTERVAL_LEVEL,
                },
                "revenue_interval": {
                    "lower": _r2(served_low * rent),
                    "upper": _r2(served_high * rent),
                    "level": INTERVAL_LEVEL,
                },
                "support_status": row["support_status"],
                "fair_status": row.get("fair_status"),
                "drivers": drivers,
            })
        totals = self._totals(out_rows)
        return {
            "scenario_id": scenario.scenario_id,
            "area_multiplier": scenario.area_multiplier,
            "rent_multiplier": scenario.rent_multiplier,
            "rows": out_rows,
            "totals": totals,
            "drivers": [],
        }

    @staticmethod
    def _totals(rows: list[dict]) -> dict[str, Any]:
        valid = [row for row in rows if row["demand_area_days"] is not None]
        refused = len(rows) - len(valid)
        totals = {
            "demand_area_days": _r3(sum(row["demand_area_days"] for row in valid)),
            "served_area_days": _r3(sum(row["served_area_days"] for row in valid)),
            "truncated_area_days": _r3(sum(row["truncated_area_days"] for row in valid)),
            "revenue": _r2(sum(row["revenue"] for row in valid)),
            "interval": {
                "lower": _r3(sum(row["interval"]["lower"] for row in valid)),
                "upper": _r3(sum(row["interval"]["upper"] for row in valid)),
                "level": INTERVAL_LEVEL,
            },
            "served_interval": {
                "lower": _r3(sum(row["served_interval"]["lower"] for row in valid)),
                "upper": _r3(sum(row["served_interval"]["upper"] for row in valid)),
                "level": INTERVAL_LEVEL,
            },
            "revenue_interval": {
                "lower": _r2(sum(row["revenue_interval"]["lower"] for row in valid)),
                "upper": _r2(sum(row["revenue_interval"]["upper"] for row in valid)),
                "level": INTERVAL_LEVEL,
            },
        }
        if refused:
            totals["refused_rows"] = refused
        return totals

    # ---------- 主流程 ----------

    def run(self) -> dict[str, Any]:
        horizon = self.horizon_weeks()
        points, closed, unconfirmed = self.training_points(self.origin)
        self._train_zones = {p.zone_id for p in points}
        version = self.space_version(self._train_zones)
        fold_indices = self.select_fold_origins()
        if len(fold_indices) < MIN_FOLDS:
            raise DomainError(
                f"起报时已知历史只能形成 {len(fold_indices)} 个回测折点（至少需要 {MIN_FOLDS} 个），"
                "拒绝给出无法验证的预测"
            )
        folds, backtest_summary = self.backtest(fold_indices)
        selectable = [name for name in CANDIDATE_ORDER if backtest_summary[name].get("mae") is not None]
        if not selectable:
            raise DomainError("回测没有产生任何可用误差，无法选模型")
        selected = min(selectable, key=lambda n: (backtest_summary[n]["mae"], CANDIDATE_ORDER.index(n)))
        frozen = fit_model(selected, points)
        self._frozen_parameters = frozen.parameters
        q_low = backtest_summary[selected]["residual_q_low"]
        q_high = backtest_summary[selected]["residual_q_high"]

        base_rows = self.base_rows(frozen, version, horizon, q_low, q_high)
        scenarios = list(self.data.scenarios) or [Scenario(PLAN_SCENARIO_ID, 1.0, 1.0)]
        scenario_results = [self.apply_scenario(s, base_rows) for s in scenarios]
        base_result = next(
            (r for r in scenario_results if r["area_multiplier"] == 1 and r["rent_multiplier"] == 1),
            self.apply_scenario(Scenario(PLAN_SCENARIO_ID, 1.0, 1.0), base_rows),
        )
        forecasts = []
        for row in base_result["rows"]:
            forecasts.append({
                "zone_id": row["zone_id"],
                "week": row["week"],
                "scenario_id": row["scenario_id"],
                "demand_area_days": row["demand_area_days"],
                "revenue": row["revenue"],
                "interval": row["interval"],
                "support_status": row["support_status"],
                "served_area_days": row["served_area_days"],
                "capacity_cap": row["capacity_cap"],
                "truncated_area_days": row["truncated_area_days"],
                "served_interval": row["served_interval"],
                "revenue_interval": row["revenue_interval"],
                "fair_status": row.get("fair_status"),
                "drivers": row["drivers"],
            })
        base_totals = base_result["totals"]
        for result in scenario_results:
            result["drivers"] = self._scenario_drivers(result, base_totals)
            result["vs_base"] = {
                "delta_revenue": _r2(result["totals"]["revenue"] - base_totals["revenue"]),
                "delta_served_area_days": _r3(result["totals"]["served_area_days"] - base_totals["served_area_days"]),
            }

        input_fingerprint = fingerprint(self.data.raw)
        run_id = "frc-" + fingerprint({
            "input": input_fingerprint,
            "origin": self.origin.isoformat(),
            "code_version": CODE_VERSION,
        })[:16]
        model_section = {
            "selected": selected,
            "selection_rule": "各回测折点检验周 MAE(需求面积日) 均值最小；并列时按候选顺序取前者",
            "candidates": [
                {
                    "name": name,
                    "benchmark": name == BENCHMARK_NAME,
                    "mae": backtest_summary[name].get("mae"),
                    "rmse": backtest_summary[name].get("rmse"),
                    "interval_coverage": backtest_summary[name].get("interval_coverage"),
                    "n_points": backtest_summary[name].get("n"),
                }
                for name in CANDIDATE_ORDER
            ],
            "frozen": {
                "name": selected,
                "parameters": _round_params(frozen.parameters),
                "trained_at_origin": self.origin.isoformat(),
                "training_cutoff": f"known_at <= {self.origin.isoformat()}",
                "training_rows": len(points),
                "weeks_per_zone": {
                    zone: sum(1 for p in points if p.zone_id == zone)
                    for zone in sorted(self._train_zones)
                },
                "excluded_closed_weeks": closed,
                "excluded_fair_unconfirmed_rows": unconfirmed,
            },
            "interval": {
                "method": "回测折点样本外残差分位数（需求率空间），按目标周可出租面积日换算",
                "level": INTERVAL_LEVEL,
                "residual_q_low": q_low,
                "residual_q_high": q_high,
                "support_widening": INTERVAL_WIDENING,
            },
            "input_fingerprint": input_fingerprint,
            "code_version": CODE_VERSION,
        }
        drivers = self._top_drivers(selected, backtest_summary, version, horizon)
        warnings = self._warnings(version, horizon, closed, unconfirmed, scenario_results, backtest_summary, selected)
        return {
            "run_id": run_id,
            "generated_at": utc_now(),
            "code_version": CODE_VERSION,
            "forecast_origin": self.origin.isoformat(),
            "horizon_weeks": [w.isoformat() for w in horizon],
            "space_version_id": version["space_version_id"],
            "space_version": version["definition"] | {"pending_events": version["pending_events"]},
            "input_fingerprint": input_fingerprint,
            "scenario_definitions": [
                {"scenario_id": s.scenario_id, "area_multiplier": s.area_multiplier, "rent_multiplier": s.rent_multiplier}
                for s in scenarios
            ],
            "folds": folds,
            "model": model_section,
            "forecasts": forecasts,
            "scenario_results": scenario_results,
            "drivers": drivers,
            "warnings": warnings,
            "notes": self._notes(),
        }

    # ---------- 解释与警告 ----------

    def _top_drivers(self, selected: str, summary: dict, version: dict, horizon: list[date]) -> list[dict]:
        if selected == BENCHMARK_NAME:
            alternatives = [n for n in CANDIDATE_ORDER if n != BENCHMARK_NAME and summary[n].get("mae") is not None]
            runner_up = min(alternatives, key=lambda n: summary[n]["mae"]) if alternatives else None
            detail = f"真实起报时间回测后朴素季节基准 {BENCHMARK_NAME} 本身最优（MAE {summary[selected]['mae']} 面积日）"
            if runner_up:
                detail += f"，优于最近的可解释候选 {runner_up}（MAE {summary[runner_up]['mae']} 面积日）"
            detail += "；按历史同相位规律外推，隐含展会周期延续的假设"
            drivers = [{"factor": "model_selection", "detail": detail}]
        else:
            drivers = [{
                "factor": "model_selection",
                "detail": (
                    f"回测选择 {selected}（MAE {summary[selected]['mae']} 面积日）；"
                    f"朴素季节基准 {BENCHMARK_NAME} MAE {summary[BENCHMARK_NAME]['mae']} 面积日"
                ),
            }]
        for event in version["applied_events"]:
            drivers.append({
                "factor": "space_split",
                "detail": (
                    f"{event['event']}：{event['parent']} 面积 {event['parent_area_before']}→"
                    f"{event['parent_area_after']}，{event['child']} 获得 {event['child_area']} 平方米；"
                    "假设单位面积需求率可移植，拆分前后合计需求不变"
                ),
            })
        visibility = self.fair_visibility(horizon, self.origin)
        confirmed = [w.isoformat() for w, v in visibility.items() if v is not None]
        unconfirmed_weeks = [w.isoformat() for w, v in visibility.items() if v is None]
        if unconfirmed_weeks:
            handling = (
                "按训练期基准率边际化"
                if "fair_base_rate" in self._frozen_parameters
                else "所选模型不依赖展会特征，按历史同相位规律外推"
            )
        else:
            handling = "无需处理"
        drivers.append({
            "factor": "fair_visibility",
            "detail": f"展会已确认周: {confirmed or '无'}；未确认周: {unconfirmed_weeks or '无'}（{handling}）",
        })
        return drivers

    def _scenario_drivers(self, result: dict, base_totals: dict) -> list[dict]:
        drivers: list[dict] = []
        if result["area_multiplier"] != 1:
            drivers.append({
                "factor": "area_multiplier",
                "detail": (
                    f"可出租上限×{result['area_multiplier']}：方案自身上限收紧，"
                    f"可成交面积日 {base_totals['served_area_days']}→{result['totals']['served_area_days']}，"
                    f"上限截断 {result['totals']['truncated_area_days']} 面积日"
                ),
            })
        if result["rent_multiplier"] != 1:
            drivers.append({
                "factor": "rent_multiplier",
                "detail": (
                    f"租金×{result['rent_multiplier']} 仅按机械换算作用于可收租金额；"
                    "历史中租金无变动，无法估计需求弹性，不构成提租增收的因果结论"
                ),
            })
        truncated = result["totals"]["truncated_area_days"]
        if truncated and truncated > 0:
            drivers.append({
                "factor": "capacity_truncation",
                "detail": f"需求超过方案上限合计截断 {truncated} 面积日；总计与区间按区-周先截断后求和",
            })
        return drivers

    def _warnings(self, version, horizon, closed, unconfirmed, scenario_results, summary, selected) -> list[str]:
        warnings: list[str] = []
        if closed:
            where = "、".join(f"{c['zone_id']}@{c['week']}" for c in closed)
            warnings.append(f"闭区周已从需求训练剔除（无供给不等于无需求）: {where}")
        if unconfirmed:
            warnings.append(f"{len(unconfirmed)} 行因展会资料起报时未确认而未参与训练")
        visibility = self.fair_visibility(horizon, self.origin)
        unknown_weeks = [w.isoformat() for w, v in visibility.items() if v is None]
        if unknown_weeks:
            if "fair_base_rate" in getattr(self, "_frozen_parameters", {}):
                warnings.append(f"预测周 {unknown_weeks} 的展会状态起报时未确认，已按训练期基准率边际化，未偷看未来实际值")
            else:
                warnings.append(
                    f"预测周 {unknown_weeks} 的展会状态起报时未确认；所选模型 {selected} 不依赖展会特征，"
                    "按历史同相位规律外推（隐含展会节奏延续的假设）"
                )
        for zone, info in version["borrow"].items():
            warnings.append(
                f"{zone} 无历史成交，预测借力父区 {info['parent']} 需求率并加宽区间"
                f"（{SUPPORT_BORROWED}），不得当作高置信度承诺"
            )
        for zone, change in version["changed_zones"].items():
            warnings.append(
                f"{zone} 拆合后面积 {change['before']}→{change['after']} 平方米；"
                "panel 中拆分生效后的旧面积记录未用于拟合，新区不偷用拆合后的实际业绩"
            )
        rent_scenarios = [r["scenario_id"] for r in scenario_results if r["rent_multiplier"] != 1]
        if rent_scenarios:
            warnings.append(
                f"情景 {rent_scenarios} 的租金倍率仅机械换算：训练期内租金无变动，"
                "无干预数据，不能声称提租会因果性地增加收入"
            )
        for result in scenario_results:
            truncated = result["totals"]["truncated_area_days"]
            if truncated and truncated > 0:
                warnings.append(
                    f"情景 {result['scenario_id']} 面积上限截断合计 {truncated} 面积日，"
                    "总计与区间为各区-周先截断后求和，右尾被方案上限削平"
                )
        coverage = summary[selected].get("interval_coverage")
        if coverage is not None:
            warnings.append(
                f"预测区间来自回测样本外残差（{INTERVAL_LEVEL:.0%} 水平），回测覆盖率 {coverage:.0%}；"
                "形态变化区与借力区已按支持度加宽"
            )
        refused = [
            zone for zone in version["zones"]
            if self.support_status(zone, version, self._train_zones) == SUPPORT_REFUSED
        ]
        if refused:
            warnings.append(f"区块 {refused} 无历史且无父区可借力，已拒绝预测（输出 null），不伪造置信度")
        for event in version["pending_events"]:
            warnings.append(f"拓扑事件 {event['event']} 未应用: {event['reason']}")
        return warnings

    @staticmethod
    def _notes() -> list[str]:
        return [
            "demand_area_days 为未截断的潜在需求面积日；served_area_days=min(需求, 方案上限)，不超过方案自身可出租上限",
            "revenue=served_area_days×租金×租金倍率；served_interval 为需求区间被方案上限截断后的面积日区间，revenue_interval 由其换算",
            "interval 为需求区间（回测残差分位，未截断）；总计区间=各区-周截断后求和，未建模跨区相关性",
            "训练可见性：known_at<=起报日 且展会 fair_known_at<=起报日；闭区周不计入需求训练",
            "所有情景在同一冻结模型上应用，面积倍率只改上限、租金倍率只改单价，需求预测本身不变",
            "本输出不含自动定价或自动签约，仅供招商方案比较",
        ]


def run_forecast(input_path: str | Path, output_path: str | Path, database: str | Path) -> dict[str, Any]:
    data = load_input(input_path)
    result = Pipeline(data).run()
    store = ForecastStore(database)
    store.record_run(result)
    output = Path(output_path)
    if output.parent and not output.parent.exists():
        output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result
