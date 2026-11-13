"""周面板输入的读取与校验。

输入契约（见 fixtures/analysis/说明.md）：
- panel 每行对应 zone_id 的一周；known_at 为成交周资料可用日，fair_known_at 为展会资料可用日。
- 字段缺省与数值零含义不同：必需字段缺失直接报错，不用零填充。
- rentable_days 为 0 表示闭区（无供给），不等于无需求。
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from market_space_twin.simulation.twin_store import DomainError


def parse_date(value: Any, field: str) -> date:
    try:
        return date.fromisoformat(str(value)[:10])
    except (ValueError, TypeError) as exc:
        raise DomainError(f"字段 {field} 不是有效 ISO 日期: {value!r}") from exc


def _num(value: Any, field: str, minimum: float | None = None) -> float:
    if isinstance(value, bool):
        raise DomainError(f"字段 {field} 必须是数值，得到布尔值")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise DomainError(f"字段 {field} 必须是数值: {value!r}") from exc
    if not math.isfinite(result):
        raise DomainError(f"字段 {field} 必须是有限数值: {value!r}")
    if minimum is not None and result < minimum:
        raise DomainError(f"字段 {field} 不能小于 {minimum}: {value!r}")
    return result


def _required(item: dict, key: str, index: int) -> Any:
    value = item.get(key)
    if value is None or value == "":
        raise DomainError(f"panel[{index}] 缺少必需字段 {key}（缺省不等于零）")
    return value


@dataclass(frozen=True, slots=True)
class PanelRow:
    zone_id: str
    week: date
    effective_area: float
    rentable_days: float
    occupied_area_days: float
    rent_per_area_day: float
    revenue: float
    fair: bool
    fair_known_at: date
    known_at: date

    @property
    def capacity(self) -> float:
        return self.effective_area * self.rentable_days

    @property
    def is_open(self) -> bool:
        return self.rentable_days > 0 and self.effective_area > 0

    @property
    def occupancy_rate(self) -> float | None:
        if self.capacity <= 0:
            return None
        return self.occupied_area_days / self.capacity


@dataclass(frozen=True, slots=True)
class TopologyEvent:
    parent: str
    child: str
    effective_at: date
    known_at: date
    area_share: float

    def event_id(self) -> str:
        return f"{self.parent}->{self.child}@{self.effective_at.isoformat()}"


@dataclass(frozen=True, slots=True)
class Scenario:
    scenario_id: str
    area_multiplier: float
    rent_multiplier: float


@dataclass(slots=True)
class PanelInput:
    forecast_origin: date
    rows: list[PanelRow]
    topology: list[TopologyEvent]
    scenarios: list[Scenario]
    seed: Any
    raw: dict[str, Any]

    @property
    def weeks(self) -> list[date]:
        return sorted({row.week for row in self.rows})

    @property
    def zones(self) -> list[str]:
        return sorted({row.zone_id for row in self.rows})


def _parse_row(item: Any, index: int) -> PanelRow:
    if not isinstance(item, dict):
        raise DomainError(f"panel[{index}] 必须是对象")
    zone_id = str(_required(item, "zone_id", index)).strip()
    if not zone_id:
        raise DomainError(f"panel[{index}] zone_id 不能为空")
    week = parse_date(_required(item, "week", index), f"panel[{index}].week")
    effective_area = _num(_required(item, "effective_area", index), f"panel[{index}].effective_area", 0.0)
    if effective_area <= 0:
        raise DomainError(f"panel[{index}].effective_area 必须为正: {effective_area!r}")
    rentable_days = _num(_required(item, "rentable_days", index), f"panel[{index}].rentable_days", 0.0)
    occupied = _num(_required(item, "occupied_area_days", index), f"panel[{index}].occupied_area_days", 0.0)
    rent = _num(_required(item, "rent_per_area_day", index), f"panel[{index}].rent_per_area_day", 0.0)
    capacity = effective_area * rentable_days
    if occupied > capacity + 1e-9:
        raise DomainError(
            f"panel[{index}] occupied_area_days={occupied} 超过供给上限 {capacity}，数据自相矛盾"
        )
    fair = _required(item, "fair", index)
    if not isinstance(fair, bool):
        raise DomainError(f"panel[{index}].fair 必须是布尔值: {fair!r}")
    fair_known_at = parse_date(_required(item, "fair_known_at", index), f"panel[{index}].fair_known_at")
    known_at = parse_date(_required(item, "known_at", index), f"panel[{index}].known_at")
    revenue_raw = item.get("revenue")
    revenue = occupied * rent if revenue_raw is None else _num(revenue_raw, f"panel[{index}].revenue", 0.0)
    return PanelRow(zone_id, week, effective_area, rentable_days, occupied, rent, revenue, fair, fair_known_at, known_at)


def _parse_topology(item: Any, index: int) -> TopologyEvent:
    if not isinstance(item, dict):
        raise DomainError(f"topology[{index}] 必须是对象")
    parent = str(item.get("parent") or "").strip()
    child = str(item.get("child") or "").strip()
    if not parent or not child or parent == child:
        raise DomainError(f"topology[{index}] 需要不同的 parent/child")
    share = _num(item.get("area_share"), f"topology[{index}].area_share")
    if not 0 < share <= 1:
        raise DomainError(f"topology[{index}].area_share 必须在 (0,1]: {share!r}")
    return TopologyEvent(
        parent=parent,
        child=child,
        effective_at=parse_date(item.get("effective_at"), f"topology[{index}].effective_at"),
        known_at=parse_date(item.get("known_at"), f"topology[{index}].known_at"),
        area_share=share,
    )


def _parse_scenario(item: Any, index: int) -> Scenario:
    if not isinstance(item, dict):
        raise DomainError(f"scenarios[{index}] 必须是对象")
    scenario_id = str(item.get("scenario_id") or "").strip()
    if not scenario_id:
        raise DomainError(f"scenarios[{index}] 缺少 scenario_id")
    area_multiplier = _num(item.get("area_multiplier", 1.0), f"scenarios[{index}].area_multiplier")
    rent_multiplier = _num(item.get("rent_multiplier", 1.0), f"scenarios[{index}].rent_multiplier")
    if area_multiplier <= 0 or rent_multiplier <= 0:
        raise DomainError(f"scenarios[{index}] 倍率必须为正")
    return Scenario(scenario_id, area_multiplier, rent_multiplier)


def parse_input(raw: Any) -> PanelInput:
    if not isinstance(raw, dict):
        raise DomainError("输入必须是 JSON 对象")
    origin_raw = raw.get("forecast_origin")
    if origin_raw is None:
        raise DomainError("缺少 forecast_origin")
    origin = parse_date(origin_raw, "forecast_origin")
    panel_raw = raw.get("panel")
    if not isinstance(panel_raw, list) or not panel_raw:
        raise DomainError("panel 必须是非空数组")
    rows = [_parse_row(item, i) for i, item in enumerate(panel_raw)]
    seen = set()
    for row in rows:
        key = (row.zone_id, row.week)
        if key in seen:
            raise DomainError(f"panel 存在重复行: zone_id={row.zone_id} week={row.week.isoformat()}")
        seen.add(key)
    topology = [_parse_topology(item, i) for i, item in enumerate(raw.get("topology") or [])]
    scenarios = [_parse_scenario(item, i) for i, item in enumerate(raw.get("scenarios") or [])]
    scenario_ids = [s.scenario_id for s in scenarios]
    if len(scenario_ids) != len(set(scenario_ids)):
        raise DomainError("scenarios 存在重复 scenario_id")
    return PanelInput(origin, rows, topology, scenarios, raw.get("seed"), raw)


def load_input(path: str | Path) -> PanelInput:
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DomainError(f"输入文件不是合法 JSON: {exc}") from exc
    return parse_input(raw)
