"""可解释候选模型与朴素季节基准。

所有模型预测"单位面积日需求率"（occupancy rate，需求面积日 / 可出租面积日），
需求面积日 = 需求率 × 目标周可出租面积日。闭区周（rentable_days=0）在进入
训练集前已被 pipeline 剔除，模型永远不会把无供给的零成交学成零需求。

fair 参数三态：True/False 表示起报时展会状态已确认；None 表示起报时未确认，
模型按训练期展会基准率边际化，绝不偷看未来实际值。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from market_space_twin.simulation.twin_store import DomainError

MIN_ZONE_POINTS = 3
SEASON = 4


@dataclass(frozen=True, slots=True)
class TrainPoint:
    zone_id: str
    week_index: int
    rate: float
    fair: bool


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


class FittedModel:
    """冻结模型：parameters 为 JSON 可序列化的全部参数，可重建、可审计。"""

    name = "base"
    benchmark = False

    def __init__(self, parameters: dict[str, Any]):
        self.parameters = parameters

    def predict_rate(self, zone_id: str, week_index: int, fair: bool | None) -> float:
        raise NotImplementedError


class ZoneFairModel(FittedModel):
    """分区基线 + 展会 uplift：rate = base[zone] + uplift * P(fair)。"""

    name = "zone_fair"

    def predict_rate(self, zone_id: str, week_index: int, fair: bool | None) -> float:
        params = self.parameters
        base = params["zone_base_rate"].get(zone_id, params["pooled_base_rate"])
        if fair is None:
            fair_share = params["fair_base_rate"]
        else:
            fair_share = 1.0 if fair else 0.0
        return base + params["fair_uplift"] * fair_share


class ZoneMeanModel(FittedModel):
    """分区历史均值，不使用展会特征。"""

    name = "zone_mean"

    def predict_rate(self, zone_id: str, week_index: int, fair: bool | None) -> float:
        params = self.parameters
        return params["zone_mean_rate"].get(zone_id, params["global_mean_rate"])


class SeasonalNaiveModel(FittedModel):
    """朴素季节基准：取同一相位（week_index % 4）最近一周的实测需求率。"""

    name = "seasonal_naive"
    benchmark = True

    def predict_rate(self, zone_id: str, week_index: int, fair: bool | None) -> float:
        params = self.parameters
        phase = str(week_index % params["season"])
        zone_phases = params["phase_rate"].get(zone_id, {})
        value = zone_phases.get(phase)
        if value is not None:
            return value
        return params["zone_mean_rate"].get(zone_id, params["global_mean_rate"])


def fit_zone_fair(points: list[TrainPoint]) -> ZoneFairModel:
    fair_rates = [p.rate for p in points if p.fair]
    plain_rates = [p.rate for p in points if not p.fair]
    global_mean = _mean([p.rate for p in points])
    pooled_base = _mean(plain_rates) if plain_rates else global_mean
    uplift = _mean(fair_rates) - pooled_base if fair_rates else 0.0
    zone_base: dict[str, float] = {}
    for zone in sorted({p.zone_id for p in points}):
        own = [p.rate for p in points if p.zone_id == zone and not p.fair]
        zone_base[zone] = _mean(own) if len(own) >= MIN_ZONE_POINTS else pooled_base
    return ZoneFairModel({
        "zone_base_rate": zone_base,
        "pooled_base_rate": pooled_base,
        "fair_uplift": uplift,
        "fair_base_rate": len(fair_rates) / len(points) if points else 0.0,
        "global_mean_rate": global_mean,
        "min_zone_points": MIN_ZONE_POINTS,
    })


def fit_zone_mean(points: list[TrainPoint]) -> ZoneMeanModel:
    global_mean = _mean([p.rate for p in points])
    zone_mean: dict[str, float] = {}
    for zone in sorted({p.zone_id for p in points}):
        own = [p.rate for p in points if p.zone_id == zone]
        zone_mean[zone] = _mean(own) if len(own) >= MIN_ZONE_POINTS else global_mean
    return ZoneMeanModel({"zone_mean_rate": zone_mean, "global_mean_rate": global_mean})


def fit_seasonal_naive(points: list[TrainPoint]) -> SeasonalNaiveModel:
    phase_rate: dict[str, dict[str, float]] = {}
    for point in sorted(points, key=lambda p: p.week_index):
        phase = str(point.week_index % SEASON)
        phase_rate.setdefault(point.zone_id, {})[phase] = point.rate
    global_mean = _mean([p.rate for p in points])
    zone_mean: dict[str, float] = {}
    for zone in sorted({p.zone_id for p in points}):
        own = [p.rate for p in points if p.zone_id == zone]
        zone_mean[zone] = _mean(own) if own else global_mean
    return SeasonalNaiveModel({
        "phase_rate": phase_rate,
        "zone_mean_rate": zone_mean,
        "global_mean_rate": global_mean,
        "season": SEASON,
    })


_FITERS = {
    ZoneFairModel.name: (ZoneFairModel, fit_zone_fair),
    ZoneMeanModel.name: (ZoneMeanModel, fit_zone_mean),
    SeasonalNaiveModel.name: (SeasonalNaiveModel, fit_seasonal_naive),
}

CANDIDATE_ORDER = [ZoneFairModel.name, ZoneMeanModel.name, SeasonalNaiveModel.name]
BENCHMARK_NAME = SeasonalNaiveModel.name


def fit_model(name: str, points: list[TrainPoint]) -> FittedModel:
    if name not in _FITERS:
        raise DomainError(f"未知模型: {name!r}")
    return _FITERS[name][1](points)


def model_from_params(name: str, parameters: dict[str, Any]) -> FittedModel:
    """从冻结参数重建模型，用于审计复算。"""
    if name not in _FITERS:
        raise DomainError(f"未知模型: {name!r}")
    return _FITERS[name][0](parameters)
