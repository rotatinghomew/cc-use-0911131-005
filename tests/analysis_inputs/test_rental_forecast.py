from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path

from market_space_twin import service
from market_space_twin.rental_forecast import run_forecast
from market_space_twin.rental_forecast.cli import main as cli_main
from market_space_twin.rental_forecast.store import ForecastStore

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "fixtures" / "analysis" / "input.json"
DEMO = ROOT / "fixtures" / "demo.json"
ORIGIN = "2026-07-16"


def build_panel(zones, weeks_count, *, occupancy=0.6, rent=3.0, closed_zones=()):
    """合成周面板：known_at=周起始+8 天，fair_known_at=周起始-10 天，展会每 4 周一次。"""
    rows = []
    start = date(2026, 1, 1)
    for zone, area in zones:
        for i in range(weeks_count):
            week = start + timedelta(days=7 * i)
            rentable = 0 if zone in closed_zones else 7
            fair = i % 4 == 0
            rate = occupancy + (0.15 if fair else 0.0)
            occupied = round(area * rentable * rate, 3)
            rows.append({
                "zone_id": zone,
                "week": week.isoformat(),
                "effective_area": area,
                "rentable_days": rentable,
                "occupied_area_days": occupied,
                "rent_per_area_day": rent,
                "revenue": round(occupied * rent, 2),
                "fair": fair,
                "fair_known_at": (week - timedelta(days=10)).isoformat(),
                "known_at": (week + timedelta(days=8)).isoformat(),
            })
    return rows


def write_input(directory, payload):
    path = Path(directory) / "input.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


class CliEndToEndTests(unittest.TestCase):
    def test_cli_output_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "forecast.json"
            database = Path(directory) / "ledger.sqlite3"
            rc = cli_main(["--input", str(FIXTURE), "--output", str(output), "--database", str(database)])
            self.assertEqual(rc, 0)
            result = json.loads(output.read_text(encoding="utf-8"))
            json.dumps(result, allow_nan=False)

        for key in ("folds", "model", "forecasts", "scenario_results", "warnings"):
            self.assertIn(key, result)
        self.assertEqual(result["forecast_origin"], ORIGIN)
        self.assertEqual(
            result["horizon_weeks"],
            ["2026-07-23", "2026-07-30", "2026-08-06", "2026-08-13"],
        )
        for row in result["forecasts"]:
            for field in ("zone_id", "week", "demand_area_days", "revenue", "interval", "support_status"):
                self.assertIn(field, row)
        zones = {row["zone_id"] for row in result["forecasts"]}
        self.assertEqual(zones, {"z0", "z1", "z2", "z3", "z4", "z5", "new-zone"})

        # 拆合后的空间版本：z0 与 new-zone 上限各 350 面积日
        z0 = next(row for row in result["forecasts"] if row["zone_id"] == "z0")
        self.assertEqual(z0["capacity_cap"], 350.0)
        self.assertEqual(z0["support_status"], "supported_space_changed")
        borrowed = next(row for row in result["forecasts"] if row["zone_id"] == "new-zone")
        self.assertEqual(borrowed["support_status"], "borrowed_parent_low_support")
        self.assertEqual(borrowed["capacity_cap"], 350.0)
        # 借力区间必须比同模型同上限的 z0 更宽，不允许伪造高置信度
        width_borrowed = borrowed["interval"]["upper"] - borrowed["interval"]["lower"]
        width_z0 = z0["interval"]["upper"] - z0["interval"]["lower"]
        self.assertGreater(width_borrowed, width_z0)

        # 展会可见性：第一周已确认，其余未确认
        first = next(row for row in result["forecasts"] if row["week"] == "2026-07-23" and row["zone_id"] == "z0")
        later = next(row for row in result["forecasts"] if row["week"] == "2026-07-30" and row["zone_id"] == "z0")
        self.assertEqual(first["fair_status"], "confirmed_no_fair")
        self.assertEqual(later["fair_status"], "unconfirmed")

        # 面积日不越过方案自身上限（两个情景都要满足）
        for scenario in result["scenario_results"]:
            for row in scenario["rows"]:
                self.assertLessEqual(row["served_area_days"], row["capacity_cap"] + 1e-6)
                self.assertLessEqual(row["served_interval"]["upper"], row["capacity_cap"] + 1e-6)

        # 候选与基准齐全，选择结果与回测一致
        candidates = {c["name"]: c for c in result["model"]["candidates"]}
        self.assertIn("seasonal_naive", candidates)
        self.assertIn("zone_fair", candidates)
        self.assertTrue(candidates["seasonal_naive"]["benchmark"])
        best = min(candidates.values(), key=lambda c: c["mae"])
        self.assertEqual(result["model"]["selected"], best["name"])

        # 闭区周不计入需求训练
        frozen = result["model"]["frozen"]
        self.assertEqual(frozen["training_rows"], 161)
        closed = {(c["zone_id"], c["week"]) for c in frozen["excluded_closed_weeks"]}
        self.assertIn(("z0", "2026-02-26"), closed)

        # 无因果警告必须存在
        self.assertTrue(any("因果" in w for w in result["warnings"]))

    def test_backtest_folds_respect_reporting_time(self):
        with tempfile.TemporaryDirectory() as directory:
            result = run_forecast(FIXTURE, Path(directory) / "o.json", Path(directory) / "d.sqlite3")
        raw = json.loads(FIXTURE.read_text(encoding="utf-8"))
        self.assertGreaterEqual(len(result["folds"]), 2)
        for fold in result["folds"]:
            # 折点起报日早于正式起报日，检验周全部不超过正式起报日
            self.assertLess(fold["origin"], ORIGIN)
            self.assertLessEqual(max(fold["test_weeks"]), ORIGIN)
            # 独立重算：该折点训练行数 == 起报时已知且开放的行数
            expected = sum(
                1
                for row in raw["panel"]
                if row["known_at"] <= fold["origin"] and row["rentable_days"] > 0
            )
            self.assertEqual(fold["train"]["rows"], expected)
        # 正式拟合同样只看见起报时已知资料
        expected_train = sum(
            1 for row in raw["panel"] if row["known_at"] <= ORIGIN and row["rentable_days"] > 0
        )
        self.assertEqual(result["model"]["frozen"]["training_rows"], expected_train)


class ImmutabilityTests(unittest.TestCase):
    def test_future_actuals_do_not_rewrite_past_fit(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "ledger.sqlite3"
            first = run_forecast(FIXTURE, Path(directory) / "o1.json", database)

            raw = json.loads(FIXTURE.read_text(encoding="utf-8"))
            for row in raw["panel"]:
                if row["week"] > ORIGIN:
                    row["occupied_area_days"] += 25  # 修改未来实际成交
                    row["revenue"] = row["occupied_area_days"] * row["rent_per_area_day"]
            tampered = write_input(directory, raw)
            second = run_forecast(tampered, Path(directory) / "o2.json", database)

            # 新输入产生新 run，但过去那次拟合的参数与预测不变
            self.assertNotEqual(first["run_id"], second["run_id"])
            self.assertEqual(
                first["model"]["frozen"]["parameters"],
                second["model"]["frozen"]["parameters"],
            )
            self.assertEqual(first["forecasts"], second["forecasts"])
            self.assertEqual(first["folds"], second["folds"])

            store = ForecastStore(database)
            self.assertEqual(store.count_runs(), 2)
            stored = store.fetch_run(first["run_id"])
            self.assertEqual(
                stored["detail"]["model"]["frozen"]["parameters"],
                first["model"]["frozen"]["parameters"],
            )

    def test_replay_is_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "ledger.sqlite3"
            first = run_forecast(FIXTURE, Path(directory) / "o1.json", database)
            second = run_forecast(FIXTURE, Path(directory) / "o2.json", database)
            self.assertEqual(first["run_id"], second["run_id"])
            self.assertEqual(ForecastStore(database).count_runs(), 1)


class LedgerSharingTests(unittest.TestCase):
    def test_legacy_entry_and_forecast_share_ledger(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "shared.sqlite3"
            legacy_first = service.run_fixture(DEMO, database)
            self.assertTrue(legacy_first)
            result = run_forecast(FIXTURE, Path(directory) / "o.json", database)
            legacy_second = service.run_fixture(DEMO, database)
            self.assertTrue(legacy_second)

            with sqlite3.connect(database) as connection:
                tables = {
                    row[0]
                    for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
                }
                # 老入口的表与记录仍在，且空间记录可读取
                self.assertIn("runs", tables)
                self.assertIn("events", tables)
                self.assertIn("forecast_runs", tables)
                self.assertIn("forecast_rows", tables)
                self.assertIn("space_versions", tables)
                self.assertGreaterEqual(connection.execute("SELECT COUNT(*) FROM runs").fetchone()[0], 1)
                payload = json.loads(connection.execute("SELECT payload FROM runs LIMIT 1").fetchone()[0])
                self.assertIn("case_id", payload)
                # 预测、场景与空间版本可通过账本关联
                joined = connection.execute(
                    """
                    SELECT COUNT(*) FROM forecast_rows fr
                    JOIN forecast_runs r ON fr.run_id = r.run_id
                    JOIN space_versions sv ON r.space_version_id = sv.space_version_id
                    """
                ).fetchone()[0]
                self.assertGreater(joined, 0)
                scenarios = {
                    row[0]
                    for row in connection.execute(
                        "SELECT DISTINCT scenario_id FROM forecast_rows WHERE run_id=?",
                        (result["run_id"],),
                    )
                }
                self.assertEqual(scenarios, {"base", "reduced-space"})


class ScenarioCapTests(unittest.TestCase):
    def test_cap_truncation_is_applied_and_explained(self):
        zones = [("za", 100.0), ("zb", 200.0)]
        payload = {
            "forecast_origin": (date(2026, 1, 1) + timedelta(days=7 * 25)).isoformat(),
            "panel": build_panel(zones, 30, occupancy=0.8),
            "scenarios": [
                {"scenario_id": "base", "area_multiplier": 1.0, "rent_multiplier": 1.0},
                {"scenario_id": "tight", "area_multiplier": 0.5, "rent_multiplier": 1.0},
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            result = run_forecast(
                write_input(directory, payload),
                Path(directory) / "o.json",
                Path(directory) / "d.sqlite3",
            )
        tight = next(s for s in result["scenario_results"] if s["scenario_id"] == "tight")
        base = next(s for s in result["scenario_results"] if s["scenario_id"] == "base")
        for row in tight["rows"]:
            self.assertLessEqual(row["served_area_days"], row["capacity_cap"] + 1e-6)
            self.assertAlmostEqual(
                row["served_area_days"] + row["truncated_area_days"],
                max(row["demand_area_days"], row["capacity_cap"])
                if row["demand_area_days"] > row["capacity_cap"]
                else row["demand_area_days"],
                places=2,
            )
        # 高占用 + 面积减半必然触发截断，且必须在总计与警告中说清楚
        self.assertGreater(tight["totals"]["truncated_area_days"], 0)
        self.assertLess(tight["totals"]["served_area_days"], base["totals"]["served_area_days"])
        self.assertTrue(any("截断" in w for w in result["warnings"]))
        self.assertTrue(any("截断" in d["detail"] for d in tight["drivers"]))
        # 截断后区间右尾被方案上限削平
        for row in tight["rows"]:
            self.assertLessEqual(row["served_interval"]["upper"], row["capacity_cap"] + 1e-6)

    def test_rent_multiplier_is_mechanical_not_causal(self):
        with tempfile.TemporaryDirectory() as directory:
            result = run_forecast(FIXTURE, Path(directory) / "o.json", Path(directory) / "d.sqlite3")
        base = next(s for s in result["scenario_results"] if s["scenario_id"] == "base")
        reduced = next(s for s in result["scenario_results"] if s["scenario_id"] == "reduced-space")
        # 本样例需求未触及 0.7 倍上限：可成交面积日不变，收入仅按 1.1 机械放大
        self.assertEqual(base["totals"]["served_area_days"], reduced["totals"]["served_area_days"])
        self.assertAlmostEqual(
            reduced["totals"]["revenue"] / base["totals"]["revenue"], 1.1, places=2
        )
        self.assertTrue(any("无干预数据" in w for w in result["warnings"]))


class RefusalTests(unittest.TestCase):
    def test_zone_without_history_is_refused_not_faked(self):
        zones = [("za", 100.0), ("zb", 150.0)]
        payload = {
            "forecast_origin": (date(2026, 1, 1) + timedelta(days=7 * 25)).isoformat(),
            "panel": build_panel(zones, 30, occupancy=0.6, closed_zones=("zb",)),
            "scenarios": [{"scenario_id": "base", "area_multiplier": 1.0, "rent_multiplier": 1.0}],
        }
        with tempfile.TemporaryDirectory() as directory:
            result = run_forecast(
                write_input(directory, payload),
                Path(directory) / "o.json",
                Path(directory) / "d.sqlite3",
            )
        refused = [row for row in result["forecasts"] if row["zone_id"] == "zb"]
        self.assertEqual(len(refused), 4)
        for row in refused:
            self.assertEqual(row["support_status"], "refused_no_history")
            self.assertIsNone(row["demand_area_days"])
            self.assertIsNone(row["revenue"])
            self.assertIsNone(row["interval"])
        self.assertTrue(any("拒绝预测" in w for w in result["warnings"]))
        # 被拒绝的行不进入总计
        totals = result["scenario_results"][0]["totals"]
        self.assertEqual(totals.get("refused_rows"), 4)


if __name__ == "__main__":
    unittest.main()
