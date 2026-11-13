"""预测账本：与既有 SQLiteLedger 共库，只增不改。

- 复用 twin_store.SQLiteLedger 的连接与既有表（runs/events/... 不受影响）。
- 新增 forecast_runs / forecast_rows / forecast_folds / space_versions 四张表，
  全部 INSERT OR IGNORE，绝不 UPDATE/DELETE：一次拟合落账后不可被后续
  实际成交的修改所改写；输入变化只会产生新的 run_id 新行。
- 关联链：forecast_rows.run_id → forecast_runs.run_id → space_versions.space_version_id，
  scenario_id 落在 forecast_rows 上，场景定义落在 forecast_runs.detail 里。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from market_space_twin.simulation.twin_store import SQLiteLedger, stable_json, utc_now

SCHEMA = """
CREATE TABLE IF NOT EXISTS space_versions(
    space_version_id TEXT PRIMARY KEY,
    as_of TEXT NOT NULL,
    definition TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS forecast_runs(
    run_id TEXT PRIMARY KEY,
    forecast_origin TEXT NOT NULL,
    input_fingerprint TEXT NOT NULL,
    space_version_id TEXT NOT NULL,
    detail TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS forecast_runs_origin ON forecast_runs(forecast_origin);
CREATE TABLE IF NOT EXISTS forecast_rows(
    run_id TEXT NOT NULL,
    scenario_id TEXT NOT NULL,
    zone_id TEXT NOT NULL,
    week TEXT NOT NULL,
    payload TEXT NOT NULL,
    PRIMARY KEY(run_id, scenario_id, zone_id, week)
);
CREATE TABLE IF NOT EXISTS forecast_folds(
    run_id TEXT NOT NULL,
    fold_id TEXT NOT NULL,
    payload TEXT NOT NULL,
    PRIMARY KEY(run_id, fold_id)
);
"""


class ForecastStore:
    def __init__(self, database: str | Path):
        self.ledger = SQLiteLedger(database)
        with self.ledger.connect() as connection:
            connection.executescript(SCHEMA)

    def record_run(self, result: dict[str, Any]) -> None:
        """把一次完整预测落账。所有写入均为 INSERT OR IGNORE，保证幂等与不可改写。"""
        now = utc_now()
        version = result["space_version"]
        with self.ledger.connect() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO space_versions(space_version_id,as_of,definition,created_at) VALUES(?,?,?,?)",
                (result["space_version_id"], result["forecast_origin"], stable_json(version), now),
            )
            detail = {
                "model": result["model"],
                "horizon_weeks": result["horizon_weeks"],
                "scenarios": result["scenario_definitions"],
                "drivers": result["drivers"],
                "warnings": result["warnings"],
                "notes": result["notes"],
                "generated_at": result["generated_at"],
                "code_version": result["code_version"],
            }
            connection.execute(
                "INSERT OR IGNORE INTO forecast_runs(run_id,forecast_origin,input_fingerprint,space_version_id,detail,status,created_at) VALUES(?,?,?,?,?,?,?)",
                (
                    result["run_id"],
                    result["forecast_origin"],
                    result["input_fingerprint"],
                    result["space_version_id"],
                    stable_json(detail),
                    "completed",
                    now,
                ),
            )
            for row in result["forecasts"]:
                connection.execute(
                    "INSERT OR IGNORE INTO forecast_rows(run_id,scenario_id,zone_id,week,payload) VALUES(?,?,?,?,?)",
                    (result["run_id"], row["scenario_id"], row["zone_id"], row["week"], stable_json(row)),
                )
            for scenario in result["scenario_results"]:
                for row in scenario["rows"]:
                    connection.execute(
                        "INSERT OR IGNORE INTO forecast_rows(run_id,scenario_id,zone_id,week,payload) VALUES(?,?,?,?,?)",
                        (result["run_id"], scenario["scenario_id"], row["zone_id"], row["week"], stable_json(row)),
                    )
            for fold in result["folds"]:
                connection.execute(
                    "INSERT OR IGNORE INTO forecast_folds(run_id,fold_id,payload) VALUES(?,?,?)",
                    (result["run_id"], fold["fold_id"], stable_json(fold)),
                )

    def fetch_run(self, run_id: str) -> dict[str, Any] | None:
        with self.ledger.connect() as connection:
            row = connection.execute("SELECT * FROM forecast_runs WHERE run_id=?", (run_id,)).fetchone()
            if row is None:
                return None
            rows = connection.execute(
                "SELECT scenario_id,zone_id,week,payload FROM forecast_rows WHERE run_id=? ORDER BY scenario_id,zone_id,week",
                (run_id,),
            ).fetchall()
        return {
            "run_id": row["run_id"],
            "forecast_origin": row["forecast_origin"],
            "input_fingerprint": row["input_fingerprint"],
            "space_version_id": row["space_version_id"],
            "status": row["status"],
            "detail": json.loads(row["detail"]),
            "rows": [json.loads(item["payload"]) for item in rows],
        }

    def count_runs(self) -> int:
        with self.ledger.connect() as connection:
            return int(connection.execute("SELECT COUNT(*) FROM forecast_runs").fetchone()[0])

    def table_names(self) -> list[str]:
        with self.ledger.connect() as connection:
            return [
                row[0]
                for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name").fetchall()
            ]
