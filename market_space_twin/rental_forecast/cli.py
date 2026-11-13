"""命令行入口：python3 -m market_space_twin.rental_forecast --input <周面板> --output <预测> --database <账本>"""
from __future__ import annotations

import argparse
import json

from market_space_twin.rental_forecast.pipeline import run_forecast


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="market_space_twin.rental_forecast",
        description="可验证的招商需求预测：真实起报时间回测、冻结模型、空间版本与情景比较",
    )
    parser.add_argument("--input", required=True, help="周面板 JSON 文件")
    parser.add_argument("--output", required=True, help="预测结果 JSON 输出路径")
    parser.add_argument("--database", required=True, help="经营账本 SQLite 路径")
    args = parser.parse_args(argv)
    result = run_forecast(args.input, args.output, args.database)
    summary = {
        "run_id": result["run_id"],
        "forecast_origin": result["forecast_origin"],
        "horizon_weeks": result["horizon_weeks"],
        "space_version_id": result["space_version_id"],
        "selected_model": result["model"]["selected"],
        "forecast_rows": len(result["forecasts"]),
        "scenarios": [s["scenario_id"] for s in result["scenario_results"]],
        "warnings": len(result["warnings"]),
        "output": args.output,
        "database": args.database,
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
