市场空间经营服务。现有空间关系、占用时间片、路径与情景处理模块记录本地经营方案及建议。

环境：Linux，Python 3.11.2 或更新的兼容版本。所有依赖来自 Python 标准库，无需安装第三方包。

测试：python3 -m unittest discover -s tests/space_scenarios -p 'test_*.py'
构建：python3 -m compileall -q market_space_twin tests/space_scenarios
验收：python3 -m market_space_twin.simulator.cli fixtures/demo.json --database /tmp/market_space_twin-acceptance.sqlite3
所有输入均为本地合成数据，运行不需要网络。

目录：market_space_twin 保存业务处理与存储模块；tests/space_scenarios 保存业务回归；fixtures/demo.json 为原有业务样例，fixtures/analysis 为经营分析资料及字段说明。

原有验收命令每次使用新的 SQLite 路径，避免复用上次状态。新增输入完整性检查：
python3 -m unittest discover -s tests/analysis_inputs -v

招商需求预测（可回测、可审计，不含自动定价或自动签约）：
python3 -m market_space_twin.rental_forecast --input fixtures/analysis/input.json --output /tmp/forecast.json --database /tmp/market_space_twin-forecast.sqlite3
输出含 folds（真实起报时间回测）、model（冻结参数与训练截止）、forecasts、scenario_results、warnings；
预测、场景与空间版本写入同一账本（forecast_runs/forecast_rows/forecast_folds/space_versions），
只增不改，修改未来实际成交不会改写过去那次拟合，原有入口的空间记录仍可读取。

容器运行可使用已经预装的 ccusr/prepared:3f007a95fcee-4441968e5ff6 镜像；将当前仓库只读挂载至 /source，在容器内复制到 /tmp/product 后运行上述命令。依赖准备和正式运行分离，运行过程中不需要网络访问。
