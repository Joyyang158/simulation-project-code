# Airport gate queue simulation

## 当前正式实验：恢复原始需求，每航站楼 4 个 gate

主报告保留两个 part：校准合成输入与 BTS 原始需求案例。两版均为 **T4/DL、T5/B6、T8/AA，每航站楼 4 个 gate，其中 2 个支持跨航站楼共享，共 12 个 gate**。均使用 seed=553、100 次重复、1440 分钟观察窗口并在日末排空。真实样本仍为 **3,877 条、31 天，平均 125.06 航班/日**；主实验 `demand_multiplier=1`，没有复制扩增航班。

合成版保持抽中日期各航司的实际航班数，使用航司半小时到达直方图和匹配经验均值/总体方差的 Gamma 服务。BTS baseline 保留实际请求时间并按航司重抽服务代理值；observed-day 模式重放原日期的服务值。两版容量、航班数、航司映射和服务分布矩对齐，不要求到达依赖结构、服务尾部及有限样本值完全一样。单 gate setting 始终为 1 个 gate，均值 100.6724 分钟和相同 Poisson 负荷。

```bash
python scripts/run_main.py
python scripts/run_scaling.py
python run.py validate --replications 100 --output results/validation_100
python run.py diagnose --data data/processed/jfk_2025_07.csv --output results/inputs_current
python scripts/build_current_report.py
python -m unittest discover -s tests
```

报告生成脚本读取相邻 `report/` 仓库保留的版式模板，更新两个 part，并在合并版最后追加 scaling 实验。PDF 编译（在 `report/` 仓库运行）：

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build/report report.tex
cp build/report/report.pdf report.pdf
```

## 最后的合成数据 scaling 测试

- 航站楼数：**1、2、3、4、5**。
- 每航站楼 gate 数：**1、5、10、20、50**，完整 **25 种配置**。
- 每种配置 100 次重复，对比无共享、零转移共享、20 分钟转移共享，共 7,500 次政策运行。
- 每航站楼每天独立 Poisson 航班数，期望固定为 **41.69**；使用 BTS 汇总的半小时到达曲线、均值 100.67 分钟且匹配总体方差的 Gamma 服务。这里全部航班为合成数据，航站楼同分布，不使用实际航司映射。
- 增加航站楼时总需求随之增加；增加 gates 时需求不变。相同重复下各容量/政策使用相同输入，增加航站楼保留已有航站楼的航班。
- 可共享 gates 为 `ceil(c/2)`，对应 1、3、5、10、25。记录等待、利用率、完成量和模拟器运行耗时；耗时是本机描述性计时，不是受控硬件基准。

结果分别位于 `results/main/`、`results/scaling/`；报告副本在 `report/results_main/`、`report/results_scaling/`。当前清洗数据、时间戳、来源审计及诊断图位于 `report/results_inputs/`。主实验逐日期检查未扩增的航班数，scaling 检查完整网格和同一配置输入一致。

之前的 20-gate 原始需求及 5 倍压力实验已退出当前报告，配置、结果和生成脚本仅保留作历史复现。`build_stress_report.py` 属于历史脚本，不是当前报告入口。旧无 `--data` 演示也不属于当前对齐的主实验。

以下为原演示及数据清洗说明。

课程项目：从单机位 FCFS 到多机位 pooling，再到受兼容性限制的跨航站楼共享。全部时间单位是 **分钟**。实现采用 Python 离散事件仿真，核心分配规则可在 `src/airport_sim/simulator.py` 中直接检查。

**默认不带 `--data` 的运行仍为合成演示。另已完成 BTS 真实数据驱动版本：2025 年 7 月 JFK 的 AA、B6、DL 子系统，3,877 次筛选后周转、31 天。** 真实版使用航班到达记录和周转代理；机位数量、兼容性和基础滑行仍是假设，模拟等待不等于实际 JFK 等待。真实版运行方式见文末。

## 快速运行

需要 Python 3.10+、NumPy、SciPy、Matplotlib。正常联网环境可安装：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python run.py run --config configs/demo.json --replications 100
python run.py validate --replications 100
python -m unittest discover -s tests -v
```

本机已有 `py310daily` conda 环境，验证时使用其中的依赖，无需联网安装：

```bash
conda activate py310daily
python run.py run --config configs/demo.json --replications 100
python run.py validate --replications 100
python -m unittest discover -s tests -v
```

以上命令在本目录执行。也可在其他目录使用 `python /绝对路径/run.py ...`；此时输出和参数路径相对于工作目录。`--help` 可查看各子命令。

## 和 notes 的对应

| 章节 | 代码中的实现 | 检查方式 |
| --- | --- | --- |
| Ch. 2 概率统计 | Student-t 置信区间 | `metrics.py`；已知数值案例测试 |
| Ch. 3 随机数生成 | `SeedSequence([seed, replication])` 拆分到达、服务、航司流 | `random_inputs.py`；同 seed 可复现 |
| Ch. 4 随机变量生成 | 指数逆变换 `-mean*log(1-U)`；Erlang 为多个指数之和；高峰到达使用混合成分抽样 | `random_inputs.py`、`synthetic_day` |
| Ch. 5 输入分布 | 按日期拆分训练/留出数据、MLE、AIC、Q-Q 图、KS 统计量、日内 lag-1 相关性、经验重采样 | `inputs.py`、`diagnose` 命令 |
| Ch. 6 输出分析 | 稳态 replication/deletion；单日 terminating simulation；独立重复层面的 t 区间 | `validate` 与 `run` 分开 |
| Ch. 7 方差缩减 | CRN：相同飞机绑定相同服务输入；配对差值 CI；与独立输入的差值方差对照 | `paired_differences.csv`、`metadata.json` |

生产随机流使用 NumPy 的生成器，不用教学 LCG 代替可靠生成器。课程中的逆变换与卷积算法显式实现。没有为凑技术数量加入重要抽样等不适合本项目核心问题的方法。

## 模型约定

- `Flight` 包含 ID、申请机位时刻、服务时长、所属航站楼、航司；`Gate` 包含所属航站楼、兼容航司、关闭区间、初始占用结束时刻。
- 事件包括到达、释放、跨楼服务开始、关闭/重新开放、观察窗口结束。相同时刻先处理完事件，再统一分配机位；飞机按到达时刻、ID 排序，机位按 ID 打破平局。
- **Pooled：** 每个航站楼统一排队，使用本楼空闲兼容机位。若队首和某机位不兼容，允许后面的兼容飞机使用该机位。
- **Dedicated：** 按到达顺序轮流固定分配一个兼容本楼机位，进入后不换队。它代表明确的一种独立队列策略，并不代表所有独立队列策略。
- **Sharing：** 先找本楼空闲兼容机位，没有时尝试外楼空闲兼容机位；目标在转移期间立即预留。策略不读取未来服务时长。远楼服务等待为纯排队加转移时间。
- **机位关闭：** 非抢占，正在服务的飞机完成后才释放；关闭期间不接新飞机。不允许新的跨楼转移预留跨过已知关闭起点。关闭结束自动重新派发等待飞机。
- **初始占用：** 通过 `initial_busy_until` 设置初始剩余占用。实验另比较所有机位空闲与均占用半个平均服务时长的启动方式；这只是敏感性假设。
- **观察窗口：** 只接收 `[0, horizon)` 的飞机，之后继续处理到全部完成；日内完成数、利用率和队长只计算固定窗口，飞机等待时间则追踪到真正入位。
- 利用率分母为总机位数乘观察窗口，关闭区间仍包含在分母中；`reserved_utilization` 额外包含转移预留。两者都包含初始占用。
- 纯排队队长不包含转移中的飞机。`mean_queue` 是时间积分除观察窗口，并非事件时刻队长的简单平均。

## 三组实验

`configs/demo.json` 控制参数；也可用 `--config 自己的配置.json` 覆盖其中一部分，未知字段会报错。

1. **单机位：** 负荷 0.50、0.70、0.85、0.95，Poisson 到达。无真实数据时服务为均值 45 分钟的指数分布；提供数据时对服务样本做经验抽样。比较原服务与 `S_low = mean + 0.5*(S-mean)`：总体均值不变，方差降到四分之一。同一重复使用相同到达和基础服务抽样。这组是空系统启动的有限日实验，不能直接套稳态公式解释结果。
2. **单航站楼：** 选择配置中按名称排序的第一个航站楼，比较 dedicated、pooled、pooled 加高峰平滑。平滑将到达时刻向等间隔时刻移动，保留飞机、顺序、服务输入和日总量；额外输出平均绝对时刻调整量。
3. **多航站楼：** 比较不共享与 0、10、20、40 分钟转移时间的共享；另比较首个机位关闭、初始占用敏感性。默认两个抽象航站楼各四个机位，每楼两个机位兼容全部航司，另外两个只服务本楼航司。所有策略比较保持需求与容量相同。

无真实数据时，每天 200 架次，65% 归 T1，35% 归 T2。每楼 65% 的到达集中于各自高峰，其他在日内较宽窗口均匀分布。多机位服务为 Erlang(4)，平均 45 分钟。该设置故意形成拥堵以展示策略差异。

默认每个条件 100 次重复，`--seed` 默认 553。合成主实验、BTS 主实验、observed-day 敏感性、scaling 和 M/M/1 验证均使用 100 次重复。

## M/M/1 理论验证

```bash
python run.py validate --replications 100 --warmup 5000 --duration 20000
```

服务均值设为 1 分钟，`rho = lambda/mu`。每次先模拟 5000 分钟预热，只统计之后 20000 分钟内到达飞机的等待；继续运行直到这些飞机完成。比较公式：

```text
E[Wq] = lambda / (mu * (mu-lambda)) = rho / (1-rho), when mu = 1.
```

输出是各负荷独立重复的均值和逐点 95% CI。理论偶尔落在区间之外是正常抽样现象；应结合误差大小、运行长度、预热敏感性解释，不能把单次覆盖当作数学正确性的证明。确定性测试还将单机位结果与独立的 Lindley 递推逐飞机比较。

## BTS 数据接入

目前没有自动下载器。下载的文件应限定为国内航班，并尽量保留飞机的全部可观察航段和样本前后相邻日期；只下载到达 JFK 的航段无法匹配后续出发。

读取以下 BTS 字段，支持常见两种列名：

| 标准名 | 下载文件别名 |
| --- | --- |
| FlightDate | FL_DATE |
| Reporting_Airline | OP_UNIQUE_CARRIER |
| Tail_Number | TAIL_NUM |
| Origin / Dest | ORIGIN / DEST |
| CRSDepTime | CRS_DEP_TIME |
| DepDelay | DEP_DELAY |
| ActualElapsedTime | ACTUAL_ELAPSED_TIME |
| TaxiIn | TAXI_IN |
| Cancelled / Diverted | CANCELLED / DIVERTED |

准备两个映射 CSV。以下只演示文件格式，航司归属应按研究时期核实或明确声明为抽象配置：

```csv
airport,timezone
JFK,America/New_York
LAX,America/Los_Angeles
```

```csv
airline,terminal
AA,T1
DL,T2
```

时区表应覆盖输入的国内起降机场。处理命令：

```bash
python run.py clean --raw data/raw/flights.csv \
  --timezones data/timezones.csv --terminals data/terminals.csv \
  --airport JFK --baseline-taxi 10 --clearance 0 --max-stay 360 \
  --output data/processed/visits.csv

python run.py diagnose --data data/processed/visits.csv
python run.py run --data data/processed/visits.csv --replications 100 --output results/bts
```

清洗逻辑：

1. 当地计划出发时刻加出发延误，在 UTC 下加实际 elapsed time 得到实际进位时间，避免跨时区与午夜的 HHMM 相减错误。
2. 同一机尾仅匹配连续观察到的到达与下一次出发；去重，剔除取消、备降、缺机尾、非正/过长停场及过夜停场。若某机尾存在无法解析的航段，保守排除整个机尾，避免跨过已知缺失航段配对。该规则可能损失较多样本，应检查审计计数。
3. 申请机位时刻为推导进位时间减 TaxiIn，再加假设的基础滑行时间。TaxiIn 并非纯排队标签。
4. 服务时间为同一飞机下一次在 JFK 的实际离开时间减去本次实际到达时间，不加清场缓冲（`clearance=0`）。清洗后保留两端 UTC 时间戳，逐条验证差值。未观察到的航段及拖移仍可能让停场代理失真。
5. 不处理时长为 23/25 小时的夏令时转换日，不猜测含糊的出发时刻。所有剔除写入 `visits.audit.json`。

生成 CSV 的必需列为 `date,id,arrival,service,terminal,airline`，也可由用户自己的清洗流程生成。`arrival` 为当日零点后的分钟数，`service` 是匹配的 gate 离开时间减到达时间，不含额外缓冲；`actual_gate_arrival_utc` 和 `next_gate_departure_utc` 用于核对差值。`run --data` 按整天重采样到达，并按航司经验重采样服务；这些是**成功匹配的子系统航班**，不能称为机场全部日需求。

`diagnose` 按日期将前约 80% 作为拟合数据，其余留出；输出指数、Gamma、Lognormal 的拟合参数、AIC、训练和留出 KS 统计量。估计了参数后不能直接套用标准 KS p 值，因此不输出误导性 p 值。拟合为诊断用途，`run` 默认仍使用经验服务重采样，不会偷偷按 AIC 切换模型；需要正式留出实验时，应将训练/留出 CSV 分开并明确实验设计。存在服务相关性时，独立重采样是假设，需要进一步分组或块重采样敏感性研究。

## 输出与解释

| 文件 | 内容 |
| --- | --- |
| `replications.csv` | 每次重复每个场景的原始指标与抽样日期 |
| `summary.csv` | 每个指标的重复均值、标准差、95% t 区间 |
| `paired_differences.csv` | 同次重复下 **alternative − baseline** 的区间；等待差值负数表示改善 |
| `metadata.json` | 参数、seed、Python/依赖版本、输入来源及校验和、兼容规则、CRN 对照 |
| `flight_records/*.csv` | 第一次重复的逐飞机轨迹，包含排队、转移、开始与完成时刻 |
| `event_trace.json` | 第一次重复、不共享基线的事件审计轨迹 |
| `experiment_overview.png/.pdf` | 三组实验概览 |
| `policy_differences.png/.pdf` | 配对政策差异及区间 |
| `mm1_validation.csv/.png/.pdf` | 独立 `validate` 命令生成的理论验证 |

指标包括平均和 95% 分位等待、`P(W>15)`、平均纯排队、平均转移、共享比例、时间平均和最大队长、利用率、日内完成数量、排空后完成数量。`p95_wait` 的 CI 针对**每天 p95 的期望**，不是把相关航班混在一起后的总体分位数 CI。其他区间也均以独立重复为单位，不把一天内的飞机错误当作独立样本。

输出目录可以重跑覆盖同名文件；改变场景配置时建议使用新的 `--output` 目录，避免遗留旧场景的轨迹文件。默认生成结果和原始数据被 `.gitignore` 排除，源代码与配置可单独提交。

## 目录

```text
configs/demo.json
src/airport_sim/
  simulator.py       # 状态、事件、兼容性和三种分配策略
  random_inputs.py   # 随机数流、逆变换、卷积
  inputs.py          # 合成数据、BTS 清洗、输入拟合
  experiments.py    # 三组实验、CRN 对照、M/M/1
  metrics.py         # Student-t 区间与配对差值
  plots.py           # 报告用 PNG/PDF
  cli.py             # 命令行入口
tests/               # 手算、Lindley、约束、时间解析、输入到结果集成测试
run.py
```

当前交付包括合成演示与下述 BTS 数据驱动版本。真实 gate 级容量、历史兼容关系与连续占位仍未取得；天气、完整滑行道网络和真实 gate 编号未纳入模型。

## 已完成的 BTS 版本（2025 年 7 月 JFK）

官方六月、七月、八月 ZIP 已下载到 `data/raw/bts/`。研究七月，保留六月 30 日与八月 1 日作为边界上下文；AA、B6、DL 的全部可观察国内航段用于确认机尾的连续访问。原始数据来源：
https://www.transtats.bts.gov/PREZIP/。

复现清洗与输入诊断：

```bash
conda activate py310daily
python scripts/prepare_bts.py
python run.py diagnose --data data/processed/jfk_2025_07.csv --output results/bts_inputs
```

脚本要求原始 ZIP 和 `data/reference/openflights_airports.dat` 已存在。
重新下载时对应文件为 `On_Time_Reporting_Carrier_On_Time_Performance_1987_present_2025_6.zip`、`..._2025_7.zip`、`..._2025_8.zip`，本地分别保存为 `2025_6.zip`、`2025_7.zip`、`2025_8.zip`。
下载 URL、ZIP 校验、SHA-256、筛选计数、时钟一致性和来源记录见 `data/processed/bts_provenance.json`；时区数据来源和许可见 `data/reference/ATTRIBUTION.md`。

主要样本统计：

- 上下文日期的三航司原始航段 215,840 条；取消/备降 7,166 条。
- 完成并解析 208,674 条，208,673 条进位时钟完全一致，1 条差一分钟。
- 候选 JFK 到达 7,211 次；剔除过夜 1,995 次、无法连续匹配 434 次、非正或超过六小时停场 666 次。
- 上下文保留 4,116 次；按机位请求日期限定七月后为 **3,877 次**：AA 690、B6 1,744、DL 1,443。
- 日样本量 72–147，服务代理均值 100.67 分钟（实际离开减到达，无额外清空缓冲）。

运行正式的 100 次重复实验：

```bash
python run.py run --data data/processed/jfk_2025_07.csv \
  --config configs/bts_jfk_2025_07.json --replications 100 \
  --output results/bts_jfk_2025_07

python run.py run --data data/processed/jfk_2025_07.csv \
  --config configs/bts_jfk_2025_07_observed.json --replications 100 \
  --output results/bts_jfk_2025_07_observed
```

两个配置的区别：`service_mode=resample_by_airline` 在航司内独立经验抽样；
`service_mode=observed_day` 保留所选观测日的飞机服务代理，检查日内相关性的影响。
两者均整天重采样到达、使用同一套 12 个假设机位（T4、T5、T8 各 4 个）及 CRN 配对。
单机位部分仍使用合成 Poisson 到达和汇总经验服务，不是机场实际单机位需求回放。

真实输入模式下，`flights_per_day` 和 `mean_service` 两个合成数据参数不参与生成多机位输入；
需求与服务来自 `--data`，单机位服务均值也由输入数据计算。四个机位/航站楼只是实验容量，
不是对真实机场机位数的估计；当前航司归属用于建模，未取得历史 gate 级数据。

输入诊断按 24 个训练日期和 7 个留出日期比较分布，参数拟合仅作诊断。
两个模拟版本均使用全部 31 天的经验输入，不能将其称为留出预测验证。
100 次重复是以 31 个观测日为条件的 Monte Carlo 重复，并不是增加到 100 个观测日。

报告位于相邻 `report/report_bts.tex` 与 `report/report_bts.pdf`，包括页码、五张图、筛选审计、
拟合诊断、结果表及相关性敏感性分析。主文件 `report/report.tex`、`report/report.pdf` 合并两部分：Part I 为合成实验，Part II 为 BTS 实验，页码连续。独立合成版保存在 `report/report_synthetic.tex/pdf`，独立 BTS 版保存在 `report/report_bts.tex/pdf`。
真实版结果表明，在这些假设下网络平均等待从不共享的 105.60 分钟降至零转移共享的 47.48 分钟；
这是模拟策略对比，不能解释为实际 JFK 的观测等待。
