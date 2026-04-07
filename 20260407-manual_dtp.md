# 2026-04-07 Data Preprocessor Manual

## 1. 从输入到输出的数据处理结构

当前主实现文件：
- `/Users/shangbo/personal/mytrae/ljcomp/code/data_preprocessor/build_daily_rank_packets.py`

兼容入口：
- `/Users/shangbo/personal/mytrae/ljcomp/code/data_preprocessor/process_data.py`

旧版备份：
- `/Users/shangbo/personal/mytrae/ljcomp/code/data_preprocessor/20260403-process_data_deprecated.py`

### 1.1 输入

输入 parquet：
- `train.parquet`
- `test.parquet`

原始主键：
- `stockid`
- `dateid`
- `timeid`

原始特征：
- `f0 ~ f383`

训练标签：
- `LabelA`
- `LabelB`
- `LabelC`

### 1.2 当前输出

当前 daily parquet 只保留：
- 主键：`stockid, dateid, timeid`
- 排名特征：`f0_xsec_rank ~ f383_xsec_rank`
- 占位特征：`f0_ts_rank ~ f383_ts_rank`
- 标签：`LabelA, LabelB, LabelC`（如果输入里存在）

当前版本不再保留原始 `f0 ~ f383`。

### 1.3 函数遍历

#### `willden/file_method/file_management.py`

- `quick_read(file_path, stockid_range, dateid_range, timeid_range)`
  - 作用：按连续范围读取 parquet，再用列过滤精确裁剪。
  - 当前数据预处理主要依赖这个函数读单日数据。

- `read_byclass(file_path, by, batch_size=1000)`
  - 作用：按主键过滤读取。
  - 当前 xsec-only 主路径未使用。

- `read_batch(file_path, batch_size=1000)`
  - 作用：按批读取 parquet/csv。
  - 当前 xsec-only 主路径未使用。

#### `willden/data_method/data_processing.py`

- `rank(df, columns, by, weights=None)`
  - 作用：对给定列按分组计算归一化 rank。
  - 当前 xsec-only 主路径用于按 `(dateid, timeid)` 计算横截面 rank。

- `standardize(...)`
  - 作用：z-score 标准化。
  - 当前数据预处理主路径未使用。

- `winsorize(...)`
  - 作用：分位数截尾。
  - 当前数据预处理主路径未使用。

- `promote_historical_ft(...)`
  - 作用：旧历史窗口特征生成尝试。
  - 当前 xsec-only 主路径未使用。

#### `ljcomp/code/data_preprocessor/build_daily_rank_packets.py`

- `load_single_date(file_path, dateid, stockid_range)`
  - 作用：通过 `quick_read` 读取单个 `dateid` 的全量数据。

- `build_daily_packet_xsec_only(file_path, dateid, output_dir, stockid_range)`
  - 作用：
    1. 读取单日数据
    2. 用 `rank(..., by=['dateid', 'timeid'])` 生成全部 `xsec_rank`
    3. 构造 `ts_rank` 的空列占位
    4. 保留主键 + rank + labels
    5. 写出 `d{dateid}.parquet`

- `_process_single_date_worker(args)`
  - 作用：并行 worker 包装器。

- `process_date_range_serial(...)`
  - 作用：串行遍历多个 `dateid`。

- `process_date_range_parallel(...)`
  - 作用：按 `dateid` 并行生成 daily parquet。

- `main()`
  - 作用：解析 CLI 参数并执行串行或并行模式。

### 1.4 当前主流程

1. 遍历 `dateid`
2. 调 `load_single_date(...)` 读取该天全部 500 * 239 行原始数据
3. 调 `rank(date_df, by=['dateid', 'timeid'])` 生成 384 维 `xsec_rank`
4. 构造 384 维 `ts_rank` 空列，保持原 schema 兼容
5. 仅保留 `主键 + ranks + labels`
6. 输出 `d{dateid}.parquet`

## 2. 不同分支的尝试

### 2.1 `xsec_rank + ts_rank`

旧实现尝试：
- 对每个 `(dateid, timeid)` 回看最多 `2 * 239 = 478` 个时点
- 对历史窗口按 `stockid` 计算 `ts_rank`
- 对当前时点按 `(dateid, timeid)` 计算 `xsec_rank`
- 最终只输出当前时点 500 行，再拼成 daily parquet

问题：
- 单个 `dateid` 里有 239 个 `timeid`
- 每个 `timeid` 都会重复构造窗口并重算一次 `ts_rank`
- 全量数据下速度过慢，不适合直接批量生产 360 天数据

### 2.2 `xsec_rank_only`

当前实现：
- 只读取单日数据
- 只算 `xsec_rank`
- `ts_rank` 保留原 schema，但全部先填 `NaN`

优点：
- 计算复杂度显著下降
- 只依赖单日横截面
- 更适合先把 daily parquet 数据资产生产出来

代价：
- 暂时没有可用的 `ts_rank`
- 下游如果真的依赖 `ts_rank` 数值，需要后续补生产或改训练逻辑

## 3. 接下来可以做什么改进

### 3.1 `ts_rank` 方向

- 需要重新设计 `ts_rank` 算子，当前按窗口整段重算的方式不适配 360 天全量生产。
- 优先考虑滑动窗口增量更新，而不是每个 `timeid` 都从头 `rank(window_df, by=['stockid'])`。
- 如果必须保留 rank 语义，可以考虑只维护每只股票近 478 个点的有序状态。

### 3.2 `file_management` 方向

- `quick_read` 目前仍然是“按 row group 读取 + 再过滤”。
- 可以进一步利用 parquet row group 与 `stockid/dateid` 的顺序关系，减少无效 row group 读取。
- 可以考虑增加“只读取指定列”的接口，避免标签或特征不需要时仍然全列搬运。

### 3.3 数据输出方向

- 当前 daily parquet 已经移除了原始 `f0 ~ f383`，显著降低了文件体积。
- 后续可以再检查是否需要保留全部 768 维 rank；如果某些训练分支只需要 `xsec_rank`，还可以继续拆分更轻的版本。

### 3.4 并行策略方向

- 当前策略是按 `dateid` 并行。
- 在 16G 内存条件下，`num_workers=2` 是较稳妥的默认值。
- 如果未来算子进一步轻量化，可以重新评估是否提升到 `3`。

### 3.5 日志与可观测性

- 当前进度条以 `dateid` 为最小单位。
- 后续可以增加：
  - 每日输出文件大小
  - 每日行数/列数检查
  - 平均每个 `dateid` 耗时
  - 异常 dateid 重试日志
