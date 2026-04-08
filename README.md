# data_preprocessor

Data preprocessing pipeline for generating daily parquet packets used by the `ljcomp` training workflow.

## What It Does

- Reads raw parquet data from `train.parquet` or `test.parquet`
- Optionally shards the large raw parquet into `d{dateid}.parquet` raw daily files with one sequential scan
- Builds EWMA output files `d{dateid}.parquet`
- Supports the pipeline `raw fetch/shard -> causal ewma impute -> calculate ewma(raw) -> xsec rank -> pack output`
- Produces xsec-rank features for present and multiple EWMA half-life horizons

## Current Output Schema

Each daily parquet keeps:

- keys: `stockid`, `dateid`, `timeid`
- labels: `LabelA`, `LabelB`, `LabelC` when present
- present `f0_xsec_rank ~ f383_xsec_rank`
- `f0_xsec_rank_ewma_5hl ~ f383_xsec_rank_ewma_5hl`
- `f0_xsec_rank_ewma_30hl ~ f383_xsec_rank_ewma_30hl`
- `f0_xsec_rank_ewma_60hl ~ f383_xsec_rank_ewma_60hl`
- `f0_xsec_rank_ewma_1dhl ~ f383_xsec_rank_ewma_1dhl`

Notes:

- raw feature columns `f0 ~ f383` are intentionally dropped from daily outputs
- `ts_rank` is not part of this production path

## Main Files

- `build_raw_daily_shards.py`: one-pass raw parquet -> raw daily shard builder
- `build_daily_rank_packets.py`: legacy xsec-rank packet builder
- `data_ewma_preprocessor.py`: build EWMA rank packets from raw parquet
- `process_data.py`: compatibility entrypoint
- `20260403-process_data_deprecated.py`: deprecated historical implementation backup
- `20260407-manual_dtp.md`: implementation manual and branch notes

## Typical Usage

Generate 30 days in parallel:

```bash
python build_daily_rank_packets.py \
  --file_path /path/to/train.parquet \
  --output_dir /path/to/daily_data \
  --start_dateid 0 \
  --end_dateid 30 \
  --mode parallel \
  --num_workers 2
```

## Current Parallel Strategy

- parallelism is applied across `dateid`
- `num_workers=2` is the current safe default on a 16 GB machine

## Next Improvements

- implement a truly efficient `ts_rank` operator
- optimize file reading paths in `willden/file_method/file_management.py`
- add stronger validation and production logging

## Recommended Usage

Step 1. Build raw daily shards once for the target date range:

```bash
python build_raw_daily_shards.py \
  --file_path /Users/shangbo/personal/mytrae/ljcomp/train.parquet \
  --output_dir /Users/shangbo/personal/mytrae/ljcomp/daily_data_raw_v2 \
  --start_dateid 12 \
  --end_dateid 30
```

Step 2. Build EWMA packets from the raw daily shards:

```bash
python data_ewma_preprocessor.py \
  --raw_daily_dir /Users/shangbo/personal/mytrae/ljcomp/daily_data_raw_v2 \
  --output_dir /Users/shangbo/personal/mytrae/ljcomp/daily_data_ewma_v2 \
  --start_dateid 12 \
  --end_dateid 30 \
  --mode parallel \
  --num_workers 2
```

Notes:

- do not mix the old `daily_data_ewma` directory with the new `*_5hl/_30hl/_60hl/_1dhl` schema
- `ewma_1dhl` is cross-day and requires the previous raw daily shard to exist
