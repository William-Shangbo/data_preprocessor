# data_preprocessor

Data preprocessing pipeline for generating daily parquet packets used by the `ljcomp` training workflow.

## What It Does

- Reads raw parquet data from `train.parquet` or `test.parquet`
- Builds daily output files `d{dateid}.parquet`
- Preserves a rank-based schema for downstream training
- Currently supports an efficient `xsec_rank`-only production path

## Current Output Schema

Each daily parquet keeps:

- keys: `stockid`, `dateid`, `timeid`
- labels: `LabelA`, `LabelB`, `LabelC` when present
- `f0_xsec_rank ~ f383_xsec_rank`
- `f0_ts_rank ~ f383_ts_rank`

Notes:

- raw feature columns `f0 ~ f383` are intentionally dropped from daily outputs
- `ts_rank` columns are currently retained as schema placeholders and filled with `NaN`

## Main Files

- `build_daily_rank_packets.py`: main executable pipeline
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
