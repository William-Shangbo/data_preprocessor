import argparse
import os
import time
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def _log(message: str) -> None:
    print(f"[raw-shard] {message}", flush=True)


def build_raw_daily_shards(
    file_path: str,
    output_dir: str,
    start_dateid: int,
    end_dateid: int,
    batch_size: int,
) -> list[dict]:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    parquet_file = pq.ParquetFile(file_path)
    writers: dict[int, pq.ParquetWriter] = {}
    write_counts = {dateid: 0 for dateid in range(start_dateid, end_dateid)}
    row_counts = {dateid: 0 for dateid in range(start_dateid, end_dateid)}
    start_time = time.time()

    try:
        for batch_idx, batch in enumerate(parquet_file.iter_batches(batch_size=batch_size), start=1):
            batch_df = batch.to_pandas()
            batch_df = batch_df[
                (batch_df["dateid"] >= start_dateid) & (batch_df["dateid"] < end_dateid)
            ]
            if batch_df.empty:
                if batch_idx % 50 == 0:
                    _log(f"scanned batch={batch_idx} no matching rows yet")
                continue

            grouped = batch_df.groupby("dateid", sort=True)
            for dateid, date_df in grouped:
                dateid = int(dateid)
                output_path = output_root / f"d{dateid}.parquet"
                table = pa.Table.from_pandas(date_df.reset_index(drop=True), preserve_index=False)
                if dateid not in writers:
                    _log(f"opening writer for dateid={dateid} path={output_path}")
                    writers[dateid] = pq.ParquetWriter(output_path, table.schema)
                writers[dateid].write_table(table)
                write_counts[dateid] += 1
                row_counts[dateid] += len(date_df)

            if batch_idx % 10 == 0:
                done_days = sum(1 for rows in row_counts.values() if rows > 0)
                _log(
                    f"batch={batch_idx} active_days={done_days}/{end_dateid - start_dateid} "
                    f"rows_written={sum(row_counts.values())}"
                )
    finally:
        for writer in writers.values():
            writer.close()

    elapsed = time.time() - start_time
    results = []
    for dateid in range(start_dateid, end_dateid):
        if row_counts[dateid] == 0:
            continue
        results.append(
            {
                "dateid": dateid,
                "rows": row_counts[dateid],
                "writes": write_counts[dateid],
                "output_path": str(output_root / f"d{dateid}.parquet"),
            }
        )
    _log(f"finished in {elapsed:.2f}s produced_days={len(results)}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="One-pass shard builder: raw parquet -> raw daily parquet files")
    parser.add_argument("--file_path", type=str, required=True, help="Path to raw source parquet")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory for raw daily shards")
    parser.add_argument("--start_dateid", type=int, default=0, help="Inclusive start dateid")
    parser.add_argument("--end_dateid", type=int, default=30, help="Exclusive end dateid")
    parser.add_argument("--batch_size", type=int, default=200_000, help="Arrow batch size during single-pass scan")
    args = parser.parse_args()

    results = build_raw_daily_shards(
        file_path=args.file_path,
        output_dir=args.output_dir,
        start_dateid=args.start_dateid,
        end_dateid=args.end_dateid,
        batch_size=args.batch_size,
    )

    for result in results:
        print(
            f"dateid={result['dateid']} rows={result['rows']} writes={result['writes']} "
            f"output={result['output_path']}"
        )


if __name__ == "__main__":
    main()
