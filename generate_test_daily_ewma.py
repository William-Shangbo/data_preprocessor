import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate daily EWMA packets for test data using the same anonymous-feature pipeline"
    )
    parser.add_argument(
        "--raw_daily_dir",
        type=str,
        default="/Volumes/Lenovo/mytrae/ljcomp/test_daily_data_raw",
        help="Directory containing test raw daily shards",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/Volumes/Lenovo/mytrae/ljcomp/test_daily_data_ewma",
        help="Output directory for test daily EWMA packets",
    )
    parser.add_argument("--start_dateid", type=int, default=0, help="Inclusive start dateid")
    parser.add_argument("--end_dateid", type=int, default=60, help="Exclusive end dateid")
    parser.add_argument("--mode", choices=["serial", "parallel"], default="serial", help="Execution mode")
    parser.add_argument("--num_workers", type=int, default=2, help="Workers for parallel mode")
    parser.add_argument("--impute_half_life", type=int, default=29, help="Half-life for raw-value EWMA imputation")
    parser.add_argument("--skip_existing", action="store_true", help="Skip already generated readable outputs")
    args = parser.parse_args()

    entry = Path(__file__).resolve().parent / "data_ewma_preprocessor.py"
    cmd = [
        sys.executable,
        str(entry),
        "--raw_daily_dir",
        args.raw_daily_dir,
        "--output_dir",
        args.output_dir,
        "--start_dateid",
        str(args.start_dateid),
        "--end_dateid",
        str(args.end_dateid),
        "--mode",
        args.mode,
        "--num_workers",
        str(args.num_workers),
        "--impute_half_life",
        str(args.impute_half_life),
    ]
    if args.skip_existing:
        cmd.append("--skip_existing")

    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
