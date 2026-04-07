import argparse
import gc
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append('/Users/shangbo/personal/mytrae/willden/file_method')
sys.path.append('/Users/shangbo/personal/mytrae/willden/data_method')

from file_management import quick_read
from data_processing import rank, TIMEIDS_PER_DAY

FEATURE_COLS = [f'f{i}' for i in range(384)]
TS_RANK_COLS = [f'f{i}_ts_rank' for i in range(384)]
XSEC_RANK_COLS = [f'f{i}_xsec_rank' for i in range(384)]
KEY_COLS = ['stockid', 'dateid', 'timeid']
LABEL_COLS = ['LabelA', 'LabelB', 'LabelC']
DEFAULT_STOCKID_RANGE = range(500)
SAFE_NUM_WORKERS = 2


def load_single_date(file_path, dateid, stockid_range=DEFAULT_STOCKID_RANGE):
    return quick_read(
        file_path,
        stockid_range=stockid_range,
        dateid_range=(dateid, dateid),
        timeid_range=(0, TIMEIDS_PER_DAY - 1),
    )


def build_daily_packet_xsec_only(file_path, dateid, output_dir, stockid_range=DEFAULT_STOCKID_RANGE):
    start_time = time.time()

    date_df = load_single_date(file_path, dateid, stockid_range=stockid_range)
    if date_df.empty:
        return None

    output_base_cols = [col for col in KEY_COLS + LABEL_COLS if col in date_df.columns]
    xsec_ranks = rank(date_df, columns=FEATURE_COLS, by=['dateid', 'timeid'])
    xsec_ranks.columns = XSEC_RANK_COLS

    ts_placeholder = pd.DataFrame(
        np.nan,
        index=date_df.index,
        columns=TS_RANK_COLS,
    )

    output_df = pd.concat(
        [
            date_df[output_base_cols].reset_index(drop=True),
            xsec_ranks.reset_index(drop=True),
            ts_placeholder.reset_index(drop=True),
        ],
        axis=1,
    )

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'd{dateid}.parquet')
    output_df.to_parquet(output_path, index=False)

    elapsed_time = time.time() - start_time

    del date_df, xsec_ranks, ts_placeholder, output_df
    gc.collect()

    return {
        'dateid': dateid,
        'output_path': output_path,
        'elapsed_time': elapsed_time,
    }


def _process_single_date_worker(args):
    return build_daily_packet_xsec_only(*args)


def process_date_range_serial(
    file_path,
    output_dir,
    start_dateid=0,
    end_dateid=360,
    stockid_range=DEFAULT_STOCKID_RANGE,
):
    dateids = list(range(start_dateid, end_dateid))
    results = []
    total_start_time = time.time()

    for dateid in tqdm(dateids, desc='dateids', total=len(dateids)):
        result = build_daily_packet_xsec_only(
            file_path,
            dateid,
            output_dir,
            stockid_range=stockid_range,
        )
        if result is not None:
            results.append(result)

    total_elapsed_time = time.time() - total_start_time
    return results, total_elapsed_time


def process_date_range_parallel(
    file_path,
    output_dir,
    start_dateid=0,
    end_dateid=360,
    num_workers=SAFE_NUM_WORKERS,
    stockid_range=DEFAULT_STOCKID_RANGE,
):
    dateids = list(range(start_dateid, end_dateid))
    args_list = [
        (file_path, dateid, output_dir, stockid_range)
        for dateid in dateids
    ]

    total_start_time = time.time()
    results = []

    with Pool(processes=num_workers) as pool:
        for result in tqdm(
            pool.imap_unordered(_process_single_date_worker, args_list),
            desc='dateids',
            total=len(args_list),
        ):
            if result is not None:
                results.append(result)

    total_elapsed_time = time.time() - total_start_time
    results.sort(key=lambda item: item['dateid'])
    return results, total_elapsed_time


def main():
    parser = argparse.ArgumentParser(
        description='Build daily xsec-rank packets from parquet data while preserving the ts-rank schema'
    )
    parser.add_argument('--file_path', type=str, required=True, help='Path to parquet file')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--start_dateid', type=int, default=0, help='Start dateid')
    parser.add_argument('--end_dateid', type=int, default=360, help='End dateid')
    parser.add_argument(
        '--mode',
        choices=['serial', 'parallel'],
        default='parallel',
        help='Execution mode',
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=SAFE_NUM_WORKERS,
        help='Number of date-level worker processes when mode=parallel',
    )

    args = parser.parse_args()

    if args.mode == 'parallel':
        results, total_elapsed_time = process_date_range_parallel(
            file_path=args.file_path,
            output_dir=args.output_dir,
            start_dateid=args.start_dateid,
            end_dateid=args.end_dateid,
            num_workers=args.num_workers,
        )
    else:
        results, total_elapsed_time = process_date_range_serial(
            file_path=args.file_path,
            output_dir=args.output_dir,
            start_dateid=args.start_dateid,
            end_dateid=args.end_dateid,
        )

    print(f'\nProcessed {len(results)} dateids in {total_elapsed_time:.2f}s')
    for result in results:
        print(
            f"dateid={result['dateid']} elapsed={result['elapsed_time']:.2f}s output={result['output_path']}"
        )


if __name__ == '__main__':
    main()
