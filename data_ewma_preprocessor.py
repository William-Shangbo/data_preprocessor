import argparse
import gc
import os
import sys
import time
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm

WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
WILLDEN_FILE_METHOD = WORKSPACE_ROOT / 'willden' / 'file_method'
WILLDEN_DATA_METHOD = WORKSPACE_ROOT / 'willden' / 'data_method'

for module_path in (WILLDEN_FILE_METHOD, WILLDEN_DATA_METHOD):
    module_path_str = str(module_path)
    if module_path.exists() and module_path_str not in sys.path:
        sys.path.append(module_path_str)

from file_management import quick_read
from data_processing import rank, TIMEIDS_PER_DAY
from ewma import causal_ewma, causal_ewma_impute

FEATURE_COLS = [f'f{i}' for i in range(384)]
KEY_COLS = ['stockid', 'dateid', 'timeid']
LABEL_COLS = ['LabelA', 'LabelB', 'LabelC']
XSEC_RANK_COLS = [f'f{i}_xsec_rank' for i in range(384)]
SAFE_NUM_WORKERS = 2
DEFAULT_STOCKID_RANGE = range(500)
DEFAULT_IMPUTE_HALF_LIFE = 29


@dataclass(frozen=True)
class EwmaFeatureSpec:
    half_life: int
    min_count: int
    suffix: str
    cross_day: bool = False


EWMA_FEATURE_SPECS = [
    EwmaFeatureSpec(half_life=5, min_count=1, suffix='ewma_5hl', cross_day=False),
    EwmaFeatureSpec(half_life=30, min_count=1, suffix='ewma_30hl', cross_day=False),
    EwmaFeatureSpec(half_life=60, min_count=1, suffix='ewma_60hl', cross_day=False),
    EwmaFeatureSpec(half_life=239, min_count=1, suffix='ewma_1dhl', cross_day=True),
]

EXPECTED_OUTPUT_COLS = len(KEY_COLS) + len(LABEL_COLS) + len(XSEC_RANK_COLS) * (1 + len(EWMA_FEATURE_SPECS))


def load_raw_date(file_path, dateid, stockid_range=DEFAULT_STOCKID_RANGE):
    return quick_read(
        file_path,
        stockid_range=stockid_range,
        dateid_range=(dateid, dateid),
        timeid_range=(0, TIMEIDS_PER_DAY - 1),
    )


def load_raw_daily_shard(raw_daily_dir: str, dateid: int) -> pd.DataFrame:
    shard_path = Path(raw_daily_dir) / f'd{dateid}.parquet'
    if not shard_path.exists():
        return pd.DataFrame()
    return pd.read_parquet(shard_path)


def _read_parquet_metadata(path: str | Path) -> tuple[int, int]:
    parquet_file = pq.ParquetFile(path)
    return parquet_file.metadata.num_rows, len(parquet_file.schema_arrow.names)


def _inspect_output_file(path: str | Path) -> dict:
    file_path = Path(path)
    if not file_path.exists():
        return {'exists': False, 'rows': 0, 'cols': 0, 'readable': False}

    try:
        rows, cols = _read_parquet_metadata(file_path)
        return {'exists': True, 'rows': rows, 'cols': cols, 'readable': True}
    except Exception:
        return {'exists': True, 'rows': 0, 'cols': 0, 'readable': False}


def verify_output_range(output_dir: str, start_dateid: int, end_dateid: int) -> dict:
    missing_dateids = []
    unreadable_dateids = []
    empty_dateids = []
    schema_mismatch_dateids = []
    rows_by_dateid = {}

    for dateid in range(start_dateid, end_dateid):
        output_path = Path(output_dir) / f'd{dateid}.parquet'
        info = _inspect_output_file(output_path)
        if not info['exists']:
            missing_dateids.append(dateid)
            continue
        if not info['readable']:
            unreadable_dateids.append(dateid)
            continue
        rows_by_dateid[dateid] = info['rows']
        if info['rows'] <= 0:
            empty_dateids.append(dateid)
        if info['cols'] != EXPECTED_OUTPUT_COLS:
            schema_mismatch_dateids.append({'dateid': dateid, 'cols': info['cols']})

    return {
        'missing_dateids': missing_dateids,
        'unreadable_dateids': unreadable_dateids,
        'empty_dateids': empty_dateids,
        'schema_mismatch_dateids': schema_mismatch_dateids,
        'rows_by_dateid': rows_by_dateid,
    }


def _rank_feature_block(df: pd.DataFrame, source_cols: list[str], suffix: str | None = None) -> pd.DataFrame:
    ranked = rank(df, columns=source_cols, by=['dateid', 'timeid'])
    if suffix is None:
        ranked.columns = XSEC_RANK_COLS
    else:
        rename_map = {f'{col}_r': f'{col}_xsec_rank_{suffix}' for col in source_cols}
        ranked = ranked.rename(columns=rename_map)
    return ranked.reset_index(drop=True)


def _build_imputed_source(
    source_df: pd.DataFrame,
    order_cols: list[str],
    impute_half_life: int,
) -> pd.DataFrame:
    ordered = source_df.sort_values(['stockid'] + order_cols).reset_index(drop=True)
    imputed_values = causal_ewma_impute(
        df=ordered,
        value_cols=FEATURE_COLS,
        group_cols=['stockid'],
        order_cols=order_cols,
        half_life=impute_half_life,
        min_count=1,
    )
    imputed_df = pd.concat([ordered[KEY_COLS].reset_index(drop=True), imputed_values.reset_index(drop=True)], axis=1)
    return imputed_df


def _log_stage(dateid: int, message: str, stage_start_time: float | None = None) -> None:
    if stage_start_time is None:
        print(f"[ewma-pre] dateid={dateid} {message}", flush=True)
        return
    elapsed = time.time() - stage_start_time
    print(f"[ewma-pre] dateid={dateid} {message} elapsed={elapsed:.2f}s", flush=True)


def _build_present_rank_features(imputed_day_df: pd.DataFrame) -> pd.DataFrame:
    imputed_df = imputed_day_df.sort_values(['dateid', 'timeid', 'stockid']).reset_index(drop=True)
    return _rank_feature_block(imputed_df, FEATURE_COLS, suffix=None)


def _build_ewma_rank_features(
    current_day_df: pd.DataFrame,
    imputed_source_df: pd.DataFrame,
    spec: EwmaFeatureSpec,
) -> pd.DataFrame:
    source_order_cols = ['dateid', 'timeid'] if spec.cross_day else ['timeid']
    ewma_values = causal_ewma(
        df=imputed_source_df,
        value_cols=FEATURE_COLS,
        group_cols=['stockid'],
        order_cols=source_order_cols,
        half_life=spec.half_life,
        min_count=spec.min_count,
    )
    ewma_source = pd.concat([imputed_source_df[KEY_COLS].reset_index(drop=True), ewma_values.reset_index(drop=True)], axis=1)
    ewma_current_day = ewma_source[ewma_source['dateid'] == int(current_day_df['dateid'].iloc[0])].copy()
    ewma_current_day = ewma_current_day.sort_values(['dateid', 'timeid', 'stockid']).reset_index(drop=True)
    return _rank_feature_block(ewma_current_day, FEATURE_COLS, suffix=spec.suffix)


def build_daily_ewma_packet(
    file_path: str | None,
    dateid: int,
    output_dir: str,
    stockid_range=DEFAULT_STOCKID_RANGE,
    impute_half_life: int = DEFAULT_IMPUTE_HALF_LIFE,
    raw_daily_dir: str | None = None,
) -> dict | None:
    start_time = time.time()
    _log_stage(dateid, 'loading current-day raw parquet slice')
    if raw_daily_dir is not None:
        day_df = load_raw_daily_shard(raw_daily_dir, dateid)
    else:
        day_df = load_raw_date(file_path, dateid, stockid_range=stockid_range)
    if day_df.empty:
        _log_stage(dateid, 'raw slice is empty, skipping output')
        return None
    _log_stage(dateid, f'loaded current-day rows={len(day_df)} cols={len(day_df.columns)}', start_time)

    base_cols = [col for col in KEY_COLS + LABEL_COLS if col in day_df.columns]
    sorted_day_base = day_df[base_cols].sort_values(['dateid', 'timeid', 'stockid']).reset_index(drop=True)
    output_blocks = [sorted_day_base]

    # Flow: fetch raw -> impute missing raw values -> calculate ewma(raw) -> xsec rank -> pack output.
    same_day_impute_start = time.time()
    _log_stage(dateid, f'imputing same-day raw features half_life={impute_half_life}')
    imputed_day_source = _build_imputed_source(
        source_df=day_df[KEY_COLS + FEATURE_COLS],
        order_cols=['timeid'],
        impute_half_life=impute_half_life,
    )
    _log_stage(dateid, 'finished same-day raw impute', same_day_impute_start)

    present_rank_start = time.time()
    _log_stage(dateid, 'building present xsec_rank from imputed raw values')
    output_blocks.append(_build_present_rank_features(imputed_day_source))
    _log_stage(dateid, 'finished present xsec_rank', present_rank_start)

    previous_day_df = None
    if dateid > 0:
        previous_day_start = time.time()
        _log_stage(dateid, f'loading previous-day raw slice dateid={dateid - 1} for cross-day features')
        if raw_daily_dir is not None:
            previous_day_df = load_raw_daily_shard(raw_daily_dir, dateid - 1)
        else:
            previous_day_df = load_raw_date(file_path, dateid - 1, stockid_range=stockid_range)
        if previous_day_df is None or previous_day_df.empty:
            raise FileNotFoundError(
                f'missing previous-day raw data for dateid={dateid - 1}; '
                f'cross-day EWMA features for dateid={dateid} would be invalid'
            )
        _log_stage(
            dateid,
            f'loaded previous-day rows={0 if previous_day_df is None else len(previous_day_df)}',
            previous_day_start,
        )

    imputed_cross_day_source = None
    if previous_day_df is not None and not previous_day_df.empty:
        cross_day_impute_start = time.time()
        _log_stage(dateid, f'imputing cross-day raw features half_life={impute_half_life}')
        cross_day_source = pd.concat([previous_day_df[KEY_COLS + FEATURE_COLS], day_df[KEY_COLS + FEATURE_COLS]], ignore_index=True)
        imputed_cross_day_source = _build_imputed_source(
            source_df=cross_day_source,
            order_cols=['dateid', 'timeid'],
            impute_half_life=impute_half_life,
        )
        _log_stage(dateid, 'finished cross-day raw impute', cross_day_impute_start)

    for spec in EWMA_FEATURE_SPECS:
        ewma_stage_start = time.time()
        _log_stage(
            dateid,
            f'calculating {spec.suffix} from raw values half_life={spec.half_life} cross_day={spec.cross_day}',
        )
        if spec.cross_day and imputed_cross_day_source is not None:
            imputed_source_df = imputed_cross_day_source
        else:
            imputed_source_df = imputed_day_source

        output_blocks.append(
            _build_ewma_rank_features(
                current_day_df=day_df,
                imputed_source_df=imputed_source_df,
                spec=spec,
            )
        )
        _log_stage(dateid, f'finished {spec.suffix}', ewma_stage_start)

    write_start = time.time()
    _log_stage(dateid, 'packing output blocks and writing parquet')
    output_df = pd.concat(output_blocks, axis=1)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'd{dateid}.parquet')
    temp_output_path = f'{output_path}.tmp'
    output_df.to_parquet(temp_output_path, index=False)
    os.replace(temp_output_path, output_path)
    _log_stage(dateid, f'finished parquet write path={output_path}', write_start)

    elapsed_time = time.time() - start_time
    del day_df, previous_day_df, output_df, output_blocks, imputed_day_source, imputed_cross_day_source, sorted_day_base
    gc.collect()
    return {
        'status': 'ok',
        'dateid': dateid,
        'output_path': output_path,
        'elapsed_time': elapsed_time,
    }


def _process_single_date_worker(args):
    dateid = args[1]
    try:
        return build_daily_ewma_packet(*args)
    except Exception as exc:
        return {
            'status': 'error',
            'dateid': dateid,
            'error': repr(exc),
        }


def process_date_range_serial(
    file_path,
    output_dir,
    start_dateid=0,
    end_dateid=30,
    stockid_range=DEFAULT_STOCKID_RANGE,
    impute_half_life: int = DEFAULT_IMPUTE_HALF_LIFE,
    raw_daily_dir: str | None = None,
):
    dateids = list(range(start_dateid, end_dateid))
    results = []
    errors = []
    total_start_time = time.time()

    for dateid in tqdm(dateids, desc='dateids', total=len(dateids)):
        result = build_daily_ewma_packet(
            file_path,
            dateid,
            output_dir,
            stockid_range=stockid_range,
            impute_half_life=impute_half_life,
            raw_daily_dir=raw_daily_dir,
        )
        if result is not None:
            results.append(result)

    total_elapsed_time = time.time() - total_start_time
    return results, errors, total_elapsed_time


def process_date_range_parallel(
    file_path,
    output_dir,
    start_dateid=0,
    end_dateid=30,
    num_workers=SAFE_NUM_WORKERS,
    stockid_range=DEFAULT_STOCKID_RANGE,
    impute_half_life: int = DEFAULT_IMPUTE_HALF_LIFE,
    raw_daily_dir: str | None = None,
):
    args_list = [
        (file_path, dateid, output_dir, stockid_range, impute_half_life, raw_daily_dir)
        for dateid in range(start_dateid, end_dateid)
    ]

    total_start_time = time.time()
    results = []
    errors = []

    with Pool(processes=num_workers, maxtasksperchild=1) as pool:
        for result in tqdm(
            pool.imap_unordered(_process_single_date_worker, args_list),
            desc='dateids',
            total=len(args_list),
        ):
            if result is None:
                continue
            if result.get('status') == 'error':
                errors.append(result)
                print(
                    f"[ewma-pre] dateid={result['dateid']} worker_error={result['error']}",
                    flush=True,
                )
                continue
            if result is not None:
                results.append(result)

    total_elapsed_time = time.time() - total_start_time
    results.sort(key=lambda item: item['dateid'])
    errors.sort(key=lambda item: item['dateid'])
    return results, errors, total_elapsed_time


def main():
    parser = argparse.ArgumentParser(
        description='Build daily EWMA rank packets: raw fetch/shard -> impute -> ewma(raw) -> xsec rank'
    )
    parser.add_argument('--file_path', type=str, default=None, help='Path to raw source parquet file')
    parser.add_argument('--raw_daily_dir', type=str, default=None, help='Directory containing raw daily shard parquet files')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--start_dateid', type=int, default=0, help='Start dateid')
    parser.add_argument('--end_dateid', type=int, default=30, help='End dateid')
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
    parser.add_argument(
        '--impute_half_life',
        type=int,
        default=DEFAULT_IMPUTE_HALF_LIFE,
        help='Half-life used by causal EWMA imputation on raw values',
    )
    parser.add_argument(
        '--skip_existing',
        action='store_true',
        help='Skip dateids whose output parquet already exists and is readable',
    )

    args = parser.parse_args()
    if not args.file_path and not args.raw_daily_dir:
        raise ValueError('Either --file_path or --raw_daily_dir must be provided')

    start_dateid = args.start_dateid
    end_dateid = args.end_dateid
    if args.skip_existing:
        pending_dateids = []
        for dateid in range(start_dateid, end_dateid):
            output_path = Path(args.output_dir) / f'd{dateid}.parquet'
            info = _inspect_output_file(output_path)
            if info['exists'] and info['readable'] and info['rows'] > 0:
                continue
            pending_dateids.append(dateid)

        if not pending_dateids:
            print('All requested dateids already have readable outputs; verifying range only.')
            start_dateid = end_dateid
        else:
            start_dateid = pending_dateids[0]
            end_dateid = pending_dateids[-1] + 1
            print(
                f"skip_existing enabled; pending dateids={pending_dateids} "
                f"collapsed_to_contiguous_range=[{start_dateid}, {end_dateid})"
            )

    if args.mode == 'parallel':
        results, errors, total_elapsed_time = process_date_range_parallel(
            file_path=args.file_path,
            raw_daily_dir=args.raw_daily_dir,
            output_dir=args.output_dir,
            start_dateid=start_dateid,
            end_dateid=end_dateid,
            num_workers=args.num_workers,
            impute_half_life=args.impute_half_life,
        )
    else:
        results, errors, total_elapsed_time = process_date_range_serial(
            file_path=args.file_path,
            raw_daily_dir=args.raw_daily_dir,
            output_dir=args.output_dir,
            start_dateid=start_dateid,
            end_dateid=end_dateid,
            impute_half_life=args.impute_half_life,
        )

    print(f'\nProcessed {len(results)} dateids in {total_elapsed_time:.2f}s')
    for result in results:
        print(
            f"dateid={result['dateid']} elapsed={result['elapsed_time']:.2f}s output={result['output_path']}"
        )
    if errors:
        print('\nWorker errors:')
        for error in errors:
            print(f"dateid={error['dateid']} error={error['error']}")

    verification = verify_output_range(args.output_dir, args.start_dateid, args.end_dateid)
    print('\nVerification summary:')
    print(f"missing_dateids={verification['missing_dateids']}")
    print(f"unreadable_dateids={verification['unreadable_dateids']}")
    print(f"empty_dateids={verification['empty_dateids']}")
    print(f"schema_mismatch_dateids={verification['schema_mismatch_dateids']}")

    if errors or verification['missing_dateids'] or verification['unreadable_dateids'] or verification['empty_dateids']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
