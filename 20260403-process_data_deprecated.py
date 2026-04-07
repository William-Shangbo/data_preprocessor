import sys
import os
import pandas as pd
import numpy as np
from multiprocessing import Pool
import time
import gc

sys.path.append('/Users/shangbo/personal/mytrae/willden/file_method')
sys.path.append('/Users/shangbo/personal/mytrae/willden/data_method')

from file_management import read_byclass, quick_read
from data_processing import rank, TIMEIDS_PER_DAY, MAX_HISTORICAL_TIMESTAMPS

FEATURE_COLS = [f'f{i}' for i in range(384)]
NUM_WORKERS = 5
TS_RANK_COLS = [f'f{i}_ts_rank' for i in range(384)]
XSEC_RANK_COLS = [f'f{i}_xsec_rank' for i in range(384)]


def calculate_historical_range(dateid, timeid, window_size=MAX_HISTORICAL_TIMESTAMPS):
    """
    计算历史时间戳范围
    
    Args:
        dateid: 目标日期ID
        timeid: 目标时间ID
        window_size: 向前回看的最大时间戳数量
    
    Returns:
        历史时间戳列表，按时间顺序排列（从早到晚）
    """
    timestamps = []
    current_dateid, current_timeid = dateid, timeid
    
    effective_window = min(window_size, MAX_HISTORICAL_TIMESTAMPS)

    for _ in range(effective_window):
        timestamps.append((current_dateid, current_timeid))
        
        if current_dateid == 0 and current_timeid == 0:
            break
        
        current_timeid -= 1
        if current_timeid < 0:
            current_timeid = TIMEIDS_PER_DAY - 1
            current_dateid -= 1
            if current_dateid < 0:
                break
    
    timestamps.reverse()
    return timestamps


def read_historical_data(file_path, historical_timestamps, stockid_range=None):
    """
    读取历史数据（优化版本）
    
    Args:
        file_path: parquet文件路径
        historical_timestamps: 历史时间戳列表
        stockid_range: 股票ID范围，默认None表示读取所有股票
    
    Returns:
        包含历史数据的DataFrame
    """
    if not historical_timestamps:
        return pd.DataFrame()
    
    # 获取目标时间戳
    target_dateid, target_timeid = historical_timestamps[-1]
    
    # 计算需要读取的dateid范围
    dateids = [ts[0] for ts in historical_timestamps]
    min_dateid, max_dateid = min(dateids), max(dateids)
    
    # 根据dateid选择读取策略
    if target_dateid >= 2:
        # 读取三天的数据（dateid=d, d-1, d-2），使用quick_read
        read_dateids = list(range(max(0, target_dateid - 2), target_dateid + 1))
        
        # 使用quick_read读取三天的所有数据
        df = quick_read(
            file_path,
            stockid_range=stockid_range,
            dateid_range=(min(read_dateids), max(read_dateids)),
            timeid_range=(0, TIMEIDS_PER_DAY - 1)
        )
    else:
        # 对于dateid=0或1，直接读取需要的dateid范围
        df = quick_read(
            file_path,
            stockid_range=stockid_range,
            dateid_range=(min_dateid, max_dateid),
            timeid_range=(0, TIMEIDS_PER_DAY - 1)
        )
    
    if df.empty:
        return pd.DataFrame()
    
    # 精确过滤：只保留在historical_timestamps中的行
    timestamp_set = set(historical_timestamps)
    df['timestamp'] = list(zip(df['dateid'], df['timeid']))
    result = df[df['timestamp'].isin(timestamp_set)].copy()
    result = result.drop('timestamp', axis=1)
    
    return result


def process_single_timestamp(file_path, dateid, timeid, stockid_range=range(500)):
    """
    处理单个时间戳的数据
    
    Args:
        file_path: parquet文件路径
        dateid: 目标日期ID
        timeid: 目标时间ID
    Returns:
        处理后的DataFrame，包含排名特征
    """
    historical_timestamps = calculate_historical_range(dateid, timeid)
    
    df = read_historical_data(file_path, historical_timestamps, stockid_range=stockid_range)
    
    if df.empty:
        return pd.DataFrame()
    
    # 计算时序排名（基于所有历史数据）
    ts_ranks = rank(df, columns=FEATURE_COLS, by=['stockid'])
    
    # 计算截面排名（只基于当前时间戳的数据）
    current_data = df[(df['dateid'] == dateid) & (df['timeid'] == timeid)].copy()
    xsec_ranks = rank(current_data, columns=FEATURE_COLS, by=['dateid', 'timeid'])
    
    current_index = current_data.index
    ts_ranks = ts_ranks.loc[current_index].copy()
    xsec_ranks = xsec_ranks.loc[current_index].copy()

    ts_ranks.columns = TS_RANK_COLS
    xsec_ranks.columns = XSEC_RANK_COLS

    # 只保留当前时间戳的行，并重置索引避免历史窗口索引残留。
    result = pd.concat(
        [
            current_data.reset_index(drop=True),
            xsec_ranks.reset_index(drop=True),
            ts_ranks.reset_index(drop=True),
        ],
        axis=1,
    )

    return result


def process_single_date(file_path, dateid, output_dir, num_workers=NUM_WORKERS):
    """
    处理单个日期的所有时间戳
    
    Args:
        file_path: parquet文件路径
        dateid: 日期ID
        output_dir: 输出目录
        num_workers: 并行worker数量
    """
    print(f"Processing dateid={dateid}...")
    start_time = time.time()
    
    args_list = [(file_path, dateid, timeid) for timeid in range(TIMEIDS_PER_DAY)]
    
    with Pool(processes=num_workers) as pool:
        results = pool.starmap(process_single_timestamp, args_list)
    
    valid_results = [r for r in results if not r.empty]
    
    if not valid_results:
        print(f"  No valid data for dateid={dateid}")
        return
    
    date_df = pd.concat(valid_results, ignore_index=True)
    
    output_path = os.path.join(output_dir, f'd{dateid}.parquet')
    date_df.to_parquet(output_path, index=False)
    
    elapsed_time = time.time() - start_time
    print(f"  Completed dateid={dateid} in {elapsed_time:.2f}s")
    print(f"  Output: {output_path}")
    print(f"  Shape: {date_df.shape}")
    
    del date_df, valid_results, results
    gc.collect()


def process_all_dates(file_path, output_dir, start_dateid=0, end_dateid=360, num_workers=NUM_WORKERS):
    """
    处理所有日期
    
    Args:
        file_path: parquet文件路径
        output_dir: 输出目录
        start_dateid: 起始日期ID
        end_dateid: 结束日期ID
        num_workers: 并行worker数量
    """
    os.makedirs(output_dir, exist_ok=True)
    
    total_start_time = time.time()
    
    for dateid in range(start_dateid, end_dateid):
        process_single_date(file_path, dateid, output_dir, num_workers)
    
    total_elapsed_time = time.time() - total_start_time
    print(f"\nAll dates processed in {total_elapsed_time:.2f}s")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Process parquet data with rank features')
    parser.add_argument('--file_path', type=str, required=True, help='Path to parquet file')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--start_dateid', type=int, default=0, help='Start dateid')
    parser.add_argument('--end_dateid', type=int, default=360, help='End dateid')
    parser.add_argument('--num_workers', type=int, default=NUM_WORKERS, help='Number of workers')
    
    args = parser.parse_args()
    
    process_all_dates(
        file_path=args.file_path,
        output_dir=args.output_dir,
        start_dateid=args.start_dateid,
        end_dateid=args.end_dateid,
        num_workers=args.num_workers
    )


if __name__ == '__main__':
    main()
