import re
import pandas as pd
import os
import glob

def parse_all_logs(directory_path, output_csv='history_data_full.csv'):
    """
    扫描指定目录下的所有 .log 文件，提取交易信号并合并
    """
    if not os.path.isdir(directory_path):
        print(f"错误: 找不到目录 {directory_path}")
        return

    # 正则表达式：匹配你的信号行
    signal_pattern = r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*regime=(\w+) dd=([-\d.]+)% slope=([-\d.]+) gate=([-\d.]+) action=(\w+) strength=([-\d.]+)"
    
    all_data = []

    # 扫描目录下所有的文件 (包括子目录)
    # 你可以根据需要修改后缀，比如 *.log* 匹配 app.log.1 这种文件
    search_pattern = os.path.join(directory_path, "**", "*.log*")
    log_files = glob.glob(search_pattern, recursive=True)

    print(f"找到 {len(log_files)} 个潜在日志文件，开始解析...")

    for file_path in log_files:
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                file_count = 0
                for line in f:
                    match = re.search(signal_pattern, line)
                    if match:
                        all_data.append({
                            'timestamp': match.group(1),
                            'regime': match.group(2),
                            'dd': float(match.group(3)),
                            'slope': float(match.group(4)),
                            'gate': float(match.group(5)),
                            'action': match.group(6),
                            'strength': float(match.group(7)),
                            'source_file': os.path.basename(file_path) # 记录来源方便排查
                        })
                        file_count += 1
                if file_count > 0:
                    print(f"  - {os.path.basename(file_path)}: 提取到 {file_count} 条记录")
        except Exception as e:
            print(f"  - 无法读取文件 {file_path}: {e}")

    if not all_data:
        print("未在目录中发现匹配的信号数据。")
        return

    # 汇总数据
    df = pd.DataFrame(all_data)
    
    # 核心操作：转换时间 -> 排序 -> 去重
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(by='timestamp')
    
    # 因为日志中可能存在重复记录，去重保留第一个
    initial_len = len(df)
    df = df.drop_duplicates(subset=['timestamp'])
    
    print("-" * 30)
    print(f"解析完成！")
    print(f"扫描文件数: {len(log_files)}")
    print(f"原始记录总数: {initial_len}")
    print(f"去重后有效记录: {len(df)}")
    print(f"时间跨度: {df['timestamp'].min()} 至 {df['timestamp'].max()}")
    
    # 保存结果
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"合并后的数据已保存至: {output_csv}")
    print("-" * 30)
    
    return df

