import pandas as pd
import os
import glob
import argparse
from collections import Counter

def analyze_csv_files(folder_path, target_column=None):
    """
    读取文件夹下所有CSV文件，找出共同列名，并统计指定列的值计数
    
    Args:
        folder_path: CSV文件所在文件夹路径
        target_column: 要统计的列名（如果为None，会显示所有共同列供选择）
    
    Returns:
        dict: 包含共同列名和指定列值计数的字典
    """
    # 获取所有CSV文件
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    
    if not csv_files:
        print(f"在 {folder_path} 中没有找到CSV文件")
        return {}
    
    print(f"找到 {len(csv_files)} 个CSV文件:")
    for file in csv_files:
        print(f"  - {os.path.basename(file)}")
    
    # 读取所有CSV文件并获取列名
    all_columns = []
    dataframes = []
    
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            dataframes.append(df)
            all_columns.append(set(df.columns))
            print(f"  {os.path.basename(file)}: {len(df.columns)} 列")
        except Exception as e:
            print(f"  读取 {os.path.basename(file)} 时出错: {e}")
    
    if not dataframes:
        print("没有成功读取任何CSV文件")
        return {}
    
    # 找出所有文件的共同列名
    common_columns = set.intersection(*all_columns) if all_columns else set()
    
    print(f"\n共同列名 ({len(common_columns)} 个):")
    for col in sorted(common_columns):
        print(f"  - {col}")
    
    # 如果没有指定目标列，显示所有共同列供选择
    if target_column is None:
        print("\n请从上述共同列中选择一个进行值计数统计")
        return {"common_columns": list(common_columns)}
    
    # 检查目标列是否存在
    if target_column not in common_columns:
        print(f"\n错误: '{target_column}' 不是所有文件的共同列")
        print("可用的共同列:", list(common_columns))
        return {"common_columns": list(common_columns)}
    
    # 统计指定列的值计数
    print(f"\n正在统计列 '{target_column}' 的值计数...")
    
    # 合并所有数据框的指定列
    all_values = []
    for i, df in enumerate(dataframes):
        values = df[target_column].dropna().tolist()
        all_values.extend(values)
        print(f"  {os.path.basename(csv_files[i])}: {len(values)} 个值")
    
    # 计算值计数
    value_counts = Counter(all_values)
    
    print(f"\n列 '{target_column}' 的值计数 (总计 {len(all_values)} 个值):")
    print("-" * 50)
    for value, count in value_counts.most_common():
        print(f"{value}: {count}")
    
    return {
        "common_columns": list(common_columns),
        "target_column": target_column,
        "value_counts": dict(value_counts),
        "total_values": len(all_values)
    }

def analyze_two_columns(folder_path, col_a, col_b):
    """
    同时统计两列（假定每列只有两种可能取值）：
    - 分别给出每列的值计数
    - 给出两列的二元交叉表（2x2）

    Returns:
        dict: {common_columns, columns: (col_a, col_b), counts_a, counts_b, crosstab}
    """
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    if not csv_files:
        print(f"在 {folder_path} 中没有找到CSV文件")
        return {}

    all_columns = []
    dataframes = []
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            dataframes.append(df)
            all_columns.append(set(df.columns))
        except Exception as e:
            print(f"  读取 {os.path.basename(file)} 时出错: {e}")

    if not dataframes:
        print("没有成功读取任何CSV文件")
        return {}

    common_columns = set.intersection(*all_columns) if all_columns else set()

    # 校验两列是否为共同列
    missing = [c for c in (col_a, col_b) if c not in common_columns]
    if missing:
        print(f"错误: 以下列不是所有文件的共同列: {missing}")
        print("可用的共同列:", sorted(common_columns))
        return {"common_columns": list(common_columns)}

    # 汇总两列的取值
    series_a_list = []
    series_b_list = []
    for df in dataframes:
        if col_a in df.columns:
            series_a_list.append(df[col_a].dropna())
        if col_b in df.columns:
            series_b_list.append(df[col_b].dropna())

    if not series_a_list or not series_b_list:
        print("没有可用于统计的数据")
        return {"common_columns": list(common_columns)}

    s_a = pd.concat(series_a_list, ignore_index=True)
    s_b = pd.concat(series_b_list, ignore_index=True)

    # 只保留两种取值的假设：检测各自唯一值，若多于2种，仅取最常见的前2种用于展示
    uniq_a = s_a.value_counts().index.tolist()
    uniq_b = s_b.value_counts().index.tolist()
    if len(uniq_a) > 2:
        uniq_a = uniq_a[:2]
    if len(uniq_b) > 2:
        uniq_b = uniq_b[:2]

    # 统一为字符串，避免 True/False 与 "True"/"False" 混淆
    s_a = s_a.astype(str)
    s_b = s_b.astype(str)
    uniq_a = [str(v) for v in uniq_a]
    uniq_b = [str(v) for v in uniq_b]

    # 单列计数（按两值顺序展示）
    counts_a = s_a.value_counts()
    counts_b = s_b.value_counts()
    counts_a = counts_a.reindex(uniq_a, fill_value=0)
    counts_b = counts_b.reindex(uniq_b, fill_value=0)

    # 2x2 交叉表
    ct = pd.crosstab(s_a, s_b)
    # 确保包含两值并按顺序排列
    for v in uniq_a:
        if v not in ct.index:
            ct.loc[v] = 0
    for v in uniq_b:
        if v not in ct.columns:
            ct[v] = 0
    ct = ct.loc[uniq_a, uniq_b].fillna(0).astype(int)

    print(f"\n=== 同时统计两列: '{col_a}' 与 '{col_b}' ===")
    print("-" * 50)
    print(f"{col_a} 值计数：")
    for v in uniq_a:
        print(f"  {v}: {int(counts_a.get(v, 0))}")
    print(f"{col_b} 值计数：")
    for v in uniq_b:
        print(f"  {v}: {int(counts_b.get(v, 0))}")

    print("\n交叉表 (行=\"" + col_a + "\"，列=\"" + col_b + "\")：")
    header = "\t" + "\t".join([str(v) for v in uniq_b])
    print(header)
    for ra in uniq_a:
        row_vals = "\t".join(str(int(ct.at[ra, cb])) for cb in uniq_b)
        print(f"{ra}\t{row_vals}")

    return {
        "common_columns": list(common_columns),
        "columns": (col_a, col_b),
        "counts_a": counts_a.to_dict(),
        "counts_b": counts_b.to_dict(),
        "crosstab": ct.to_dict()
    }

def analyze_multi_columns(folder_path, columns):
    """
    统合统计：支持 1..N 列。
      - 1 列：值计数
      - 2 列：各自值计数 + 2x2 交叉表（若超过两值，仅取前2常见值展示）
      - ≥3 列：各列值计数 + 组合频次表
    """
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    if not csv_files:
        print(f"在 {folder_path} 中没有找到CSV文件")
        return {}

    dataframes = []
    all_columns = []
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            dataframes.append(df)
            all_columns.append(set(df.columns))
        except Exception as e:
            print(f"  读取 {os.path.basename(file)} 时出错: {e}")

    if not dataframes:
        print("没有成功读取任何CSV文件")
        return {}

    common_columns = set.intersection(*all_columns) if all_columns else set()
    missing = [c for c in columns if c not in common_columns]
    if missing:
        print(f"错误: 以下列不是所有文件的共同列: {missing}")
        print("可用的共同列:", sorted(common_columns))
        return {"common_columns": list(common_columns)}

    merged = []
    for df in dataframes:
        merged.append(df[columns].copy())
    big = pd.concat(merged, ignore_index=True)

    print("\n=== 各列值计数（显示最常见的两种） ===")
    per_col_counts = {}
    for c in columns:
        counts = big[c].astype(str).value_counts()
        per_col_counts[c] = counts.to_dict()
        top2 = counts.head(2)
        print(f"- {c}:")
        for idx, val in top2.items():
            print(f"  {idx}: {int(val)}")

    if len(columns) == 1:
        return {"common_columns": list(common_columns), "counts": per_col_counts}

    if len(columns) == 2:
        return analyze_two_columns(folder_path, columns[0], columns[1])

    print("\n=== 多列组合频次表 ===")
    combo_counts = (
        big[columns]
        .astype(str)
        .value_counts()
        .reset_index(name="count")
    )
    head_n = min(50, len(combo_counts))
    print(f"显示前 {head_n} 行（共 {len(combo_counts)} 种组合）：")
    header = "\t".join(columns + ["count"])
    print(header)
    for i in range(head_n):
        row = combo_counts.iloc[i]
        fields = [str(row[c]) for c in columns] + [str(int(row["count"]))]
        print("\t".join(fields))

    return {
        "common_columns": list(common_columns),
        "columns": columns,
        "per_column_counts": per_col_counts,
        "combo_counts_head": combo_counts.head(head_n).to_dict(orient="records"),
        "total_combinations": int(len(combo_counts))
    }

def main():
    parser = argparse.ArgumentParser(description="CSV 计数与交叉分析")
    parser.add_argument(
        "--folder", required=False,
        default="/disk1/users/wangjh/supp_exp/results/pipelines/or-bench-hard-1k/implicit_pipeline",
        help="包含CSV文件的文件夹路径"
    )
    parser.add_argument(
        "--cols", nargs="+", help="要统计的列名，支持1个或多个"
    )
    args = parser.parse_args()

    folder_path = args.folder
    print("=== CSV文件分析工具 ===")
    print(f"分析文件夹: {folder_path}")

    if not args.cols:
        analyze_csv_files(folder_path)
        print("\n未提供 --cols。示例：--cols is_original_filtered_by_pipeline is_final_filtered_by_pipeline")
        return

    columns = args.cols
    if len(columns) == 1:
        analyze_csv_files(folder_path, columns[0])
    else:
        analyze_multi_columns(folder_path, columns)

if __name__ == "__main__":
    main()
