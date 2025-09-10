import pandas as pd
import os
import glob

def read_and_concatenate_csvs(folder_path, file_pattern="*.csv"):
    """
    读取指定文件夹下的所有CSV文件并拼接成一个DataFrame
    
    参数:
    folder_path (str): 文件夹路径
    file_pattern (str): 文件匹配模式，默认为"*.csv"
    
    返回:
    pandas.DataFrame: 拼接后的DataFrame，保持与输入相同的列结构
    """
    # 获取所有CSV文件路径
    csv_files = glob.glob(os.path.join(folder_path, file_pattern))
    
    if not csv_files:
        print(f"在文件夹 {folder_path} 中没有找到匹配 {file_pattern} 的文件")
        return pd.DataFrame()
    
    print(f"找到 {len(csv_files)} 个CSV文件:")
    for file in csv_files:
        print(f"  - {os.path.basename(file)}")
    
    # 读取所有CSV文件并存储到列表中
    dataframes = []
    for file in csv_files:
        try:
            # 使用更宽松的参数来读取CSV文件
            df = pd.read_csv(file, 
                           on_bad_lines='skip',  # 跳过有问题的行
                           engine='python',      # 使用python引擎，更灵活
                           quoting=1,            # 处理引号
                           skipinitialspace=True)  # 跳过初始空格
            
            # 检查列数并调整DataFrame结构（不修改原始文件）
            if len(df.columns) == 11:
                print(f"检测到11列，在倒数第一和第二列之间插入空白列: {os.path.basename(file)}")
                # 获取列名列表
                columns = list(df.columns)
                # 在倒数第一和第二列之间插入空白列
                columns.insert(-1, 'blank_column')
                # 重新排列DataFrame的列
                df = df.reindex(columns=columns)
                # 将新插入的列填充为NaN
                df['blank_column'] = None
                print(f"调整后列数: {len(df.columns)}")
                # 输出一行作为演示
                print(f"演示 - 调整后的列名: {list(df.columns)}")
            elif len(df.columns) == 12:
                print(f"检测到12列，无需调整: {os.path.basename(file)}")
            else:
                print(f"警告: 检测到 {len(df.columns)} 列，不是预期的11或12列: {os.path.basename(file)}")
            
            # 添加源文件信息列（可选）
            df['source_file'] = os.path.basename(file)
            dataframes.append(df)
            print(f"成功读取 {os.path.basename(file)}: {len(df)} 行, {len(df.columns)} 列")
        except Exception as e:
            print(f"读取文件 {file} 时出错: {e}")
            # 尝试使用更宽松的方法
            try:
                print(f"尝试使用更宽松的方法读取 {os.path.basename(file)}...")
                df = pd.read_csv(file, 
                               sep=',', 
                               on_bad_lines='skip',
                               engine='python',
                               quoting=3,  # QUOTE_NONE
                               skipinitialspace=True,
                               error_bad_lines=False)
                
                # 同样检查列数并调整DataFrame结构（不修改原始文件）
                if len(df.columns) == 11:
                    print(f"检测到11列，在倒数第一和第二列之间插入空白列: {os.path.basename(file)}")
                    columns = list(df.columns)
                    columns.insert(-1, 'blank_column')
                    df = df.reindex(columns=columns)
                    df['blank_column'] = None
                    print(f"调整后列数: {len(df.columns)}")
                    # 输出一行作为演示
                    print(f"演示 - 调整后的列名: {list(df.columns)}")
                elif len(df.columns) == 12:
                    print(f"检测到12列，无需调整: {os.path.basename(file)}")
                else:
                    print(f"警告: 检测到 {len(df.columns)} 列，不是预期的11或12列: {os.path.basename(file)}")
                
                df['source_file'] = os.path.basename(file)
                dataframes.append(df)
                print(f"成功读取 {os.path.basename(file)}: {len(df)} 行, {len(df.columns)} 列")
            except Exception as e2:
                print(f"仍然无法读取 {file}: {e2}")
                continue
    
    if not dataframes:
        print("没有成功读取任何CSV文件")
        return pd.DataFrame()
    
    # 拼接所有DataFrame
    if len(dataframes) > 1:
        # 检查所有DataFrame的列是否一致
        all_columns = [set(df.columns) for df in dataframes]
        common_columns = set.intersection(*all_columns)
        
        if len(common_columns) > 0:
            print(f"\n发现 {len(common_columns)} 个共同列: {list(common_columns)}")
            
            # 只保留共同列进行拼接
            aligned_dataframes = []
            for df in dataframes:
                # 只保留共同列，缺失的列用NaN填充
                aligned_df = df.reindex(columns=common_columns)
                aligned_dataframes.append(aligned_df)
            
            combined_df = pd.concat(aligned_dataframes, ignore_index=True)
            print(f"使用共同列拼接完成! 总共 {len(combined_df)} 行数据")
        else:
            # 如果没有共同列，使用outer join方式拼接
            print("\n没有发现共同列，使用outer join方式拼接...")
            combined_df = pd.concat(dataframes, ignore_index=True, sort=False)
            print(f"拼接完成! 总共 {len(combined_df)} 行数据")
    else:
        combined_df = dataframes[0] if dataframes else pd.DataFrame()
    
    print(f"最终列名: {list(combined_df.columns)}")
    
    return combined_df

# 使用示例
if __name__ == "__main__":
    # 示例1: 读取当前目录下的所有CSV文件
    # df = read_and_concatenate_csvs("./")
    
    # 示例2: 读取特定文件夹下的CSV文件
    # df = read_and_concatenate_csvs("./results/pipelines/SAP/composed_pipeline/")
    
    # 示例3: 读取特定模式的CSV文件
    # df = read_and_concatenate_csvs("./results/", "TA-*results.csv")
    
    # 读取or文件夹下的CSV文件
    df = read_and_concatenate_csvs("./pipelines/SAP/implicit_pipeline/")
    
    if not df.empty:
        # 查看数据基本信息
        print(f'\n数据总长度: {len(df)}')
        
        # 分析某个列的值分布
        column_name = "is_filtered_by_pipeline"
        if column_name in df.columns:
            counts = df[column_name].value_counts()
            print(f'\n{column_name} 列的值分布:')
            print(counts)
        else:
            print(f"列 '{column_name}' 不存在")
            print("可用的列:", list(df.columns))
