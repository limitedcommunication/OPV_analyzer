import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from rdkit import Chem
from rdkit.Chem import Draw
import os
from tkinter import Tk, filedialog, simpledialog
from math import sqrt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (mean_squared_error,
                            r2_score,
                            mean_absolute_error,
                            explained_variance_score)
from xgboost import XGBRegressor
from scipy.stats import pearsonr
from skopt import BayesSearchCV
from skopt.space import Integer, Real, Categorical 
from skopt.callbacks import DeadlineStopper
from sklearn.model_selection import GridSearchCV

def select_csv_file_and_target():
    """让用户选择CSV文件并选择目标列"""
    root = Tk()
    root.withdraw()
    
    # 选择文件
    file_path = filedialog.askopenfilename(
        title="选择包含ECFP指纹和预测值的CSV文件",
        filetypes=[("CSV文件", "*.csv"), ("所有文件", "*.*")]
    )
    
    if not file_path:
        return None, None
    
    # 读取文件获取列名
    df = pd.read_csv(file_path)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # 选择目标列
    col_idx = simpledialog.askinteger(
        "选择目标列",
        f"可用的数值列(0-{len(numeric_cols)-1}):\n" + 
        "\n".join(f"{i}: {col}" for i, col in enumerate(numeric_cols)),
        minvalue=0, 
        maxvalue=len(numeric_cols)-1
    )
    
    root.destroy()
    
    if col_idx is None:
        return None, None
    
    return file_path, numeric_cols[col_idx]

def load_and_prepare_data(csv_path, target_col):
    """
    加载数据并准备特征和目标（改进版）
    新增功能：
    1. 严格的参数一致性检查
    2. 详细的调试输出
    3. 自动处理数据类型问题
    4. 验证bit位有效性
    """
    # 1. 加载数据并验证基本结构
    try:
        df = pd.read_csv(csv_path)
        print(f"\n[DEBUG] 成功加载文件: {os.path.basename(csv_path)}")
        print(f"[DEBUG] 数据维度: {df.shape}")
    except Exception as e:
        raise ValueError(f"无法读取CSV文件: {e}")

    # 2. 验证必要列存在
    required_cols = ['ECFP', target_col, 'smiles']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"缺少必要列: {missing_cols}")

    # 3. 转换ECFP列并清洗数据
    print("\n[DEBUG] 正在处理ECFP列...")
    
    def safe_convert_ecfp(ecfp_str):
        """安全转换ECFP字符串，处理异常值"""
        try:
            if pd.isna(ecfp_str):
                return None
            ecfp = eval(ecfp_str)
            if not isinstance(ecfp, (list, np.ndarray)):
                raise TypeError("ECFP格式不是列表/数组")
            return ecfp
        except Exception as e:
            print(f"警告: 无法转换ECFP字符串 '{ecfp_str[:50]}...' - {e}")
            return None

    df['ECFP'] = df['ECFP'].apply(safe_convert_ecfp)
    original_count = len(df)
    df = df.dropna(subset=['ECFP'])
    print(f"[DEBUG] 移除无效ECFP记录: {original_count} -> {len(df)}")

    # 4. 验证ECFP格式一致性
    ecfp_lengths = df['ECFP'].apply(len).unique()
    if len(ecfp_lengths) != 1:
        raise ValueError(f"ECFP长度不一致: 发现多种长度 {ecfp_lengths}")
    n_bits = ecfp_lengths[0]
    print(f"[DEBUG] 确认ECFP长度: {n_bits} bits")

    # 5. 提取特征矩阵
    print("\n[DEBUG] 构建特征矩阵...")
    X = np.array(df['ECFP'].tolist(), dtype=np.int8)
    
    # 6. 验证bit位范围
    active_bits = np.where(X.sum(axis=0) > 0)[0]
    print(f"[DEBUG] 数据集中激活的bit位: {len(active_bits)}/{n_bits}")
    print(f"[DEBUG] bit位ID范围: {min(active_bits)}-{max(active_bits)}")

    # 7. 处理目标列
    y = df[target_col].values
    print(f"\n[DEBUG] 目标列 '{target_col}' 统计:")
    print(f"  数据类型: {y.dtype}")
    print(f"  有效值范围: {y.min():.3f} - {y.max():.3f}")
    print(f"  缺失值数量: {pd.isna(y).sum()}")

    # 8. 最终数据验证
    assert len(X) == len(y) == len(df), "特征/目标/数据框长度不一致"
    print("\n[DEBUG] 数据准备完成")
    print(f"  最终样本数: {len(X)}")
    print(f"  特征维度: {X.shape[1]}")
    
    return X, y, df, target_col

def save_predictions(y_true, y_pred, smiles, filename, target_col):
    """保存预测结果到CSV"""
    pred_df = pd.DataFrame({
        'smiles': smiles,
        f'actual_{target_col}': y_true,
        f'predicted_{target_col}': y_pred,
        'residual': y_true - y_pred
    })
    pred_df.to_csv(filename, index=False)
    print(f"已保存预测结果到: {filename}")

def save_model_metrics(metrics_list, filename, target_col):
    """保存模型评估指标到CSV"""
    # 在列名中添加目标列信息
    renamed_metrics = []
    for metric in metrics_list:
        renamed_metric = {}
        for k, v in metric.items():
            if k in ['RMSE', 'MSE', 'R2']:
                renamed_metric[f"{k}_{target_col}"] = v
            else:
                renamed_metric[k] = v
        renamed_metrics.append(renamed_metric)
    
    metrics_df = pd.DataFrame(renamed_metrics)
    metrics_df.to_csv(filename, index=False)
    print(f"已保存模型评估指标到: {filename}")

def save_best_params(best_params, filename, target_col):
    """保存最佳超参数到CSV"""
    params_df = pd.DataFrame([best_params])
    params_df.to_csv(filename, index=False)
    print(f"已保存最佳超参数到: {filename}")

def train_xgboost_with_gridsearch(X, y, df, target_col, output_dir="model_results"):
    """使用网格搜索训练XGBoost模型并评估"""
    # 创建带目标列名的输出目录
    output_dir = os.path.join(output_dir, f"results_{target_col}")
    os.makedirs(output_dir, exist_ok=True)
    
    # 划分训练集和测试集 - 先获取索引
    train_idx, test_idx = train_test_split(
        np.arange(len(X)), test_size=0.1, random_state=42
    )
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # 获取对应的SMILES
    train_smiles = df.iloc[train_idx]['smiles'].values
    test_smiles = df.iloc[test_idx]['smiles'].values
    
    # 定义基础模型
    model = xgb.XGBRegressor(random_state=42)

    # 定义参数网格
    param_grid = {
        'n_estimators': [100],
        'max_depth': [6],
        'learning_rate': [0.1],
        'subsample': [0.6],
        'colsample_bytree': [0.6],
        'gamma': [0],
        'reg_alpha': [0],
        'reg_lambda': [0]
    }
    
    # 设置网格搜索
    print(f"\n开始网格搜索寻找最佳超参数(目标列: {target_col})...")
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )
    
    # 执行网格搜索
    grid_search.fit(X_train, y_train)
    
    # 获取最佳模型
    best_model = grid_search.best_estimator_
    
    # 输出最佳参数
    print(f"\n目标列 {target_col} 的最佳超参数组合:")
    for param, value in grid_search.best_params_.items():
        print(f"{param}: {value}")
    
    # 保存最佳超参数
    save_best_params(grid_search.best_params_, 
                    os.path.join(output_dir, f"best_hyperparameters_{target_col}.csv"),
                    target_col)
    
    # 评估训练集和测试集
    def evaluate_model(model, X, y, dataset_name, smiles, target_col):
        y_pred = model.predict(X)
        mse = mean_squared_error(y, y_pred)
        rmse = sqrt(mse)
        r2 = r2_score(y, y_pred)
        
        # 保存预测结果
        save_predictions(y, y_pred, smiles, 
                        os.path.join(output_dir, f"{dataset_name}_predictions_{target_col}.csv"),
                        target_col)
        
        metrics = {
            'dataset': dataset_name,
            'RMSE': rmse,
            'MSE': mse,
            'R2': r2,
            'n_samples': len(y),
            'target_column': target_col
        }
        
        print(f"\n{dataset_name}性能(目标列 {target_col}):")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MSE: {mse:.4f}")
        print(f"  R²: {r2:.4f}")
        return y_pred, metrics
    
    # 评估训练集
    y_train_pred, train_metrics = evaluate_model(best_model, X_train, y_train, "train", train_smiles, target_col)
    # 评估测试集
    y_test_pred, test_metrics = evaluate_model(best_model, X_test, y_test, "test", test_smiles, target_col)
    
    # 保存模型评估指标
    save_model_metrics([train_metrics, test_metrics], 
                      os.path.join(output_dir, f"model_metrics_{target_col}.csv"),
                      target_col)
    
    # 绘制预测值与实际值的散点图
    plot_pred_vs_actual(y_test, y_test_pred, f"测试集({target_col})")
    plot_pred_vs_actual(y_train, y_train_pred, f"训练集({target_col})")
    
    return best_model, X_train, X_test, y_train, y_test, y_train_pred, y_test_pred, target_col

def plot_pred_vs_actual(y_true, y_pred, dataset_name):
    """绘制预测值与实际值的散点图"""
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.6)
    
    # 添加对角线
    max_val = max(max(y_true), max(y_pred))
    min_val = min(min(y_true), min(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.xlabel(f'Actual {dataset_name.split("(")[-1].rstrip(")")}')
    plt.ylabel(f'Predicted {dataset_name.split("(")[-1].rstrip(")")}')
    plt.title(f'Predicted vs Actual Values ({dataset_name})')
    
    # 添加R²和MSE到图中
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    plt.text(0.05, 0.95, f'R² = {r2:.3f}\nMSE = {mse:.3f}', 
             transform=plt.gca().transAxes,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show()

def shap_analysis(model, X, df, target_col, output_dir="shap_results"):
    """执行SHAP分析并可视化结果"""
    # 创建带目标列名的输出目录
    output_dir = os.path.join(output_dir, f"shap_{target_col}")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n执行SHAP分析(目标列: {target_col})...")
    
    # 计算SHAP值
    explainer = shap.Explainer(model)
    shap_values = explainer(X)

    # 1. 计算全局特征重要性
    global_shap_importance = pd.DataFrame({
        'Bit': [f"Bit_{i}" for i in range(shap_values.values.shape[1])],
        f'Importance_{target_col}': np.mean(np.abs(shap_values.values), axis=0)
    })

    # 按重要性降序排序
    global_shap_importance = global_shap_importance.sort_values(f'Importance_{target_col}', ascending=False)

    # 保存全局特征重要性
    global_shap_importance.to_csv(
        os.path.join(output_dir, f"global_shap_importance_{target_col}.csv"), 
        index=False
    )
    print(f"已保存全局特征重要性到: {os.path.join(output_dir, f'global_shap_importance_{target_col}.csv')}")
    
    # 2. 保存每个分子最重要的前10个位
    top_bits_df = pd.DataFrame()
    top_bits_data = []
    
    for i in range(len(df)):
        abs_shap = np.abs(shap_values.values[i])
        top_indices = np.argsort(-abs_shap)[:10]
        top_bits_data.append({
            "smiles": df.iloc[i]["smiles"],
            f"actual_{target_col}": df.iloc[i][target_col],
            **{f"top_{j+1}_bit": top_indices[j] for j in range(10)},
            **{f"top_{j+1}_shap_{target_col}": shap_values.values[i][top_indices[j]] for j in range(10)}
        })
    
    top_bits_df = pd.DataFrame(top_bits_data)
    top_bits_df.to_csv(
        os.path.join(output_dir, f"top_10_bits_per_molecule_{target_col}.csv"), 
        index=False
    )
    print(f"已保存每个分子最重要的前10个位到: {os.path.join(output_dir, f'top_10_bits_per_molecule_{target_col}.csv')}")
    
    # 3. 总体特征重要性
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X, plot_type="bar")
    plt.title(f"SHAP特征重要性(目标列: {target_col})")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"shap_feature_importance_{target_col}.png"))
    plt.show()
    
    # 4. SHAP值分布图
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X)
    plt.title(f"SHAP值分布(目标列: {target_col})")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"shap_summary_plot_{target_col}.png"))
    plt.show()

def analyze_with_shap():
    # 让用户选择CSV文件和目标列
    csv_path, target_col = select_csv_file_and_target()
    if not csv_path or not target_col:
        print("未选择文件或目标列，程序退出")
        return
    
    # 加载和准备数据
    X, y, df, target_col = load_and_prepare_data(csv_path, target_col)
    
    # 训练XGBoost模型(带网格搜索)
    model, X_train, X_test, y_train, y_test, y_train_pred, y_test_pred, target_col = train_xgboost_with_gridsearch(X, y, df, target_col)
    
    # SHAP分析
    shap_analysis(model, X, df, target_col)

if __name__ == "__main__":
    analyze_with_shap()