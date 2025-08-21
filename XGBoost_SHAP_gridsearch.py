import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap
import sys
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

# 设置字体为 Arial，并支持负号显示
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False

# ----------------------------
# 1. 用户配置部分
# ----------------------------
CSV_FILE = 'D:\\PROJECT_4\\分子计算信息汇总\\MOL_merged.csv'  
TARGET_COL = 54           
FEATURE_START_COL = 1     
FEATURE_END_COL = 50       
# 其他配置
TEST_SIZE = 0.1
RANDOM_STATE = 42
MAX_SHAP_SAMPLES = 200        # 控制用于 SHAP 分析的最大样本数

# ----------------------------
# 2. 初始化环境函数
# ----------------------------
def initialize_environment():
    """Initialize visualization settings"""
    sns.set()
    sns.set_palette("husl")
    pd.set_option('display.max_columns', 50)
    print("Environment initialized")

# ----------------------------
# 3. 数据加载函数（处理两行标题）
# ----------------------------
def load_and_explore_data():
    """Load special CSV file with two header rows"""
    try:
        encodings = ['utf-8', 'gbk', 'gb18030', 'ansi']
        data = None
        for encoding in encodings:
            try:
                with open(CSV_FILE, 'r', encoding=encoding) as f:
                    next(f)  # Skip first line
                    header_line = next(f).strip()
                    column_names = header_line.split(',')
                data = pd.read_csv(CSV_FILE, skiprows=2, header=None,
                                   names=column_names, encoding=encoding)
                print(f"\nSuccessfully loaded data using {encoding} encoding")
                break
            except UnicodeDecodeError:
                continue
            except Exception as e:
                print(f"Error with {encoding} encoding: {str(e)}")
                continue
        if data is None:
            raise ValueError("Failed to read file with any common encoding. Please check manually.")
        print(f"\nData loaded successfully: {data.shape[0]} rows, {data.shape[1]} columns")
        print(f"Used column names: {column_names}")
    except Exception as e:
        print(f"\nData loading error: {str(e)}")
        print("Suggested solutions:")
        print("1. Open the CSV in Notepad and save as UTF-8 format")
        print("2. Manually specify correct encoding (e.g., gbk) in code")
        sys.exit(1)

    # 确认列范围是否正确
    all_columns = data.columns.tolist()
    if FEATURE_END_COL >= len(all_columns):
        raise IndexError("FEATURE_END_COL 超出实际列数，请检查列索引是否正确")

    # 提取特征列和目标列
    feature_columns = all_columns[FEATURE_START_COL : FEATURE_END_COL + 1]
    target_column = all_columns[TARGET_COL]

    # 确保 PCE 列存在并加入分析
    if 'PCE' not in data.columns:
        raise KeyError("Column 'PCE' not found in dataset")
    
    features_with_target = feature_columns + ['PCE']

    print("\nColumn usage:")
    print(f"- Target variable: {target_column} (Index {TARGET_COL})")
    print(f"- Feature variables: {list(feature_columns)} (Index {FEATURE_START_COL} to {FEATURE_END_COL})")
    ignored = set(data.columns) - set(feature_columns) - {target_column, 'PCE'}
    print(f"- Ignored columns: {list(ignored) if ignored else 'None'}")

    print("\nDescriptive statistics of features:")
    print(data[feature_columns].describe())

    # 目标变量分布图
    plt.figure(figsize=(10, 5))
    sns.histplot(data[target_column], kde=True)
    plt.title(f'Distribution of Target Variable: {target_column}')
    plt.tight_layout()
    plt.show()

    # 绘制相关性热图（包括 PCE）
    plt.figure(figsize=(30, 30))
    corr_matrix_with_pce = data[features_with_target].corr()
    sns.heatmap(corr_matrix_with_pce, annot=False, cmap='coolwarm', center=0, fmt='.2f')
    plt.title('Feature Correlation Matrix Including PCE')
    plt.xticks(fontsize=17)
    plt.yticks(fontsize=17)
    plt.tight_layout()
    plt.show()
    # 保存相关系数矩阵到 CSV 文件
    corr_output_path = 'feature_correlation_matrix.csv'
    corr_matrix_with_pce.to_csv(corr_output_path, index=True, header=True)

    print(f"\n相关系数矩阵已保存至：{corr_output_path}")

    return data, feature_columns, target_column
# ----------------------------
# 4. 数据预处理函数
# ----------------------------
def preprocess_data(data, feature_columns, target_column):
    X = data[feature_columns]
    y = data[target_column]
    print("\nMissing value check:")
    print(X.isnull().sum())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    train_means = X_train.mean()
    X_train = X_train.fillna(train_means)
    X_test = X_test.fillna(train_means)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_scaled, X_val_scaled, y_train, y_val = train_test_split(
        X_train_scaled, y_train, test_size=0.1, random_state=RANDOM_STATE
    )

    return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test, scaler, feature_columns
# ----------------------------
# 5. 模型训练和调优函数
# ----------------------------
'''def train_and_tune_model(X_train, y_train, X_val, y_val):
    # 定义搜索空间
    param_space = {
        'n_estimators': Integer(200, 800),
        'max_depth': Integer(1, 10),
        'learning_rate': Real(0.001, 0.3, prior='log-uniform'),
        'subsample': Real(0.1, 1.0),
        'colsample_bytree': Real(0.1, 1.0),
        'gamma': Real(0.0, 1.0),
        'reg_alpha': Real(0.001, 0.1, prior='log-uniform'),
        'reg_lambda': Real(0.001, 0.1, prior='log-uniform')
    }

    model = XGBRegressor(
        objective='reg:squarederror',
        random_state=RANDOM_STATE,
        early_stopping_rounds=10,
        eval_metric=['rmse', 'mae']
    )

    opt = BayesSearchCV(
        estimator=model,
        search_spaces=param_space,
        scoring='neg_mean_squared_error',
        n_iter=60,
        cv=5,
        n_jobs=-1,
        verbose=2,
        random_state=42
    )

    print("\nStarting Bayesian optimization...")
    opt.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    print("\nOptimization completed")
    print("Best parameters:", opt.best_params_)
    print("Best CV score (negative MSE):", opt.best_score_)

    return opt.best_estimator_ '''

def train_and_tune_model(X_train, y_train, X_val, y_val):
   # os.makedirs('catboost_temp', exist_ok=True)
    param_grid = {
        'n_estimators': [300],
        'max_depth': [6],
        'learning_rate': [0.06],
        'subsample': [0.6],
        'colsample_bytree': [0.5],
        'gamma': [0.2],
        'reg_alpha': [0.1],
        'reg_lambda': [0.1]
    }

    model = XGBRegressor(
        objective='reg:squarederror',  
        random_state=RANDOM_STATE,     
        early_stopping_rounds=10,
        eval_metric=['rmse', 'mae'],    
        verbosity=0                     
    )

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring=['neg_mean_squared_error', 'r2'],
        refit='neg_mean_squared_error',
        cv=5,
        n_jobs=-1,
        verbose=2
    )

    print("\nStarting grid search...")
    grid_search.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    print("\nGrid search completed")
    print("Best parameters:", grid_search.best_params_)
    print("Best CV score (negative MSE):", grid_search.best_score_)

    return grid_search.best_estimator_
# ----------------------------
# 6. 模型评估函数（支持训练集 & 测试集）
# ----------------------------
def evaluate_model(model, X_train, y_train, X_test, y_test, feature_names):
    result = {}

    def get_metrics(y_true, y_pred, prefix=""):
        # 计算标准指标
        metrics = {
            f'{prefix}_MSE': mean_squared_error(y_true, y_pred),
            f'{prefix}_RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            f'{prefix}_MAE': mean_absolute_error(y_true, y_pred),
            f'{prefix}_R²': r2_score(y_true, y_pred),
            f'{prefix}_Explained Variance': explained_variance_score(y_true, y_pred),
            f'{prefix}_Pearson r': pearsonr(y_true, y_pred)[0],
            # 新增预测分数指标
            f'{prefix}_Prediction Score (1-MSE)': 1 - mean_squared_error(y_true, y_pred),
            f'{prefix}_Prediction Score (1-RMSE)': 1 - np.sqrt(mean_squared_error(y_true, y_pred)),
            f'{prefix}_Prediction Score (R²)': r2_score(y_true, y_pred)
        }
        return metrics

    # Train set evaluation
    y_train_pred = model.predict(X_train)
    train_metrics = get_metrics(y_train, y_train_pred, "Train")
    result.update(train_metrics)

    # Test set evaluation
    y_test_pred = model.predict(X_test)
    test_metrics = get_metrics(y_test, y_test_pred, "Test")
    result.update(test_metrics)

    # 格式化输出结果
    print("\n【Training Set】Model Evaluation Metrics:")
    print("{:<30} {:<15}".format("Metric", "Value"))
    print("-" * 45)
    for name, value in train_metrics.items():
        if "Score" in name:
            print("{:<30} {:<15.4f} (Score)".format(name, value))
        else:
            print("{:<30} {:<15.4f}".format(name, value))

    print("\n【Test Set】Model Evaluation Metrics:")
    print("{:<30} {:<15}".format("Metric", "Value"))
    print("-" * 45)
    for name, value in test_metrics.items():
        if "Score" in name:
            print("{:<30} {:<15.4f} (Score)".format(name, value))
        else:
            print("{:<30} {:<15.4f}".format(name, value))

    # 保存训练集预测结果到CSV
    train_results = pd.DataFrame({
        'Actual': y_train,
        'Predicted': y_train_pred,
        'Residual': y_train - y_train_pred
    })
    train_results.to_csv('train_predictions.csv', index=False)
    print("\n训练集预测结果已保存到 train_predictions.csv")

    # 保存测试集预测结果到CSV
    test_results = pd.DataFrame({
        'Actual': y_test,
        'Predicted': y_test_pred,
        'Residual': y_test - y_test_pred
    })
    test_results.to_csv('test_predictions.csv', index=False)
    print("测试集预测结果已保存到 test_predictions.csv")

    # 可视化部分保持不变...
    # Combined Actual vs Predicted Plot (Training and Test sets)
    plt.figure(figsize=(10, 8))
    
    # Plot training set
    sns.scatterplot(x=y_train, y=y_train_pred, alpha=0.7, 
                    color='blue', label='Training Set')
    
    # Plot test set
    sns.scatterplot(x=y_test, y=y_test_pred, alpha=0.7, 
                    color='red', label='Test Set')
    
    # Plot ideal fit line
    all_values = np.concatenate([y_train, y_test])
    plt.plot([all_values.min(), all_values.max()], 
             [all_values.min(), all_values.max()], 
             'k--', label='Ideal Fit Line')
    
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs Predicted Values (Training and Test Sets)')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Combined Residual Plot (Training and Test sets)
    residuals_train = y_train - y_train_pred
    residuals_test = y_test - y_test_pred
    
    plt.figure(figsize=(10, 8))
    
    # Plot training set residuals
    sns.scatterplot(x=y_train_pred, y=residuals_train, alpha=0.7, 
                    color='blue', label='Training Set')
    
    # Plot test set residuals
    sns.scatterplot(x=y_test_pred, y=residuals_test, alpha=0.7, 
                    color='red', label='Test Set')
    
    plt.axhline(y=0, color='k', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residual Plot (Training and Test Sets)')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Feature Importance Plot
    feature_imp = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)

    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_imp)
    plt.title('XGBoost Feature Importance')
    plt.tight_layout()
    plt.show()

    return result

# ----------------------------
# 7. SHAP分析函数（增强版）
# ----------------------------
def perform_shap_analysis(model, X, feature_names):
    try:
        explainer = shap.TreeExplainer(model)
        
        # 确保X是二维数组
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        
        # 计算SHAP值
        shap_values = explainer(X)
        
        # 确保shap_values存在且包含feature_names
        if shap_values is None:
            raise ValueError("无法计算SHAP值，返回值为None")
            
        if not hasattr(shap_values, 'feature_names'):
            shap_values.feature_names = list(feature_names)
        elif shap_values.feature_names is None or len(shap_values.feature_names) == 0:
            shap_values.feature_names = list(feature_names)
        
        # 计算交互值
        try:
            shap_interaction_values = explainer.shap_interaction_values(X)
        except Exception as e:
            print(f"无法计算SHAP交互值: {str(e)}")
            shap_interaction_values = None

        print(f"\nGenerating SHAP plots (based on {X.shape[0]} samples)...")

        # SHAP Summary Bar Plot
        try:
            plt.figure()
            shap.summary_plot(shap_values.values, X, feature_names=feature_names, plot_type="bar", show=False, max_display=48)
            plt.title('SHAP Feature Importance (Bar Chart)')
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"无法生成SHAP条形图: {str(e)}")

        # SHAP Summary Dot Plot
        try:
            plt.figure()
            shap.summary_plot(shap_values.values, X, feature_names=feature_names, show=False, max_display=48)
            plt.title('SHAP Feature Importance (Dot Chart)')
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"无法生成SHAP点图: {str(e)}")

        # SHAP Heatmap Plot
        try:
            plt.figure(figsize=(12, 12))
            shap.plots.heatmap(shap_values, show=False, max_display=48)
            plt.title('SHAP Values Heatmap')
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"无法生成SHAP热图: {str(e)}")
            # 备选方案
            try:
                plt.figure(figsize=(12, 8))
                sns.heatmap(shap_values.values, cmap='viridis')
                plt.title('SHAP Values (Alternative)')
                plt.tight_layout()
                plt.show()
            except:
                print("也无法生成备选热图")


        # SHAP Dependence Plots (Top 3 Features)
        if len(feature_names) > 0:
            for i in range(min(10, len(feature_names))):
                try:
                    plt.figure()
                    shap.dependence_plot(i, shap_values.values, X, feature_names=feature_names, show=False)
                    plt.title(f'SHAP Dependence Plot - {feature_names[i]}')
                    plt.tight_layout()
                    plt.show()
                except Exception as e:
                    print(f"无法生成依赖图 {feature_names[i]}: {str(e)}")

        # SHAP Interaction Heatmap
        if shap_interaction_values is not None and len(feature_names) > 1:
            try:
                plt.figure(figsize=(12, 10))
                shap_interaction_avg = np.abs(shap_interaction_values).mean(0)
                sns.heatmap(shap_interaction_avg, annot=False, cmap='viridis',
                            xticklabels=feature_names, yticklabels=feature_names, max_display=48)
                plt.title('SHAP Interaction Heatmap')
                plt.tight_layout()
                plt.show()
            except Exception as e:
                print(f"无法生成交互热图: {str(e)}")

        # SHAP Waterfall Plot
        try:
            plt.figure(figsize=(10, 6))
            shap.plots.waterfall(shap_values[0], max_display=48, show=False)
            plt.title('SHAP Waterfall Plot (One Sample)')
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"无法生成瀑布图: {str(e)}")

        # SHAP Force Plots
        shap.initjs()
        print("\nGenerating force plot for a single sample...")
        try:
            force_plot_single = shap.force_plot(
                explainer.expected_value,
                shap_values.values[0, :],
                pd.DataFrame(X[0:1], columns=feature_names),
                feature_names=feature_names
            )
            shap.save_html("shap_force_plot_single.html", force_plot_single)
        except Exception as e:
            print(f"无法生成单样本力导向图: {str(e)}")

        print("Generating force plot for multiple samples...")
        try:
            force_plot_multi = shap.force_plot(
                explainer.expected_value,
                shap_values.values[:20],
                pd.DataFrame(X[:20], columns=feature_names),
                feature_names=feature_names
            )
            shap.save_html("shap_force_plot_multi.html", force_plot_multi)
        except Exception as e:
            print(f"无法生成多样本力导向图: {str(e)}")

        # 保存SHAP值
        try:
            np.save('shap_values.npy', shap_values.values)
            print("\nSHAP values saved to shap_values.npy")
            if shap_interaction_values is not None:
                np.save('shap_interaction_values.npy', shap_interaction_values)
                print("SHAP interaction values saved to shap_interaction_values.npy")
        except Exception as e:
            print(f"无法保存SHAP值: {str(e)}")

        print("Interactive Force Plots saved as shap_force_plot_*.html")

    except Exception as e:
        print(f"\nSHAP分析过程中发生严重错误: {str(e)}")
        print("建议检查:")
        print("1. 确认模型和输入数据格式正确")
        print("2. 尝试减少样本数量")
        print("3. 检查SHAP版本是否最新")
# ----------------------------
# 主程序流程
# ----------------------------
def main():
    initialize_environment()
    data, feature_columns, target_column = load_and_explore_data()
    (X_train, X_val, X_test,
     y_train, y_val, y_test,
     scaler, feature_names) = preprocess_data(data, feature_columns, target_column)
    best_model = train_and_tune_model(X_train, y_train, X_val, y_val)
    metrics = evaluate_model(best_model, X_train, y_train, X_test, y_test, feature_names)

    print("\n=== Starting SHAP Analysis using Training Set ===")
    print(f"Total training samples: {X_train.shape[0]}")

    if MAX_SHAP_SAMPLES > 0 and MAX_SHAP_SAMPLES < X_train.shape[0]:
        sample_idx = np.random.choice(X_train.shape[0], size=MAX_SHAP_SAMPLES, replace=False)
        X_shap = X_train[sample_idx]
    else:
        X_shap = X_train

    print(f"Performing SHAP analysis on {X_shap.shape[0]} samples...")
    perform_shap_analysis(best_model, X_shap, feature_names)

    joblib.dump(best_model, 'xgb_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    pd.DataFrame([metrics]).to_csv('model_metrics.csv', index=False)
    print("\nAll processing completed! Model and results have been saved.")

if __name__ == "__main__":
    main()
