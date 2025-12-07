"""
完整資料前處理 - 強化版
新增功能：
1. 每個步驟後檢查NaN
2. 自動填補NaN值
3. 詳細報告資料品質
"""

import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("完整資料前處理 (強化版 - 含NaN檢查)".center(80))
print("="*80)

# ============================================================================
# 1. 載入原始資料
# ============================================================================
print("\n[步驟1] 載入原始資料")
df = pd.read_csv("C:/Users/fiona/Downloads/corporate_work_hours_productivity (1).csv")
print(f"✓ 原始資料: {len(df)}筆, {df.shape[1]}欄")

# 檢查原始資料的NaN
print("\n[步驟1.1] 檢查原始資料NaN")
nan_count = df.isnull().sum()
if nan_count.sum() > 0:
    print("⚠️ 發現原始資料有NaN:")
    for col in nan_count[nan_count > 0].index:
        print(f"  {col}: {nan_count[col]}個NaN")
    
    # 填補數值型NaN
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            print(f"  → {col} 用中位數 {median_val:.2f} 填補")
    
    # 填補類別型NaN
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df[col].isnull().any():
            mode_val = df[col].mode()[0]
            df[col].fillna(mode_val, inplace=True)
            print(f"  → {col} 用眾數 {mode_val} 填補")
else:
    print("✓ 原始資料無NaN")

# ============================================================================
# 2. 創建目標變數
# ============================================================================
print("\n[步驟2] 創建目標變數")
threshold = df['Productivity_Score'].quantile(0.67)
print(f"✓ 67百分位數: {threshold:.2f}")

df['Productivity_Category'] = df['Productivity_Score'].apply(
    lambda x: 'High' if x >= threshold else 'Not_High'
)

print(f"✓ 目標分布:")
print(f"  Not_High: {(df['Productivity_Category']=='Not_High').sum()} "
      f"({(df['Productivity_Category']=='Not_High').sum()/len(df)*100:.1f}%)")
print(f"  High: {(df['Productivity_Category']=='High').sum()} "
      f"({(df['Productivity_Category']=='High').sum()/len(df)*100:.1f}%)")

# ============================================================================
# 3. 特徵工程
# ============================================================================
print("\n[步驟3] 特徵工程 - 創建衍生特徵")

# 3.1 效率指標
print("\n3.1 效率指標")
df['Tasks_Per_Hour'] = df['Tasks_Completed_Per_Day'] / (df['Monthly_Hours_Worked'] / 22 + 0.001)
df['Tasks_Per_Meeting'] = df['Tasks_Completed_Per_Day'] / (df['Meetings_per_Week'] + 1)
df['Overtime_Efficiency'] = df['Tasks_Completed_Per_Day'] / (df['Overtime_Hours_Per_Week'] + 1)
print("  ✓ Tasks_Per_Hour, Tasks_Per_Meeting, Overtime_Efficiency")

# 3.2 工作負荷指標
print("\n3.2 工作負荷指標")
df['Total_Work_Hours'] = df['Monthly_Hours_Worked'] + df['Overtime_Hours_Per_Week'] * 4
df['Meeting_Burden'] = df['Meetings_per_Week'] / (df['Monthly_Hours_Worked'] / 22 / 8 + 0.001)
df['Overtime_Ratio'] = df['Overtime_Hours_Per_Week'] / (df['Monthly_Hours_Worked'] / 4 + 1)
print("  ✓ Total_Work_Hours, Meeting_Burden, Overtime_Ratio")

# 3.3 經驗資歷指標
print("\n3.3 經驗資歷指標")
job_level_map = {'Entry': 1, 'Mid': 2, 'Senior': 3, 'Manager': 4}
df['Job_Level_Numeric'] = df['Job_Level'].map(job_level_map)
df['Age_Experience_Ratio'] = df['Age'] / (df['Years_at_Company'] + 1)
df['Seniority_Score'] = df['Years_at_Company'] * df['Job_Level_Numeric']
print("  ✓ Job_Level_Numeric, Age_Experience_Ratio, Seniority_Score")

# 3.4 滿意度指標
print("\n3.4 滿意度指標")
work_life_map = {'Poor': 1, 'Fair': 2, 'Good': 3, 'Excellent': 4}
df['Work_Life_Score'] = df['Work_Life_Balance'].map(work_life_map)
df['Satisfaction_Balance'] = df['Job_Satisfaction'] * df['Work_Life_Score']
print("  ✓ Work_Life_Score, Satisfaction_Balance")

print(f"\n✓ 新增衍生特徵: 10個")

# ============================================================================
# 3.5 檢查特徵工程後的NaN和inf
# ============================================================================
print("\n[步驟3.5] 檢查特徵工程後的資料品質")

new_features = ['Tasks_Per_Hour', 'Tasks_Per_Meeting', 'Overtime_Efficiency',
                'Total_Work_Hours', 'Meeting_Burden', 'Overtime_Ratio',
                'Age_Experience_Ratio', 'Seniority_Score', 
                'Work_Life_Score', 'Satisfaction_Balance', 'Job_Level_Numeric']

# 檢查NaN
nan_found = False
for col in new_features:
    nan_count = df[col].isnull().sum()
    if nan_count > 0:
        nan_found = True
        print(f"⚠️ {col}: {nan_count}個NaN")
        # 用中位數填補
        median_val = df[col].median()
        df[col].fillna(median_val, inplace=True)
        print(f"  → 已用中位數 {median_val:.4f} 填補")

if not nan_found:
    print("✓ 無NaN")

# 檢查inf
inf_found = False
for col in new_features:
    inf_count = np.isinf(df[col]).sum()
    if inf_count > 0:
        inf_found = True
        print(f"⚠️ {col}: {inf_count}個inf")
        # 將inf替換為該欄位的最大有限值
        max_finite = df[col].replace([np.inf, -np.inf], np.nan).max()
        df[col].replace([np.inf, -np.inf], max_finite, inplace=True)
        print(f"  → 已用最大有限值 {max_finite:.4f} 替換")

if not inf_found:
    print("✓ 無inf")

# ============================================================================
# 4. 相關性分析
# ============================================================================
print("\n[步驟4] 新特徵相關性分析")

y_numeric = df['Productivity_Category'].map({'Not_High': 0, 'High': 1})

print("\n新特徵與High類別的相關性:")
correlations = {}
for col in new_features:
    corr = df[col].corr(y_numeric)
    correlations[col] = corr
    print(f"  {col:30s}: {corr:7.4f}")

max_corr_feature = max(correlations, key=lambda x: abs(correlations[x]))
print(f"\n✓ 最強相關特徵: {max_corr_feature} ({correlations[max_corr_feature]:.4f})")

# ============================================================================
# 5. 異常值處理
# ============================================================================
print("\n[步驟5] 異常值處理")

def handle_outliers_iqr(data, columns):
    df_clean = data.copy()
    outlier_counts = {}
    
    for col in columns:
        # 跳過NaN
        valid_data = df_clean[col].dropna()
        
        Q1 = valid_data.quantile(0.25)
        Q3 = valid_data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = ((df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)).sum()
        if outliers > 0:
            outlier_counts[col] = outliers
        
        df_clean[col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound)
    
    return df_clean, outlier_counts

numerical_to_process = ['Age', 'Years_at_Company'] + [f for f in new_features if f != 'Job_Level_Numeric']

df, outlier_info = handle_outliers_iqr(df, numerical_to_process)

if outlier_info:
    print("異常值數量:")
    for col, count in outlier_info.items():
        print(f"  {col}: {count}")
print("✓ 異常值處理完成")

# ============================================================================
# 6. 類別變數編碼
# ============================================================================
print("\n[步驟6] 類別變數編碼")

print("\n6.1 One-Hot編碼:")

# Department
dept_dummies = pd.get_dummies(df['Department'], prefix='Dept', drop_first=True)
df = pd.concat([df, dept_dummies], axis=1)
print(f"  ✓ Department → {list(dept_dummies.columns)}")

# Remote_Work
remote_dummies = pd.get_dummies(df['Remote_Work'], prefix='Remote', drop_first=True)
df = pd.concat([df, remote_dummies], axis=1)
print(f"  ✓ Remote_Work → {list(remote_dummies.columns)}")

# Work_Life_Balance
wlb_dummies = pd.get_dummies(df['Work_Life_Balance'], prefix='WLB', drop_first=True)
df = pd.concat([df, wlb_dummies], axis=1)
print(f"  ✓ Work_Life_Balance → {list(wlb_dummies.columns)}")

print("\n6.2 Label編碼:")
print(f"  ✓ Job_Level → Job_Level_Numeric (Entry=1, Mid=2, Senior=3, Manager=4)")

# ============================================================================
# 7. 移除不需要的欄位
# ============================================================================
print("\n[步驟7] 移除不需要的欄位")

columns_to_remove = [
    # 識別欄位
    'Employee_ID',
    
    # 目標相關
    'Productivity_Score',
    
    # 結果變數
    'Annual_Salary',
    'Absences_Per_Year',
    
    # 已轉為衍生特徵的原始欄位
    'Tasks_Completed_Per_Day',
    'Monthly_Hours_Worked',
    'Meetings_per_Week',
    'Overtime_Hours_Per_Week',
    
    # 原始類別欄位 (已One-Hot編碼)
    'Department',
    'Remote_Work',
    'Work_Life_Balance',
    'Job_Level',  # 已有 Job_Level_Numeric
]

print(f"要移除的欄位: {columns_to_remove}")

# 執行移除
removed = []
for col in columns_to_remove:
    if col in df.columns:
        df = df.drop(col, axis=1)
        removed.append(col)

print(f"✓ 已移除 {len(removed)} 個欄位")

# ============================================================================
# 8. 準備最終特徵
# ============================================================================
print("\n[步驟8] 準備最終特徵")

# 從df中分離X和y
X = df.drop('Productivity_Category', axis=1).copy()
y = df['Productivity_Category'].copy()

print(f"✓ 最終特徵數: {X.shape[1]}")
print(f"✓ 樣本數: {X.shape[0]}")

# ============================================================================
# 8.1 最終資料品質檢查
# ============================================================================
print("\n[步驟8.1] 最終資料品質檢查")

# 檢查資料類型
print(f"\n資料類型:")
print(f"  數值型: {(X.dtypes != 'object').sum()}")
print(f"  物件型: {(X.dtypes == 'object').sum()}")

if (X.dtypes == 'object').any():
    print("\n⚠️ 警告: 發現非數值欄位!")
    non_numeric = X.columns[X.dtypes == 'object'].tolist()
    print(f"  {non_numeric}")
else:
    print("✓ 所有特徵都是數值型")

# 檢查NaN
print(f"\nNaN檢查:")
nan_counts = X.isnull().sum()
total_nan = nan_counts.sum()
if total_nan > 0:
    print(f"⚠️ 發現 {total_nan} 個NaN:")
    for col in nan_counts[nan_counts > 0].index:
        print(f"  {col}: {nan_counts[col]}")
    
    print("\n正在處理NaN...")
    # 用中位數填補所有NaN
    for col in X.columns:
        if X[col].isnull().any():
            median_val = X[col].median()
            X[col].fillna(median_val, inplace=True)
            print(f"  → {col} 已填補")
    print("✓ NaN處理完成")
else:
    print("✓ 無NaN")

# 檢查inf
print(f"\ninf檢查:")
inf_counts = np.isinf(X).sum()
total_inf = inf_counts.sum()
if total_inf > 0:
    print(f"⚠️ 發現 {total_inf} 個inf:")
    for col in inf_counts[inf_counts > 0].index:
        print(f"  {col}: {inf_counts[col]}")
    
    print("\n正在處理inf...")
    for col in X.columns:
        if np.isinf(X[col]).any():
            max_finite = X[col].replace([np.inf, -np.inf], np.nan).max()
            X[col].replace([np.inf, -np.inf], max_finite, inplace=True)
            print(f"  → {col} 已替換為 {max_finite:.4f}")
    print("✓ inf處理完成")
else:
    print("✓ 無inf")

# 統計資料範圍
print(f"\n資料範圍檢查:")
print(f"  最小值的最小值: {X.min().min():.4f}")
print(f"  最大值的最大值: {X.max().max():.4f}")

print(f"\n完整特徵列表:")
for i, col in enumerate(X.columns, 1):
    nan_in_col = X[col].isnull().sum()
    inf_in_col = np.isinf(X[col]).sum()
    status = "✓" if (nan_in_col == 0 and inf_in_col == 0) else "⚠️"
    print(f"  {status} {i:2d}. {col:30s} (dtype: {X[col].dtype}, min: {X[col].min():.2f}, max: {X[col].max():.2f})")

# ============================================================================
# 9. 儲存
# ============================================================================
print("\n[步驟9] 儲存處理好的資料")

X.to_csv('X_processed.csv', index=False, encoding='utf-8-sig')
y.to_csv('y_processed.csv', index=False, encoding='utf-8-sig', header=['Productivity_Category'])

data_dict = {
    'X': X,
    'y': y,
    'feature_names': list(X.columns),
    'threshold': threshold,
    'class_counts': y.value_counts().to_dict(),
    'feature_correlations': correlations,
    'removed_columns': removed
}

with open('processed_data.pkl', 'wb') as f:
    pickle.dump(data_dict, f)

print("✓ X_processed.csv")
print("✓ y_processed.csv")
print("✓ processed_data.pkl")

# 儲存摘要
with open('preprocessing_summary.txt', 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("完整資料前處理總結 (強化版)\n")
    f.write("="*80 + "\n\n")
    
    f.write(f"原始資料: {len(df)}筆\n")
    f.write(f"67百分位數: {threshold:.2f}\n\n")
    
    f.write("目標變數分布:\n")
    f.write(f"  Not_High: {(y=='Not_High').sum()} ({(y=='Not_High').sum()/len(y)*100:.1f}%)\n")
    f.write(f"  High: {(y=='High').sum()} ({(y=='High').sum()/len(y)*100:.1f}%)\n\n")
    
    f.write("="*80 + "\n")
    f.write("資料品質檢查\n")
    f.write("="*80 + "\n\n")
    
    f.write(f"NaN總數: {X.isnull().sum().sum()}\n")
    f.write(f"inf總數: {np.isinf(X).sum().sum()}\n")
    f.write(f"所有特徵都是數值型: {(X.dtypes != 'object').all()}\n\n")
    
    f.write("="*80 + "\n")
    f.write("特徵工程\n")
    f.write("="*80 + "\n\n")
    
    f.write("新增衍生特徵:\n")
    for i, feat in enumerate(new_features, 1):
        f.write(f"  {i}. {feat}\n")
    
    f.write("\n新特徵相關性:\n")
    for feat in sorted(correlations, key=lambda x: abs(correlations[x]), reverse=True):
        f.write(f"  {feat:30s}: {correlations[feat]:7.4f}\n")
    
    f.write("\n="*80 + "\n")
    f.write("最終特徵\n")
    f.write("="*80 + "\n\n")
    
    f.write(f"特徵數: {X.shape[1]}\n\n")
    f.write("完整特徵列表:\n")
    for i, col in enumerate(X.columns, 1):
        f.write(f"  {i}. {col}\n")
    
    f.write("\n="*80 + "\n")
    f.write("移除的欄位\n")
    f.write("="*80 + "\n\n")
    for i, col in enumerate(removed, 1):
        f.write(f"  {i}. {col}\n")

print("✓ preprocessing_summary.txt")

# ============================================================================
# 10. 完成
# ============================================================================
print("\n" + "="*80)
print("前處理完成！".center(80))
print("="*80)

print(f"\n✅ 資料品質保證:")
print(f"  • 無NaN: {X.isnull().sum().sum() == 0}")
print(f"  • 無inf: {np.isinf(X).sum().sum() == 0}")
print(f"  • 全數值型: {(X.dtypes != 'object').all()}")
print(f"  • 最終特徵: {X.shape[1]}個")
print(f"  • 有效樣本: {X.shape[0]}筆")

print("\n✅ 關鍵改進:")
print("  1. 創建10個衍生特徵")
print("  2. 每個步驟後檢查並處理NaN/inf")
print("  3. 移除原始欄位避免資訊重複")
print("  4. 移除結果變數避免資料洩漏")
print("  5. 確保所有特徵都是數值型")

if abs(correlations[max_corr_feature]) > 0.1:
    print(f"\n✅ 發現強相關特徵: {max_corr_feature} ({correlations[max_corr_feature]:.4f})")
elif abs(correlations[max_corr_feature]) > 0.05:
    print(f"\n⚠️ 最強相關特徵: {max_corr_feature} ({correlations[max_corr_feature]:.4f})")
    print("  → 相關性偏低")
else:
    print(f"\n⚠️ 最強相關特徵: {max_corr_feature} ({correlations[max_corr_feature]:.4f})")
    print("  → 相關性很低，預測可能仍然困難")

print("\n下一步:")
print("  執行 model.py 訓練模型")