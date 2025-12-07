"""
步驟1: 資料前處理 + 特徵工程
創建更有預測力的衍生特徵
"""

import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("步驟1: 資料前處理 + 特徵工程".center(80))
print("="*80)

# ============================================================================
# 1. 載入原始資料
# ============================================================================
print("\n[1] 載入原始資料")
df = pd.read_csv("C:/Users/fiona/Downloads/corporate_work_hours_productivity (1).csv")
print(f"✓ 載入完成: {len(df)}筆")

# 識別變數
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
if 'Employee_ID' in numerical_cols:
    numerical_cols.remove('Employee_ID')

# ============================================================================
# 2. 創建目標變數
# ============================================================================
print("\n[2] 創建目標變數")
threshold = df['Productivity_Score'].quantile(0.67)
print(f"✓ 67百分位數: {threshold:.2f}")

df['Productivity_Category'] = df['Productivity_Score'].apply(
    lambda x: 'High' if x >= threshold else 'Not_High'
)
print(f"✓ 目標分布: {df['Productivity_Category'].value_counts().to_dict()}")

# ============================================================================
# 3. 特徵工程 ⭐ 新增！
# ============================================================================
print("\n[3] 特徵工程 - 創建衍生特徵")

# 3.1 效率指標
print("\n3.1 效率指標")

# 每小時完成任務數 (效率核心指標)
df['Tasks_Per_Hour'] = df['Tasks_Completed_Per_Day'] / (df['Monthly_Hours_Worked'] / 22)
print("  ✓ Tasks_Per_Hour: 每小時完成任務數")

# 每次會議的任務產出
df['Tasks_Per_Meeting'] = df['Tasks_Completed_Per_Day'] / (df['Meetings_per_Week'] + 1)  # +1避免除0
print("  ✓ Tasks_Per_Meeting: 每次會議的任務產出")

# 加班效率
df['Overtime_Efficiency'] = df['Tasks_Completed_Per_Day'] / (df['Overtime_Hours_Per_Week'] + 1)
print("  ✓ Overtime_Efficiency: 加班效率")

# 3.2 工作負荷指標
print("\n3.2 工作負荷指標")

# 總工作時數
df['Total_Work_Hours'] = df['Monthly_Hours_Worked'] + df['Overtime_Hours_Per_Week'] * 4
print("  ✓ Total_Work_Hours: 月總工作時數")

# 會議負擔比例
df['Meeting_Burden'] = df['Meetings_per_Week'] / (df['Monthly_Hours_Worked'] / 22 / 8)  # 每天工時
print("  ✓ Meeting_Burden: 會議佔工作時間比例")

# 加班比例
df['Overtime_Ratio'] = df['Overtime_Hours_Per_Week'] / (df['Monthly_Hours_Worked'] / 4)
print("  ✓ Overtime_Ratio: 加班佔正常工時比例")

# 3.3 經驗與資歷指標
print("\n3.3 經驗與資歷指標")

# 年齡經驗比
df['Age_Experience_Ratio'] = df['Age'] / (df['Years_at_Company'] + 1)
print("  ✓ Age_Experience_Ratio: 年齡經驗比")

# 資歷等級分數 (年資 × 職等)
df['Seniority_Score'] = df['Years_at_Company'] * df['Job_Level'].map(
    {'Entry': 1, 'Mid': 2, 'Senior': 3, 'Manager': 4}
)
print("  ✓ Seniority_Score: 資歷分數")

# 3.4 工作滿意度相關
print("\n3.4 綜合指標")

# 工作生活平衡分數
work_life_map = {'Poor': 1, 'Fair': 2, 'Good': 3, 'Excellent': 4}
df['Work_Life_Score'] = df['Work_Life_Balance'].map(work_life_map)
print("  ✓ Work_Life_Score: 工作生活平衡分數")

# 滿意度 × 工作生活平衡
df['Satisfaction_Balance'] = df['Job_Satisfaction'] * df['Work_Life_Score']
print("  ✓ Satisfaction_Balance: 滿意度×平衡")

# 3.5 部門與遠端工作組合
print("\n3.5 類別組合特徵")

# 遠端工作 × 部門
df['Remote_Dept'] = df['Remote_Work'] + '_' + df['Department']
print("  ✓ Remote_Dept: 遠端工作×部門組合")

print(f"\n✓ 新增特徵數: 11個")
print(f"✓ 總特徵數: {df.shape[1] - 3}個 (扣除ID, Score, Category)")

# ============================================================================
# 4. 檢查新特徵與目標的相關性
# ============================================================================
print("\n[4] 新特徵相關性檢查")

y_numeric = df['Productivity_Category'].map({'Not_High': 0, 'High': 1})

new_features = ['Tasks_Per_Hour', 'Tasks_Per_Meeting', 'Overtime_Efficiency',
                'Total_Work_Hours', 'Meeting_Burden', 'Overtime_Ratio',
                'Age_Experience_Ratio', 'Seniority_Score', 
                'Work_Life_Score', 'Satisfaction_Balance']

correlations = {}
for col in new_features:
    corr = df[col].corr(y_numeric)
    correlations[col] = corr

corr_df = pd.DataFrame.from_dict(correlations, orient='index', columns=['Correlation'])
corr_df = corr_df.sort_values('Correlation', ascending=False)

print("\n新特徵與High類別的相關性:")
print(corr_df.round(4))

# ============================================================================
# 5. 異常值處理
# ============================================================================
print("\n[5] 異常值處理")

def handle_outliers_iqr(data, columns):
    df_clean = data.copy()
    for col in columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_clean[col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound)
    return df_clean

# 處理原始數值特徵
numerical_for_outlier = ['Age', 'Years_at_Company', 'Monthly_Hours_Worked',
                         'Meetings_per_Week', 'Tasks_Completed_Per_Day',
                         'Overtime_Hours_Per_Week', 'Job_Satisfaction']

df = handle_outliers_iqr(df, numerical_for_outlier)

# 處理新特徵
df = handle_outliers_iqr(df, new_features)

print("✓ 異常值處理完成")

# ============================================================================
# 6. 類別編碼
# ============================================================================
print("\n[6] 類別編碼")

df_encoded = df.copy()

# One-Hot Encoding
nominal_vars = ['Department', 'Remote_Work', 'Work_Life_Balance', 'Remote_Dept']
for col in nominal_vars:
    if col in df_encoded.columns:
        dummies = pd.get_dummies(df_encoded[col], prefix=col, drop_first=True)
        df_encoded = pd.concat([df_encoded, dummies], axis=1)
        df_encoded.drop(col, axis=1, inplace=True)
        print(f"  ✓ {col}: One-Hot編碼")

# Label Encoding
if 'Job_Level' in df_encoded.columns:
    level_mapping = {'Entry': 0, 'Mid': 1, 'Senior': 2, 'Manager': 3}
    df_encoded['Job_Level'] = df_encoded['Job_Level'].map(level_mapping)
    print(f"  ✓ Job_Level: Label編碼")

# ============================================================================
# 7. 準備X和y
# ============================================================================
print("\n[7] 準備特徵和目標")

cols_to_drop = ['Employee_ID', 'Productivity_Score', 'Productivity_Category',
                'Annual_Salary', 'Absences_Per_Year']

X = df_encoded.drop([col for col in cols_to_drop if col in df_encoded.columns], axis=1)
y = df['Productivity_Category']

print(f"✓ 特徵數: {X.shape[1]} (原始18 + 新增特徵)")
print(f"✓ 樣本數: {X.shape[0]}")

# 顯示前10個特徵
print(f"\n前10個特徵: {list(X.columns[:10])}")

# ============================================================================
# 8. 儲存
# ============================================================================
print("\n[8] 儲存處理好的資料")

X.to_csv('X_processed_FE.csv', index=False, encoding='utf-8-sig')
y.to_csv('y_processed_FE.csv', index=False, encoding='utf-8-sig', header=['Productivity_Category'])

data_dict = {
    'X': X,
    'y': y,
    'feature_names': list(X.columns),
    'threshold': threshold,
    'class_counts': y.value_counts().to_dict(),
    'new_features': new_features,
    'feature_correlations': corr_df.to_dict()['Correlation']
}

with open('processed_data_FE.pkl', 'wb') as f:
    pickle.dump(data_dict, f)

print("✓ 已儲存: X_processed_FE.csv")
print("✓ 已儲存: y_processed_FE.csv")
print("✓ 已儲存: processed_data_FE.pkl")

# 儲存摘要
with open('preprocessing_FE_summary.txt', 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("資料前處理 + 特徵工程總結\n")
    f.write("="*80 + "\n\n")
    f.write(f"原始特徵數: 18\n")
    f.write(f"新增特徵數: 11\n")
    f.write(f"處理後總特徵數: {X.shape[1]}\n\n")
    f.write("新增特徵列表:\n")
    for i, feat in enumerate(new_features, 1):
        f.write(f"  {i}. {feat}\n")
    f.write("\n新特徵相關性:\n")
    f.write(corr_df.to_string())
    f.write("\n\n所有特徵:\n")
    for i, col in enumerate(X.columns, 1):
        f.write(f"  {i}. {col}\n")

print("✓ 已儲存: preprocessing_FE_summary.txt")

print("\n" + "="*80)
print("前處理 + 特徵工程完成！".center(80))
print("="*80)

print("\n重要發現:")
if corr_df['Correlation'].abs().max() > 0.1:
    print("  ✅ 發現相關性>0.1的新特徵！")
    print(f"  ✅ 最強相關: {corr_df.index[0]} ({corr_df.iloc[0,0]:.4f})")
else:
    print("  ⚠️ 新特徵相關性仍然較低")
    print("  → 可能需要更複雜的特徵或非線性轉換")

print("\n下一步:")
print("  執行 step2_modeling.py (記得改成載入 processed_data_FE.pkl)")