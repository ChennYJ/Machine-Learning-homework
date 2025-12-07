"""
企業生產力資料建模專案
Group 4: Chen Yi-jun, Chen Ying-chu, Huang Yong-xuan

完整實施流程:
1. 資料載入與探索
2. 描述性統計
3. 資料前處理
4. 模型訓練與評估 (CART, SVM, KNN)
5. 結果比較與視覺化
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report)
import warnings
warnings.filterwarnings('ignore')

# 設定中文字體
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

print("="*80)
print("企業生產力資料建模專案".center(80))
print("="*80)

# ============================================================================
# 1. 資料載入與初步探索
# ============================================================================
print("\n[步驟1] 資料載入與初步探索")
print("-"*80)

# 載入資料
df = pd.read_csv('corporate_work_hours_productivity.csv')

print(f"資料筆數: {len(df)}")
print(f"特徵數量: {df.shape[1]}")
print(f"\n欄位名稱:\n{df.columns.tolist()}")
print(f"\n前5筆資料:")
print(df.head())

# 資料型態檢查
print(f"\n資料型態:")
print(df.dtypes)

# 遺失值檢查
print(f"\n遺失值統計:")
missing_values = df.isnull().sum()
print(missing_values[missing_values > 0] if missing_values.sum() > 0 else "無遺失值")

# ============================================================================
# 2. 變數定義與描述性統計
# ============================================================================
print("\n[步驟2] 變數定義與描述性統計")
print("-"*80)

# 2.1 識別數值型與類別型變數
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

# 移除ID欄位
if 'Employee_ID' in numerical_cols:
    numerical_cols.remove('Employee_ID')

print(f"\n數值型變數 ({len(numerical_cols)}個):")
print(numerical_cols)
print(f"\n類別型變數 ({len(categorical_cols)}個):")
print(categorical_cols)

# 2.2 數值型變數描述性統計
print("\n數值型變數描述性統計:")
print(df[numerical_cols].describe().round(2))

# 儲存詳細統計
detailed_stats = pd.DataFrame()
for col in numerical_cols:
    stats = {
        'Variable': col,
        'Mean': df[col].mean(),
        'Median': df[col].median(),
        'Std': df[col].std(),
        'Min': df[col].min(),
        'Max': df[col].max(),
        'Q1': df[col].quantile(0.25),
        'Q3': df[col].quantile(0.75),
        'Skewness': df[col].skew(),
        'Kurtosis': df[col].kurtosis()
    }
    detailed_stats = pd.concat([detailed_stats, pd.DataFrame([stats])], ignore_index=True)

detailed_stats.to_csv('numerical_statistics.csv', index=False, encoding='utf-8-sig')
print("\n✓ 數值型變數詳細統計已儲存至 'numerical_statistics.csv'")

# 2.3 類別型變數描述性統計
print("\n類別型變數描述性統計:")
categorical_stats = pd.DataFrame()
for col in categorical_cols:
    freq = df[col].value_counts()
    stats = {
        'Variable': col,
        'Categories': len(freq),
        'Mode': df[col].mode()[0],
        'Mode_Frequency': freq.iloc[0],
        'Mode_Percentage': f"{(freq.iloc[0]/len(df)*100):.2f}%"
    }
    categorical_stats = pd.concat([categorical_stats, pd.DataFrame([stats])], ignore_index=True)
    print(f"\n{col}:")
    print(df[col].value_counts())

categorical_stats.to_csv('categorical_statistics.csv', index=False, encoding='utf-8-sig')
print("\n✓ 類別型變數統計已儲存至 'categorical_statistics.csv'")

# ============================================================================
# 3. 創建目標變數 (將Productivity_Score轉為三分類)
# ============================================================================
print("\n[步驟3] 創建分類目標變數")
print("-"*80)

# 確認Productivity_Score存在
if 'Productivity_Score' in df.columns:
    # 計算分位數
    q33 = df['Productivity_Score'].quantile(0.33)
    q67 = df['Productivity_Score'].quantile(0.67)
    
    print(f"Productivity_Score分布:")
    print(f"  33百分位數: {q33:.2f}")
    print(f"  67百分位數: {q67:.2f}")
    
    # 創建三分類目標變數
    def categorize_productivity(score):
        if score < q33:
            return 'Low'
        elif score < q67:
            return 'Medium'
        else:
            return 'High'
    
    df['Productivity_Category'] = df['Productivity_Score'].apply(categorize_productivity)
    
    # 檢查分類分布
    print(f"\n目標變數分布:")
    print(df['Productivity_Category'].value_counts())
    print(f"\n百分比分布:")
    print(df['Productivity_Category'].value_counts(normalize=True).round(3))
    
    # 移除原始分數(避免資料洩漏)
    target_col = 'Productivity_Category'
else:
    print("警告: 找不到Productivity_Score欄位,請檢查資料")
    target_col = None

# ============================================================================
# 4. 資料視覺化
# ============================================================================
print("\n[步驟4] 資料視覺化")
print("-"*80)

# 4.1 數值變數分布圖
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.ravel()

for idx, col in enumerate(numerical_cols[:9]):
    axes[idx].hist(df[col], bins=30, edgecolor='black', alpha=0.7)
    axes[idx].set_title(f'{col} 分布')
    axes[idx].set_xlabel(col)
    axes[idx].set_ylabel('頻率')

plt.tight_layout()
plt.savefig('numerical_distributions.png', dpi=300, bbox_inches='tight')
print("✓ 數值變數分布圖已儲存")
plt.close()

# 4.2 箱型圖
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.ravel()

for idx, col in enumerate(numerical_cols[:9]):
    axes[idx].boxplot(df[col])
    axes[idx].set_title(f'{col} 箱型圖')
    axes[idx].set_ylabel(col)

plt.tight_layout()
plt.savefig('numerical_boxplots.png', dpi=300, bbox_inches='tight')
print("✓ 箱型圖已儲存")
plt.close()

# 4.3 相關係數矩陣
correlation_matrix = df[numerical_cols].corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, square=True, linewidths=1)
plt.title('特徵相關係數矩陣', fontsize=16)
plt.tight_layout()
plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
print("✓ 相關係數矩陣已儲存")
plt.close()

# 4.4 目標變數分布
if target_col:
    plt.figure(figsize=(8, 6))
    df[target_col].value_counts().plot(kind='bar', color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    plt.title('生產力類別分布', fontsize=14)
    plt.xlabel('生產力類別')
    plt.ylabel('員工數')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig('target_distribution.png', dpi=300, bbox_inches='tight')
    print("✓ 目標變數分布圖已儲存")
    plt.close()

# ============================================================================
# 5. 資料前處理
# ============================================================================
print("\n[步驟5] 資料前處理")
print("-"*80)

# 5.1 處理異常值 (使用IQR方法)
print("\n5.1 異常值處理")
def remove_outliers_iqr(data, columns):
    df_clean = data.copy()
    outliers_count = {}
    
    for col in columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = ((df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)).sum()
        outliers_count[col] = outliers
        
        # 使用截斷法處理異常值
        df_clean[col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound)
    
    return df_clean, outliers_count

# 對數值變數處理異常值(排除Productivity_Score因為已用於分類)
numerical_cols_for_outlier = [col for col in numerical_cols if col != 'Productivity_Score']
df_processed, outliers_info = remove_outliers_iqr(df, numerical_cols_for_outlier)

print("各變數異常值數量:")
for col, count in outliers_info.items():
    print(f"  {col}: {count}")

# 5.2 類別編碼
print("\n5.2 類別變數編碼")
df_encoded = df_processed.copy()

# One-Hot Encoding for nominal variables
nominal_vars = [col for col in categorical_cols if col not in ['Job_Level']]
for col in nominal_vars:
    if col in df_encoded.columns:
        dummies = pd.get_dummies(df_encoded[col], prefix=col, drop_first=True)
        df_encoded = pd.concat([df_encoded, dummies], axis=1)
        df_encoded.drop(col, axis=1, inplace=True)
        print(f"  {col}: One-Hot編碼完成")

# Label Encoding for ordinal variable (Job_Level有順序性)
if 'Job_Level' in df_encoded.columns:
    level_mapping = {'Entry': 0, 'Mid': 1, 'Senior': 2, 'Manager': 3}
    df_encoded['Job_Level'] = df_encoded['Job_Level'].map(level_mapping)
    print(f"  Job_Level: Label編碼完成 (Entry=0, Mid=1, Senior=2, Manager=3)")

# 5.3 準備特徵和目標
print("\n5.3 準備訓練資料")

# 移除不需要的欄位
cols_to_drop = ['Employee_ID', 'Productivity_Score', 'Productivity_Category']
X = df_encoded.drop([col for col in cols_to_drop if col in df_encoded.columns], axis=1)
y = df_processed[target_col]

print(f"特徵數量: {X.shape[1]}")
print(f"樣本數量: {X.shape[0]}")
print(f"特徵名稱: {X.columns.tolist()}")

# ============================================================================
# 6. 模型訓練與評估 (使用多個random_state)
# ============================================================================
print("\n[步驟6] 模型訓練與評估")
print("="*80)

# 定義random seeds (參考作業4)
random_seeds = [4, 40, 400]
results_all_seeds = []

for seed in random_seeds:
    print(f"\n{'='*80}")
    print(f"Random Seed: {seed}".center(80))
    print(f"{'='*80}")
    
    # 資料分割 (80-20 split, stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    
    print(f"\n訓練集大小: {X_train.shape[0]}")
    print(f"測試集大小: {X_test.shape[0]}")
    
    # 訓練集目標分布
    print(f"\n訓練集目標分布:")
    print(y_train.value_counts())
    
    # ========================================================================
    # 6.1 CART (不需標準化)
    # ========================================================================
    print(f"\n{'─'*80}")
    print("6.1 CART (Decision Tree)".center(80))
    print(f"{'─'*80}")
    
    # 超參數網格
    cart_param_grid = {
        'max_depth': [3, 5, 7, 10, None],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': [1, 2, 4],
        'criterion': ['gini', 'entropy']
    }
    
    cart_model = DecisionTreeClassifier(random_state=seed)
    cart_grid = GridSearchCV(cart_model, cart_param_grid, cv=5, 
                            scoring='accuracy', n_jobs=-1, verbose=0)
    
    print("開始GridSearch...")
    cart_grid.fit(X_train, y_train)
    
    print(f"\n最佳參數: {cart_grid.best_params_}")
    print(f"最佳CV分數: {cart_grid.best_score_:.4f}")
    
    # 預測
    cart_pred = cart_grid.predict(X_test)
    
    # 評估
    cart_results = {
        'Model': 'CART',
        'Seed': seed,
        'Accuracy': accuracy_score(y_test, cart_pred),
        'Precision_Macro': precision_score(y_test, cart_pred, average='macro'),
        'Recall_Macro': recall_score(y_test, cart_pred, average='macro'),
        'F1_Macro': f1_score(y_test, cart_pred, average='macro'),
        'Best_Params': str(cart_grid.best_params_)
    }
    
    print(f"\n測試集評估:")
    print(f"  Accuracy: {cart_results['Accuracy']:.4f}")
    print(f"  Macro F1: {cart_results['F1_Macro']:.4f}")
    
    print(f"\n詳細分類報告:")
    print(classification_report(y_test, cart_pred))
    
    # ========================================================================
    # 6.2 SVM (需要標準化)
    # ========================================================================
    print(f"\n{'─'*80}")
    print("6.2 SVM (Support Vector Machine) - 使用標準化".center(80))
    print(f"{'─'*80}")
    
    # 標準化
    scaler_svm = StandardScaler()
    X_train_scaled_svm = scaler_svm.fit_transform(X_train)
    X_test_scaled_svm = scaler_svm.transform(X_test)
    
    print("✓ 特徵標準化完成 (SVM需要)")
    
    # 超參數網格
    svm_param_grid = {
        'C': [0.1, 1, 10, 100],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto']
    }
    
    svm_model = SVC(random_state=seed)
    svm_grid = GridSearchCV(svm_model, svm_param_grid, cv=5,
                           scoring='accuracy', n_jobs=-1, verbose=0)
    
    print("開始GridSearch...")
    svm_grid.fit(X_train_scaled_svm, y_train)
    
    print(f"\n最佳參數: {svm_grid.best_params_}")
    print(f"最佳CV分數: {svm_grid.best_score_:.4f}")
    
    # 預測
    svm_pred = svm_grid.predict(X_test_scaled_svm)
    
    # 評估
    svm_results = {
        'Model': 'SVM',
        'Seed': seed,
        'Accuracy': accuracy_score(y_test, svm_pred),
        'Precision_Macro': precision_score(y_test, svm_pred, average='macro'),
        'Recall_Macro': recall_score(y_test, svm_pred, average='macro'),
        'F1_Macro': f1_score(y_test, svm_pred, average='macro'),
        'Best_Params': str(svm_grid.best_params_)
    }
    
    print(f"\n測試集評估:")
    print(f"  Accuracy: {svm_results['Accuracy']:.4f}")
    print(f"  Macro F1: {svm_results['F1_Macro']:.4f}")
    
    print(f"\n詳細分類報告:")
    print(classification_report(y_test, svm_pred))
    
    # ========================================================================
    # 6.3 KNN (需要標準化 - 基於作業4的重要發現!)
    # ========================================================================
    print(f"\n{'─'*80}")
    print("6.3 KNN (K-Nearest Neighbors) - 使用標準化".center(80))
    print("*** 重要:基於作業4經驗,KNN必須使用標準化! ***".center(80))
    print(f"{'─'*80}")
    
    # 標準化
    scaler_knn = StandardScaler()
    X_train_scaled_knn = scaler_knn.fit_transform(X_train)
    X_test_scaled_knn = scaler_knn.transform(X_test)
    
    print("✓ 特徵標準化完成 (KNN必需!)")
    
    # 超參數網格
    knn_param_grid = {
        'n_neighbors': [3, 5, 7, 9, 11, 15],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }
    
    knn_model = KNeighborsClassifier()
    knn_grid = GridSearchCV(knn_model, knn_param_grid, cv=5,
                           scoring='accuracy', n_jobs=-1, verbose=0)
    
    print("開始GridSearch...")
    knn_grid.fit(X_train_scaled_knn, y_train)
    
    print(f"\n最佳參數: {knn_grid.best_params_}")
    print(f"最佳CV分數: {knn_grid.best_score_:.4f}")
    
    # 預測
    knn_pred = knn_grid.predict(X_test_scaled_knn)
    
    # 評估
    knn_results = {
        'Model': 'KNN',
        'Seed': seed,
        'Accuracy': accuracy_score(y_test, knn_pred),
        'Precision_Macro': precision_score(y_test, knn_pred, average='macro'),
        'Recall_Macro': recall_score(y_test, knn_pred, average='macro'),
        'F1_Macro': f1_score(y_test, knn_pred, average='macro'),
        'Best_Params': str(knn_grid.best_params_)
    }
    
    print(f"\n測試集評估:")
    print(f"  Accuracy: {knn_results['Accuracy']:.4f}")
    print(f"  Macro F1: {knn_results['F1_Macro']:.4f}")
    
    print(f"\n詳細分類報告:")
    print(classification_report(y_test, knn_pred))
    
    # 儲存本次seed的結果
    results_all_seeds.extend([cart_results, svm_results, knn_results])
    
    # 混淆矩陣視覺化
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, (pred, name) in enumerate([(cart_pred, 'CART'), 
                                         (svm_pred, 'SVM'), 
                                         (knn_pred, 'KNN')]):
        cm = confusion_matrix(y_test, pred, labels=['Low', 'Medium', 'High'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                   xticklabels=['Low', 'Medium', 'High'],
                   yticklabels=['Low', 'Medium', 'High'])
        axes[idx].set_title(f'{name} 混淆矩陣 (Seed={seed})')
        axes[idx].set_ylabel('實際類別')
        axes[idx].set_xlabel('預測類別')
    
    plt.tight_layout()
    plt.savefig(f'confusion_matrices_seed_{seed}.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ 混淆矩陣已儲存 (seed={seed})")
    plt.close()

# ============================================================================
# 7. 結果彙整與比較
# ============================================================================
print(f"\n{'='*80}")
print("最終結果彙整".center(80))
print(f"{'='*80}")

# 轉換為DataFrame
results_df = pd.DataFrame(results_all_seeds)

# 計算平均性能
print("\n各模型平均性能 (跨3個random seeds):")
avg_results = results_df.groupby('Model')[['Accuracy', 'Precision_Macro', 
                                            'Recall_Macro', 'F1_Macro']].mean()
avg_results = avg_results.round(4)
print(avg_results)

# 計算標準差
print("\n各模型性能標準差:")
std_results = results_df.groupby('Model')[['Accuracy', 'Precision_Macro', 
                                           'Recall_Macro', 'F1_Macro']].std()
std_results = std_results.round(4)
print(std_results)

# 儲存完整結果
results_df.to_csv('all_results_detailed.csv', index=False, encoding='utf-8-sig')
avg_results.to_csv('average_results.csv', encoding='utf-8-sig')
std_results.to_csv('std_results.csv', encoding='utf-8-sig')

print("\n✓ 所有結果已儲存")
print("  - all_results_detailed.csv: 所有seed的詳細結果")
print("  - average_results.csv: 平均性能")
print("  - std_results.csv: 性能標準差")

# 視覺化比較
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

metrics = ['Accuracy', 'Precision_Macro', 'Recall_Macro', 'F1_Macro']
metric_names = ['準確率', '精確率(Macro)', '召回率(Macro)', 'F1分數(Macro)']

for idx, (metric, name) in enumerate(zip(metrics, metric_names)):
    ax = axes[idx // 2, idx % 2]
    
    # 繪製長條圖
    avg_results[metric].plot(kind='bar', ax=ax, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    ax.set_title(f'{name}比較', fontsize=12)
    ax.set_ylabel(name)
    ax.set_xlabel('模型')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.grid(axis='y', alpha=0.3)
    
    # 添加數值標籤
    for i, v in enumerate(avg_results[metric]):
        ax.text(i, v + 0.01, f'{v:.4f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
print("\n✓ 模型比較圖已儲存")
plt.close()

# ============================================================================
# 8. 關鍵發現總結
# ============================================================================
print(f"\n{'='*80}")
print("關鍵發現與建議".center(80))
print(f"{'='*80}")

best_model = avg_results['Accuracy'].idxmax()
best_acc = avg_results.loc[best_model, 'Accuracy']

print(f"\n✓ 最佳模型: {best_model}")
print(f"✓ 平均準確率: {best_acc:.4f}")
print(f"\n✓ 特徵縮放影響:")
print(f"   - CART: 不需要標準化 (對縮放不敏感)")
print(f"   - SVM: 需要標準化 (對特徵尺度敏感)")
print(f"   - KNN: 必須標準化 (基於作業4的重要發現!)")

print(f"\n✓ 模型穩健性 (多seed驗證):")
for model in ['CART', 'SVM', 'KNN']:
    std = std_results.loc[model, 'Accuracy']
    print(f"   - {model}: 準確率標準差 = {std:.4f}")

print("\n" + "="*80)
print("分析完成!".center(80))
print("="*80)
print("\n所有結果檔案已儲存,可用於製作報告和簡報。")
print("\n生成的檔案:")
print("  1. numerical_statistics.csv - 數值變數統計")
print("  2. categorical_statistics.csv - 類別變數統計")
print("  3. numerical_distributions.png - 分布圖")
print("  4. numerical_boxplots.png - 箱型圖")
print("  5. correlation_matrix.png - 相關係數矩陣")
print("  6. target_distribution.png - 目標變數分布")
print("  7. confusion_matrices_seed_*.png - 混淆矩陣 (每個seed)")
print("  8. all_results_detailed.csv - 詳細結果")
print("  9. average_results.csv - 平均性能")
print("  10. std_results.csv - 性能標準差")
print("  11. model_comparison.png - 模型比較圖")