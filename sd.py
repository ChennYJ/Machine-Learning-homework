"""
企業生產力資料建模 - 修正版
Group 4: Chen Yi-jun, Chen Ying-chu, Huang Yong-xuan

修正項目:
1. GridSearchCV使用f1_weighted (避免nan)
2. 擴大參數搜索範圍
3. 移除可能造成洩漏的變數選項
4. 增加診斷輸出
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report,
                             make_scorer)
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

print("="*80)
print("企業生產力資料建模 - 修正版 (前33%為高生產力)".center(80))
print("="*80)

# 載入資料
df = pd.read_csv("C:/Users/fiona/Downloads/corporate_work_hours_productivity (1).csv")
print(f"\n資料: {len(df)}筆, {df.shape[1]}欄")
print(f"欄位: {df.columns.tolist()}")

# 識別變數類型
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
if 'Employee_ID' in numerical_cols:
    numerical_cols.remove('Employee_ID')

print(f"\n數值變數: {numerical_cols}")
print(f"類別變數: {categorical_cols}")

# 創建目標變數 - 前33%為高生產力
if 'Productivity_Score' in df.columns:
    threshold = df['Productivity_Score'].quantile(0.67)
    print(f"\n✓ 67百分位數: {threshold:.2f}")
    
    df['Productivity_Category'] = df['Productivity_Score'].apply(
        lambda x: 'High' if x >= threshold else 'Not_High'
    )
    
    print(f"\n目標分布:")
    print(df['Productivity_Category'].value_counts())
    print(df['Productivity_Category'].value_counts(normalize=True))

# 異常值處理
def handle_outliers(data, cols):
    df_clean = data.copy()
    for col in cols:
        Q1, Q3 = df_clean[col].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        df_clean[col] = df_clean[col].clip(Q1-1.5*IQR, Q3+1.5*IQR)
    return df_clean

numerical_for_outlier = [c for c in numerical_cols 
                         if c not in ['Productivity_Score', 'Annual_Salary', 'Absences_Per_Year']]
df = handle_outliers(df, numerical_for_outlier)
print("\n✓ 異常值處理完成")

# 類別編碼
df_encoded = df.copy()

# One-Hot
nominal_vars = [col for col in categorical_cols if col != 'Job_Level']
for col in nominal_vars:
    if col in df_encoded.columns:
        dummies = pd.get_dummies(df_encoded[col], prefix=col, drop_first=True)
        df_encoded = pd.concat([df_encoded, dummies], axis=1)
        df_encoded.drop(col, axis=1, inplace=True)

# Label Encoding
if 'Job_Level' in df_encoded.columns:
    df_encoded['Job_Level'] = df_encoded['Job_Level'].map(
        {'Entry':0, 'Mid':1, 'Senior':2, 'Manager':3}
    )

print("✓ 編碼完成")

# 準備X和y - 移除可能造成資料洩漏的變數
cols_to_drop = ['Employee_ID', 'Productivity_Score', 'Productivity_Category',
                'Annual_Salary', 'Absences_Per_Year']  # 移除可能的結果變數

X = df_encoded.drop([col for col in cols_to_drop if col in df_encoded.columns], axis=1)
y = df['Productivity_Category']

print(f"\n✓ 特徵數: {X.shape[1]}")
print(f"✓ 特徵: {X.columns.tolist()}")

# 定義F1 scorer for 'High' class
from sklearn.metrics import make_scorer
f1_high_scorer = make_scorer(f1_score, pos_label='High', average='binary')

# 實驗
results = []
for seed in [4, 40, 400]:
    print(f"\n{'='*80}\nSeed: {seed}\n{'='*80}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    
    print(f"訓練: {X_train.shape[0]}, 測試: {X_test.shape[0]}")
    print(f"訓練集分布:\n{y_train.value_counts()}")
    print(f"測試集分布:\n{y_test.value_counts()}")
    
    # CART
    print("\n[CART - class_weight='balanced']")
    cart = GridSearchCV(
        DecisionTreeClassifier(random_state=seed, class_weight='balanced'),
        {
            'max_depth': [3, 5, 7, 10, 15, None],
            'min_samples_split': [2, 5, 10, 20],
            'min_samples_leaf': [1, 2, 4, 8],
            'criterion': ['gini', 'entropy']
        },
        cv=5, scoring=f1_high_scorer, n_jobs=-1, verbose=0
    )
    cart.fit(X_train, y_train)
    cart_pred = cart.predict(X_test)
    
    print(f"最佳參數: {cart.best_params_}")
    print(f"最佳CV F1(High): {cart.best_score_:.4f}")
    
    results.append({
        'Model': 'CART', 'Seed': seed,
        'Accuracy': accuracy_score(y_test, cart_pred),
        'Precision': precision_score(y_test, cart_pred, pos_label='High', zero_division=0),
        'Recall': recall_score(y_test, cart_pred, pos_label='High', zero_division=0),
        'Specificity': recall_score(y_test, cart_pred, pos_label='Not_High', zero_division=0),
        'F1': f1_score(y_test, cart_pred, pos_label='High', zero_division=0)
    })
    
    print(f"測試集 - Acc:{results[-1]['Accuracy']:.4f}, F1:{results[-1]['F1']:.4f}, "
          f"Recall:{results[-1]['Recall']:.4f}, Precision:{results[-1]['Precision']:.4f}")
    
    # SVM
    print("\n[SVM - 標準化 + class_weight='balanced']")
    scaler = StandardScaler()
    X_tr_svm = scaler.fit_transform(X_train)
    X_te_svm = scaler.transform(X_test)
    
    svm = GridSearchCV(
        SVC(random_state=seed, class_weight='balanced'),
        {
            'C': [0.01, 0.1, 1, 10, 100],
            'kernel': ['linear', 'rbf', 'poly'],
            'gamma': ['scale', 'auto']
        },
        cv=5, scoring=f1_high_scorer, n_jobs=-1, verbose=0
    )
    svm.fit(X_tr_svm, y_train)
    svm_pred = svm.predict(X_te_svm)
    
    print(f"最佳參數: {svm.best_params_}")
    print(f"最佳CV F1(High): {svm.best_score_:.4f}")
    
    results.append({
        'Model': 'SVM', 'Seed': seed,
        'Accuracy': accuracy_score(y_test, svm_pred),
        'Precision': precision_score(y_test, svm_pred, pos_label='High', zero_division=0),
        'Recall': recall_score(y_test, svm_pred, pos_label='High', zero_division=0),
        'Specificity': recall_score(y_test, svm_pred, pos_label='Not_High', zero_division=0),
        'F1': f1_score(y_test, svm_pred, pos_label='High', zero_division=0)
    })
    
    print(f"測試集 - Acc:{results[-1]['Accuracy']:.4f}, F1:{results[-1]['F1']:.4f}, "
          f"Recall:{results[-1]['Recall']:.4f}, Precision:{results[-1]['Precision']:.4f}")
    
    # KNN  
    print("\n[KNN - 標準化 ★作業4經驗★]")
    scaler = StandardScaler()
    X_tr_knn = scaler.fit_transform(X_train)
    X_te_knn = scaler.transform(X_test)
    
    knn = GridSearchCV(
        KNeighborsClassifier(),
        {
            'n_neighbors': [3, 5, 7, 9, 11, 15, 20, 25, 30],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan', 'minkowski']
        },
        cv=5, scoring=f1_high_scorer, n_jobs=-1, verbose=0
    )
    knn.fit(X_tr_knn, y_train)
    knn_pred = knn.predict(X_te_knn)
    
    print(f"最佳參數: {knn.best_params_}")
    print(f"最佳CV F1(High): {knn.best_score_:.4f}")
    
    results.append({
        'Model': 'KNN', 'Seed': seed,
        'Accuracy': accuracy_score(y_test, knn_pred),
        'Precision': precision_score(y_test, knn_pred, pos_label='High', zero_division=0),
        'Recall': recall_score(y_test, knn_pred, pos_label='High', zero_division=0),
        'Specificity': recall_score(y_test, knn_pred, pos_label='Not_High', zero_division=0),
        'F1': f1_score(y_test, knn_pred, pos_label='High', zero_division=0)
    })
    
    print(f"測試集 - Acc:{results[-1]['Accuracy']:.4f}, F1:{results[-1]['F1']:.4f}, "
          f"Recall:{results[-1]['Recall']:.4f}, Precision:{results[-1]['Precision']:.4f}")
    
    # 詳細報告
    print(f"\n詳細分類報告 (KNN):")
    print(classification_report(y_test, knn_pred, labels=['Not_High', 'High'], target_names=['Not_High', 'High']))
    
    # 混淆矩陣
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for idx, (pred, name) in enumerate([(cart_pred,'CART'), (svm_pred,'SVM'), (knn_pred,'KNN')]):
        cm = confusion_matrix(y_test, pred, labels=['Not_High', 'High'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                   xticklabels=['Not_High', 'High'], 
                   yticklabels=['Not_High', 'High'])
        axes[idx].set_title(f'{name} (Seed={seed})')
        axes[idx].set_ylabel('實際')
        axes[idx].set_xlabel('預測')
    plt.tight_layout()
    plt.savefig(f'confusion_matrices_seed_{seed}.png', dpi=300)
    plt.close()

# 彙整
df_res = pd.DataFrame(results)
avg = df_res.groupby('Model')[['Accuracy', 'Precision', 'Recall', 'Specificity', 'F1']].mean()
std = df_res.groupby('Model')[['Accuracy', 'Precision', 'Recall', 'Specificity', 'F1']].std()

print(f"\n{'='*80}\n平均結果 (High = 前33%)\n{'='*80}")
print(avg.round(4))
print(f"\n標準差:")
print(std.round(4))

# 儲存
df_res.to_csv('all_results_detailed.csv', index=False, encoding='utf-8-sig')
avg.to_csv('average_results.csv', encoding='utf-8-sig')
std.to_csv('std_results.csv', encoding='utf-8-sig')

print("\n✓ 分析完成!")
print(f"\n最佳模型: {avg['F1'].idxmax()}, F1={avg['F1'].max():.4f}")
print(f"Recall(High): {avg.loc[avg['F1'].idxmax(), 'Recall']:.4f}")

print("\n重要說明:")
print("  • 已移除 Annual_Salary 和 Absences_Per_Year (可能的結果變數)")
print("  • 使用 f1_score(pos_label='High') 作為優化目標")
print("  • class_weight='balanced' 處理不平衡")
print("  • 擴大了參數搜索範圍")