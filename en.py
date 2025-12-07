"""
企業生產力分析 - 增強版
包含特徵重要性分析和診斷工具
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
print("企業生產力分析 - 增強版 (含診斷)".center(80))
print("="*80)

# 載入資料
df = pd.read_csv("C:/Users/fiona/Downloads/corporate_work_hours_productivity (1).csv")
print(f"\n資料: {len(df)}筆")

numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
if 'Employee_ID' in numerical_cols:
    numerical_cols.remove('Employee_ID')

# 創建目標變數
threshold = df['Productivity_Score'].quantile(0.67)
print(f"\n67百分位數: {threshold:.2f}")

df['Productivity_Category'] = df['Productivity_Score'].apply(
    lambda x: 'High' if x >= threshold else 'Not_High'
)

print(f"\n目標分布:")
print(df['Productivity_Category'].value_counts())

# ============================================================================
# 特徵與目標相關性分析
# ============================================================================
print(f"\n{'='*80}")
print("特徵與目標相關性分析".center(80))
print(f"{'='*80}")

# 將目標轉為數值
y_numeric = df['Productivity_Category'].map({'Not_High': 0, 'High': 1})

# 計算相關性
correlations = {}
for col in numerical_cols:
    if col != 'Productivity_Score':
        corr = df[col].corr(y_numeric)
        correlations[col] = corr

# 排序
corr_df = pd.DataFrame.from_dict(correlations, orient='index', columns=['Correlation'])
corr_df = corr_df.sort_values('Correlation', ascending=False)

print("\n數值特徵與High類別的相關性:")
print(corr_df.round(4))

# 視覺化
plt.figure(figsize=(10, 6))
corr_df.plot(kind='barh', legend=False)
plt.title('特徵與High類別的相關性', fontsize=14, fontweight='bold')
plt.xlabel('Pearson相關係數')
plt.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
plt.tight_layout()
plt.savefig('feature_correlations.png', dpi=300)
plt.close()
print("\n✓ 相關性圖已儲存: feature_correlations.png")

# ============================================================================
# 資料前處理
# ============================================================================
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

# 編碼
df_encoded = df.copy()

nominal_vars = [col for col in categorical_cols if col != 'Job_Level']
for col in nominal_vars:
    if col in df_encoded.columns:
        dummies = pd.get_dummies(df_encoded[col], prefix=col, drop_first=True)
        df_encoded = pd.concat([df_encoded, dummies], axis=1)
        df_encoded.drop(col, axis=1, inplace=True)

if 'Job_Level' in df_encoded.columns:
    df_encoded['Job_Level'] = df_encoded['Job_Level'].map(
        {'Entry':0, 'Mid':1, 'Senior':2, 'Manager':3}
    )

# 準備X和y
cols_to_drop = ['Employee_ID', 'Productivity_Score', 'Productivity_Category',
                'Annual_Salary', 'Absences_Per_Year']

X = df_encoded.drop([col for col in cols_to_drop if col in df_encoded.columns], axis=1)
y = df['Productivity_Category']

print(f"\n特徵數: {X.shape[1]}")
print(f"特徵: {X.columns.tolist()}")

# ============================================================================
# 單次實驗（用於特徵重要性分析）
# ============================================================================
print(f"\n{'='*80}")
print("訓練模型並分析特徵重要性".center(80))
print(f"{'='*80}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# CART - 用於特徵重要性
cart_model = DecisionTreeClassifier(
    max_depth=7, min_samples_leaf=4, 
    class_weight='balanced', random_state=42
)
cart_model.fit(X_train, y_train)

# 特徵重要性
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': cart_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nCART特徵重要性 (Top 10):")
print(feature_importance.head(10).to_string(index=False))

# 視覺化
plt.figure(figsize=(10, 6))
top_features = feature_importance.head(10)
plt.barh(range(len(top_features)), top_features['Importance'])
plt.yticks(range(len(top_features)), top_features['Feature'])
plt.xlabel('重要性')
plt.title('CART特徵重要性 (Top 10)', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300)
plt.close()
print("\n✓ 特徵重要性圖已儲存: feature_importance.png")

# ============================================================================
# 完整實驗（3個seeds）
# ============================================================================
print(f"\n{'='*80}")
print("完整模型訓練 (3 seeds)".center(80))
print(f"{'='*80}")

f1_high_scorer = make_scorer(f1_score, pos_label='High', average='binary')

results = []
for seed in [4, 40, 400]:
    print(f"\n{'─'*80}\nSeed: {seed}\n{'─'*80}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    
    # CART
    print("\n[CART]")
    cart = GridSearchCV(
        DecisionTreeClassifier(random_state=seed, class_weight='balanced'),
        {
            'max_depth': [3, 5, 7, 10, 15],
            'min_samples_leaf': [2, 4, 8, 16],
            'criterion': ['gini', 'entropy']
        },
        cv=5, scoring=f1_high_scorer, n_jobs=-1, verbose=0
    )
    cart.fit(X_train, y_train)
    cart_pred = cart.predict(X_test)
    
    results.append({
        'Model': 'CART', 'Seed': seed,
        'Accuracy': accuracy_score(y_test, cart_pred),
        'Precision': precision_score(y_test, cart_pred, pos_label='High', zero_division=0),
        'Recall': recall_score(y_test, cart_pred, pos_label='High', zero_division=0),
        'F1': f1_score(y_test, cart_pred, pos_label='High', zero_division=0)
    })
    
    print(f"CV F1: {cart.best_score_:.4f}, 測試F1: {results[-1]['F1']:.4f}, "
          f"Recall: {results[-1]['Recall']:.4f}")
    
    # SVM
    print("\n[SVM]")
    scaler = StandardScaler()
    X_tr_svm = scaler.fit_transform(X_train)
    X_te_svm = scaler.transform(X_test)
    
    svm = GridSearchCV(
        SVC(random_state=seed, class_weight='balanced'),
        {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']},
        cv=5, scoring=f1_high_scorer, n_jobs=-1, verbose=0
    )
    svm.fit(X_tr_svm, y_train)
    svm_pred = svm.predict(X_te_svm)
    
    results.append({
        'Model': 'SVM', 'Seed': seed,
        'Accuracy': accuracy_score(y_test, svm_pred),
        'Precision': precision_score(y_test, svm_pred, pos_label='High', zero_division=0),
        'Recall': recall_score(y_test, svm_pred, pos_label='High', zero_division=0),
        'F1': f1_score(y_test, svm_pred, pos_label='High', zero_division=0)
    })
    
    print(f"CV F1: {svm.best_score_:.4f}, 測試F1: {results[-1]['F1']:.4f}, "
          f"Recall: {results[-1]['Recall']:.4f}")
    
    # KNN
    print("\n[KNN]")
    scaler = StandardScaler()
    X_tr_knn = scaler.fit_transform(X_train)
    X_te_knn = scaler.transform(X_test)
    
    knn = GridSearchCV(
        KNeighborsClassifier(),
        {'n_neighbors': [5, 10, 15, 20, 25], 'weights': ['uniform', 'distance']},
        cv=5, scoring=f1_high_scorer, n_jobs=-1, verbose=0
    )
    knn.fit(X_tr_knn, y_train)
    knn_pred = knn.predict(X_te_knn)
    
    results.append({
        'Model': 'KNN', 'Seed': seed,
        'Accuracy': accuracy_score(y_test, knn_pred),
        'Precision': precision_score(y_test, knn_pred, pos_label='High', zero_division=0),
        'Recall': recall_score(y_test, knn_pred, pos_label='High', zero_division=0),
        'F1': f1_score(y_test, knn_pred, pos_label='High', zero_division=0)
    })
    
    print(f"CV F1: {knn.best_score_:.4f}, 測試F1: {results[-1]['F1']:.4f}, "
          f"Recall: {results[-1]['Recall']:.4f}")

# 彙整
df_res = pd.DataFrame(results)
avg = df_res.groupby('Model')[['Accuracy', 'Precision', 'Recall', 'F1']].mean()

print(f"\n{'='*80}\n平均結果\n{'='*80}")
print(avg.round(4))

# 儲存
df_res.to_csv('results_enhanced.csv', index=False, encoding='utf-8-sig')
avg.to_csv('results_avg.csv', encoding='utf-8-sig')

print(f"\n{'='*80}")
print("診斷結論".center(80))
print(f"{'='*80}")

print(f"\n✓ 最佳模型: {avg['F1'].idxmax()}, F1={avg['F1'].max():.4f}")
print(f"✓ Recall(High): {avg.loc[avg['F1'].idxmax(), 'Recall']:.4f}")

print("\n✓ 生成檔案:")
print("  - feature_correlations.png: 特徵相關性")
print("  - feature_importance.png: 特徵重要性")
print("  - results_enhanced.csv: 詳細結果")
print("  - results_avg.csv: 平均結果")

print("\n✓ 性能分析:")
if avg['F1'].max() < 0.5:
    print("  ⚠️ F1 < 0.5: 預測性能偏低")
    print("  → 可能原因:")
    print("    1. 特徵與目標相關性不強（查看feature_correlations.png）")
    print("    2. 67百分位界限不夠明顯")
    print("    3. 需要更多特徵工程（交互項、多項式特徵）")
else:
    print("  ✅ F1 ≥ 0.5: 預測性能可接受")

print("\n✓ 下一步建議:")
print("  1. 查看 feature_correlations.png 找出最相關特徵")
print("  2. 查看 feature_importance.png 了解CART使用的主要特徵")
print("  3. 考慮只使用Top 10重要特徵重新訓練")
print("  4. 或嘗試創建新特徵（如效率指標：Tasks_Completed_Per_Day/Hours_Worked）")