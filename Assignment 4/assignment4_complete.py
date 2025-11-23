# Assignment 4 - KNN、CART、C4.5 比較

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
import seaborn as sns

# 資料預處理
from sklearn.preprocessing import StandardScaler

# 模型相關
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree

# 評估指標
from sklearn.metrics import (accuracy_score, f1_score, precision_score, 
                             recall_score, roc_auc_score, confusion_matrix,
                             roc_curve)

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Microsoft JhengHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 80)
print("Assignment 4: KNN、CART、C4.5 模型比較分析")
print("使用三個 Random Seeds: 4, 40, 400")
print("=" * 80)

# ============================================================
# 1. 資料載入與預處理
# ============================================================
print("\n[步驟 1] 資料載入與預處理")

df = pd.read_csv('Assignment 4/new_df.csv')

print(f"資料形狀: {df.shape}")
print(f"欄位名稱: {list(df.columns)}")

# 檢查資料是否已預處理
print("\n檢查資料狀態...")
if 'Pclass_1' in df.columns:
    print("資料已進行 One-Hot Encoding")
else:
    print("資料需要進行編碼")

# 檢查缺失值
print(f"\n缺失值統計:\n{df.isnull().sum()}")

# 處理缺失值
if df.isnull().sum().sum() > 0:
    print("\n處理缺失值...")
    if 'Age' in df.columns and df['Age'].isnull().sum() > 0:
        df['Age'].fillna(df['Age'].median(), inplace=True)
    if 'Embarked' in df.columns and df['Embarked'].isnull().sum() > 0:
        df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
else:
    print("無缺失值")

print(f"\n最終欄位: {list(df.columns)}")

# 確定目標變數
if 'Survived' in df.columns:
    target_column = 'Survived'
elif 'Survived_yes' in df.columns:
    target_column = 'Survived_yes'
else:
    raise ValueError("找不到目標變數 (Survived 或 Survived_yes)")

print(f"使用目標變數: {target_column}")

# 分離 X 和 y
X = df.drop([col for col in df.columns if col.startswith('Survived')], axis=1)
y = df[target_column]

print(f"\n特徵數量: {X.shape[1]}")
print(f"類別分布:\n{y.value_counts()}")

# ============================================================
# 2. 定義三個 Random Seeds
# ============================================================
SEEDS = [4, 40, 400]
print(f"\n使用的 Random Seeds: {SEEDS}")

# 儲存所有結果
all_results = {
    'KNN_原始': {},
    'KNN_標準化': {},
    'CART': {},
    'C4.5': {}
}

# 為每個 seed 準備資料分割
data_splits = {}
for seed in SEEDS:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    
    # 標準化處理
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    data_splits[seed] = {
        'X_train': X_train,
        'X_test': X_test,
        'X_train_scaled': X_train_scaled,
        'X_test_scaled': X_test_scaled,
        'y_train': y_train,
        'y_test': y_test
    }

# ============================================================
# 3. KNN 模型（原始數據）- 跑所有 Seeds
# ============================================================
print("\n" + "=" * 80)
print("模型 1a: K-Nearest Neighbors (KNN) - 原始數據（不標準化）")
print("=" * 80)

knn_param_grid = {
    'n_neighbors': [3, 5, 7, 9, 11, 15, 20],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

for seed in SEEDS:
    print(f"\n{'─' * 80}")
    print(f"Random Seed = {seed}")
    print(f"{'─' * 80}")
    
    X_train = data_splits[seed]['X_train']
    X_test = data_splits[seed]['X_test']
    y_train = data_splits[seed]['y_train']
    y_test = data_splits[seed]['y_test']
    
    print(f"訓練集大小: {X_train.shape}, 測試集大小: {X_test.shape}")
    
    # GridSearchCV
    knn_grid = GridSearchCV(
        KNeighborsClassifier(),
        knn_param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )
    
    knn_grid.fit(X_train, y_train)
    knn_best_model = knn_grid.best_estimator_
    
    print(f"最佳參數: {knn_grid.best_params_}")
    print(f"最佳 CV 分數: {knn_grid.best_score_:.4f}")
    
    # 評估
    y_train_pred = knn_best_model.predict(X_train)
    y_test_pred = knn_best_model.predict(X_test)
    y_test_proba = knn_best_model.predict_proba(X_test)[:, 1]
    
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred)
    test_pre = precision_score(y_test, y_test_pred)
    test_rec = recall_score(y_test, y_test_pred)
    test_auc = roc_auc_score(y_test, y_test_proba)
    
    cm = confusion_matrix(y_test, y_test_pred)
    tn, fp, fn, tp = cm.ravel()
    test_spe = tn / (tn + fp)
    
    print(f"\n訓練集準確率: {train_acc:.4f} ({train_acc*100:.2f}%)")
    print(f"測試集準確率: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"測試集 F1-Score: {test_f1:.4f} ({test_f1*100:.2f}%)")
    print(f"測試集 Precision: {test_pre:.4f} ({test_pre*100:.2f}%)")
    print(f"測試集 Sensitivity: {test_rec:.4f} ({test_rec*100:.2f}%)")
    print(f"測試集 Specificity: {test_spe:.4f} ({test_spe*100:.2f}%)")
    print(f"測試集 AUC: {test_auc:.4f}")
    
    print(f"\n混淆矩陣:")
    print(f"  TN={tn}, FP={fp}")
    print(f"  FN={fn}, TP={tp}")
    
    all_results['KNN_原始'][seed] = {
        'train_acc': train_acc,
        'test_acc': test_acc,
        'test_f1': test_f1,
        'test_precision': test_pre,
        'test_recall': test_rec,
        'test_specificity': test_spe,
        'test_auc': test_auc,
        'best_params': knn_grid.best_params_,
        'confusion_matrix': cm,
        'y_test': y_test,
        'y_test_pred': y_test_pred,
        'y_test_proba': y_test_proba
    }

# KNN 原始數據結果彙整表格
print("\n" + "=" * 80)
print("KNN 原始數據結果彙整")
print("=" * 80)

knn_orig_summary = []
for seed in SEEDS:
    result = all_results['KNN_原始'][seed]
    knn_orig_summary.append({
        'Seed': seed,
        'Test_ACC (%)': result['test_acc'] * 100,
        'Test_F1 (%)': result['test_f1'] * 100,
        'Test_Precision (%)': result['test_precision'] * 100,
        'Test_Sensitivity (%)': result['test_recall'] * 100,
        'Test_Specificity (%)': result['test_specificity'] * 100,
        'Test_AUC': result['test_auc']
    })

knn_orig_avg_acc = np.mean([r['test_acc'] for r in all_results['KNN_原始'].values()])
knn_orig_avg_f1 = np.mean([r['test_f1'] for r in all_results['KNN_原始'].values()])

knn_orig_summary.append({
    'Seed': 'Average',
    'Test_ACC (%)': knn_orig_avg_acc * 100,
    'Test_F1 (%)': knn_orig_avg_f1 * 100,
    'Test_Precision (%)': np.mean([r['test_precision'] for r in all_results['KNN_原始'].values()]) * 100,
    'Test_Sensitivity (%)': np.mean([r['test_recall'] for r in all_results['KNN_原始'].values()]) * 100,
    'Test_Specificity (%)': np.mean([r['test_specificity'] for r in all_results['KNN_原始'].values()]) * 100,
    'Test_AUC': np.mean([r['test_auc'] for r in all_results['KNN_原始'].values()])
})

knn_orig_df = pd.DataFrame(knn_orig_summary)
print("\n" + knn_orig_df.to_string(index=False))

# ============================================================
# 4. KNN 模型（標準化數據）- 所有Seeds
# ============================================================
print("\n\n" + "=" * 80)
print("模型 1b: K-Nearest Neighbors (KNN) - 標準化數據")
print("=" * 80)

for seed in SEEDS:
    print(f"\n{'─' * 80}")
    print(f"Random Seed = {seed}")
    print(f"{'─' * 80}")
    
    X_train_scaled = data_splits[seed]['X_train_scaled']
    X_test_scaled = data_splits[seed]['X_test_scaled']
    y_train = data_splits[seed]['y_train']
    y_test = data_splits[seed]['y_test']
    
    print(f"訓練集大小: {X_train_scaled.shape}, 測試集大小: {X_test_scaled.shape}")
    
    # GridSearchCV
    knn_grid_scaled = GridSearchCV(
        KNeighborsClassifier(),
        knn_param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )
    
    knn_grid_scaled.fit(X_train_scaled, y_train)
    knn_best_model_scaled = knn_grid_scaled.best_estimator_
    
    print(f"最佳參數: {knn_grid_scaled.best_params_}")
    print(f"最佳 CV 分數: {knn_grid_scaled.best_score_:.4f}")
    
    # 評估
    y_train_pred = knn_best_model_scaled.predict(X_train_scaled)
    y_test_pred = knn_best_model_scaled.predict(X_test_scaled)
    y_test_proba = knn_best_model_scaled.predict_proba(X_test_scaled)[:, 1]
    
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred)
    test_pre = precision_score(y_test, y_test_pred)
    test_rec = recall_score(y_test, y_test_pred)
    test_auc = roc_auc_score(y_test, y_test_proba)
    
    cm = confusion_matrix(y_test, y_test_pred)
    tn, fp, fn, tp = cm.ravel()
    test_spe = tn / (tn + fp)
    
    print(f"\n訓練集準確率: {train_acc:.4f} ({train_acc*100:.2f}%)")
    print(f"測試集準確率: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"測試集 F1-Score: {test_f1:.4f} ({test_f1*100:.2f}%)")
    print(f"測試集 Precision: {test_pre:.4f} ({test_pre*100:.2f}%)")
    print(f"測試集 Sensitivity: {test_rec:.4f} ({test_rec*100:.2f}%)")
    print(f"測試集 Specificity: {test_spe:.4f} ({test_spe*100:.2f}%)")
    print(f"測試集 AUC: {test_auc:.4f}")
    
    print(f"\n混淆矩陣:")
    print(f"  TN={tn}, FP={fp}")
    print(f"  FN={fn}, TP={tp}")
    
    # 計算與原始數據的改進幅度
    orig_acc = all_results['KNN_原始'][seed]['test_acc']
    improvement = (test_acc - orig_acc) * 100
    print(f"\n 相比原始數據，準確率提升: {improvement:+.2f}%")
    
    all_results['KNN_標準化'][seed] = {
        'train_acc': train_acc,
        'test_acc': test_acc,
        'test_f1': test_f1,
        'test_precision': test_pre,
        'test_recall': test_rec,
        'test_specificity': test_spe,
        'test_auc': test_auc,
        'best_params': knn_grid_scaled.best_params_,
        'confusion_matrix': cm,
        'y_test': y_test,
        'y_test_pred': y_test_pred,
        'y_test_proba': y_test_proba
    }

# KNN 標準化數據結果彙整表格
print("\n" + "=" * 80)
print("KNN 標準化數據結果彙整")
print("=" * 80)

knn_scaled_summary = []
for seed in SEEDS:
    result = all_results['KNN_標準化'][seed]
    knn_scaled_summary.append({
        'Seed': seed,
        'Test_ACC (%)': result['test_acc'] * 100,
        'Test_F1 (%)': result['test_f1'] * 100,
        'Test_Precision (%)': result['test_precision'] * 100,
        'Test_Sensitivity (%)': result['test_recall'] * 100,
        'Test_Specificity (%)': result['test_specificity'] * 100,
        'Test_AUC': result['test_auc']
    })

knn_scaled_avg_acc = np.mean([r['test_acc'] for r in all_results['KNN_標準化'].values()])
knn_scaled_avg_f1 = np.mean([r['test_f1'] for r in all_results['KNN_標準化'].values()])

knn_scaled_summary.append({
    'Seed': 'Average',
    'Test_ACC (%)': knn_scaled_avg_acc * 100,
    'Test_F1 (%)': knn_scaled_avg_f1 * 100,
    'Test_Precision (%)': np.mean([r['test_precision'] for r in all_results['KNN_標準化'].values()]) * 100,
    'Test_Sensitivity (%)': np.mean([r['test_recall'] for r in all_results['KNN_標準化'].values()]) * 100,
    'Test_Specificity (%)': np.mean([r['test_specificity'] for r in all_results['KNN_標準化'].values()]) * 100,
    'Test_AUC': np.mean([r['test_auc'] for r in all_results['KNN_標準化'].values()])
})

knn_scaled_df = pd.DataFrame(knn_scaled_summary)
print("\n" + knn_scaled_df.to_string(index=False))

# KNN 標準化效果分析
print("\n" + "=" * 80)
print(" KNN 標準化效果分析")
print("=" * 80)

improvement_avg = knn_scaled_avg_acc - knn_orig_avg_acc
print(f"\nKNN 原始數據平均準確率: {knn_orig_avg_acc*100:.2f}%")
print(f"KNN 標準化數據平均準確率: {knn_scaled_avg_acc*100:.2f}%")
print(f"平均提升幅度: {improvement_avg*100:+.2f}%")

print("\n各 Seed 的提升幅度:")
for seed in SEEDS:
    orig = all_results['KNN_原始'][seed]['test_acc'] * 100
    scaled = all_results['KNN_標準化'][seed]['test_acc'] * 100
    print(f"  Seed {seed}: {orig:.2f}% → {scaled:.2f}% (提升 {scaled-orig:+.2f}%)")

# ============================================================
# 5. CART 模型 - 跑所有 Seeds
# ============================================================
print("\n\n" + "=" * 80)
print("模型 2: CART (Classification and Regression Trees)")
print("=" * 80)

cart_param_grid = {
    'criterion': ['gini'],
    'max_depth': [3, 4, 5, 6, 7],
    'min_samples_leaf': [1, 2, 5, 10, 15],
    'min_samples_split': [2, 5, 10]
}

for seed in SEEDS:
    print(f"\n{'─' * 80}")
    print(f"Random Seed = {seed}")
    print(f"{'─' * 80}")
    
    X_train = data_splits[seed]['X_train']
    X_test = data_splits[seed]['X_test']
    y_train = data_splits[seed]['y_train']
    y_test = data_splits[seed]['y_test']
    
    print(f"訓練集大小: {X_train.shape}, 測試集大小: {X_test.shape}")
    
    # GridSearchCV
    cart_grid = GridSearchCV(
        DecisionTreeClassifier(random_state=seed),
        cart_param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )
    
    cart_grid.fit(X_train, y_train)
    cart_best_model = cart_grid.best_estimator_
    
    print(f"最佳參數: {cart_grid.best_params_}")
    print(f"最佳 CV 分數: {cart_grid.best_score_:.4f}")
    
    # 評估
    y_train_pred = cart_best_model.predict(X_train)
    y_test_pred = cart_best_model.predict(X_test)
    y_test_proba = cart_best_model.predict_proba(X_test)[:, 1]
    
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred)
    test_pre = precision_score(y_test, y_test_pred)
    test_rec = recall_score(y_test, y_test_pred)
    test_auc = roc_auc_score(y_test, y_test_proba)
    
    cm = confusion_matrix(y_test, y_test_pred)
    tn, fp, fn, tp = cm.ravel()
    test_spe = tn / (tn + fp)
    
    print(f"\n訓練集準確率: {train_acc:.4f} ({train_acc*100:.2f}%)")
    print(f"測試集準確率: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"測試集 F1-Score: {test_f1:.4f} ({test_f1*100:.2f}%)")
    print(f"測試集 Precision: {test_pre:.4f} ({test_pre*100:.2f}%)")
    print(f"測試集 Sensitivity: {test_rec:.4f} ({test_rec*100:.2f}%)")
    print(f"測試集 Specificity: {test_spe:.4f} ({test_spe*100:.2f}%)")
    print(f"測試集 AUC: {test_auc:.4f}")
    
    print(f"\n混淆矩陣:")
    print(f"  TN={tn}, FP={fp}")
    print(f"  FN={fn}, TP={tp}")
    
    # 特徵重要性
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': cart_best_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print(f"\n特徵重要性 (Top 5):")
    print(feature_importance.head().to_string(index=False))
    
    all_results['CART'][seed] = {
        'train_acc': train_acc,
        'test_acc': test_acc,
        'test_f1': test_f1,
        'test_precision': test_pre,
        'test_recall': test_rec,
        'test_specificity': test_spe,
        'test_auc': test_auc,
        'best_params': cart_grid.best_params_,
        'feature_importance': feature_importance,
        'confusion_matrix': cm,
        'model': cart_best_model,
        'y_test': y_test,
        'y_test_pred': y_test_pred,
        'y_test_proba': y_test_proba
    }

# CART 結果彙整表格
print("\n" + "=" * 80)
print("CART 模型結果彙整")
print("=" * 80)

cart_summary = []
for seed in SEEDS:
    result = all_results['CART'][seed]
    cart_summary.append({
        'Seed': seed,
        'Test_ACC (%)': result['test_acc'] * 100,
        'Test_F1 (%)': result['test_f1'] * 100,
        'Test_Precision (%)': result['test_precision'] * 100,
        'Test_Sensitivity (%)': result['test_recall'] * 100,
        'Test_Specificity (%)': result['test_specificity'] * 100,
        'Test_AUC': result['test_auc']
    })

cart_avg_acc = np.mean([r['test_acc'] for r in all_results['CART'].values()])
cart_avg_f1 = np.mean([r['test_f1'] for r in all_results['CART'].values()])

cart_summary.append({
    'Seed': 'Average',
    'Test_ACC (%)': cart_avg_acc * 100,
    'Test_F1 (%)': cart_avg_f1 * 100,
    'Test_Precision (%)': np.mean([r['test_precision'] for r in all_results['CART'].values()]) * 100,
    'Test_Sensitivity (%)': np.mean([r['test_recall'] for r in all_results['CART'].values()]) * 100,
    'Test_Specificity (%)': np.mean([r['test_specificity'] for r in all_results['CART'].values()]) * 100,
    'Test_AUC': np.mean([r['test_auc'] for r in all_results['CART'].values()])
})

cart_df = pd.DataFrame(cart_summary)
print("\n" + cart_df.to_string(index=False))

# ============================================================
# 6. C4.5 模型 - 跑所有 Seeds
# ============================================================
print("\n\n" + "=" * 80)
print("模型 3: C4.5 Decision Tree")
print("=" * 80)

c45_param_grid = {
    'criterion': ['entropy'],
    'max_depth': [3, 4, 5, 6, 7],
    'min_samples_leaf': [1, 2, 5, 10, 15],
    'min_samples_split': [2, 5, 10]
}

for seed in SEEDS:
    print(f"\n{'─' * 80}")
    print(f"Random Seed = {seed}")
    print(f"{'─' * 80}")
    
    X_train = data_splits[seed]['X_train']
    X_test = data_splits[seed]['X_test']
    y_train = data_splits[seed]['y_train']
    y_test = data_splits[seed]['y_test']
    
    print(f"訓練集大小: {X_train.shape}, 測試集大小: {X_test.shape}")
    
    # GridSearchCV
    c45_grid = GridSearchCV(
        DecisionTreeClassifier(random_state=seed),
        c45_param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )
    
    c45_grid.fit(X_train, y_train)
    c45_best_model = c45_grid.best_estimator_
    
    print(f"最佳參數: {c45_grid.best_params_}")
    print(f"最佳 CV 分數: {c45_grid.best_score_:.4f}")
    
    # 評估
    y_train_pred = c45_best_model.predict(X_train)
    y_test_pred = c45_best_model.predict(X_test)
    y_test_proba = c45_best_model.predict_proba(X_test)[:, 1]
    
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred)
    test_pre = precision_score(y_test, y_test_pred)
    test_rec = recall_score(y_test, y_test_pred)
    test_auc = roc_auc_score(y_test, y_test_proba)
    
    cm = confusion_matrix(y_test, y_test_pred)
    tn, fp, fn, tp = cm.ravel()
    test_spe = tn / (tn + fp)
    
    print(f"\n訓練集準確率: {train_acc:.4f} ({train_acc*100:.2f}%)")
    print(f"測試集準確率: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"測試集 F1-Score: {test_f1:.4f} ({test_f1*100:.2f}%)")
    print(f"測試集 Precision: {test_pre:.4f} ({test_pre*100:.2f}%)")
    print(f"測試集 Sensitivity: {test_rec:.4f} ({test_rec*100:.2f}%)")
    print(f"測試集 Specificity: {test_spe:.4f} ({test_spe*100:.2f}%)")
    print(f"測試集 AUC: {test_auc:.4f}")
    
    print(f"\n混淆矩陣:")
    print(f"  TN={tn}, FP={fp}")
    print(f"  FN={fn}, TP={tp}")
    
    # 特徵重要性
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': c45_best_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print(f"\n特徵重要性 (Top 5):")
    print(feature_importance.head().to_string(index=False))
    
    all_results['C4.5'][seed] = {
        'train_acc': train_acc,
        'test_acc': test_acc,
        'test_f1': test_f1,
        'test_precision': test_pre,
        'test_recall': test_rec,
        'test_specificity': test_spe,
        'test_auc': test_auc,
        'best_params': c45_grid.best_params_,
        'feature_importance': feature_importance,
        'confusion_matrix': cm,
        'model': c45_best_model,
        'y_test': y_test,
        'y_test_pred': y_test_pred,
        'y_test_proba': y_test_proba
    }

# C4.5 結果彙整表格
print("\n" + "=" * 80)
print("C4.5 模型結果彙整")
print("=" * 80)

c45_summary = []
for seed in SEEDS:
    result = all_results['C4.5'][seed]
    c45_summary.append({
        'Seed': seed,
        'Test_ACC (%)': result['test_acc'] * 100,
        'Test_F1 (%)': result['test_f1'] * 100,
        'Test_Precision (%)': result['test_precision'] * 100,
        'Test_Sensitivity (%)': result['test_recall'] * 100,
        'Test_Specificity (%)': result['test_specificity'] * 100,
        'Test_AUC': result['test_auc']
    })

c45_avg_acc = np.mean([r['test_acc'] for r in all_results['C4.5'].values()])
c45_avg_f1 = np.mean([r['test_f1'] for r in all_results['C4.5'].values()])

c45_summary.append({
    'Seed': 'Average',
    'Test_ACC (%)': c45_avg_acc * 100,
    'Test_F1 (%)': c45_avg_f1 * 100,
    'Test_Precision (%)': np.mean([r['test_precision'] for r in all_results['C4.5'].values()]) * 100,
    'Test_Sensitivity (%)': np.mean([r['test_recall'] for r in all_results['C4.5'].values()]) * 100,
    'Test_Specificity (%)': np.mean([r['test_specificity'] for r in all_results['C4.5'].values()]) * 100,
    'Test_AUC': np.mean([r['test_auc'] for r in all_results['C4.5'].values()])
})

c45_df = pd.DataFrame(c45_summary)
print("\n" + c45_df.to_string(index=False))

# ============================================================
# 7. 四個模型版本綜合比較
# ============================================================
print("\n\n" + "=" * 80)
print("四個模型版本綜合比較（KNN_原始、KNN_標準化、CART、C4.5）")
print("=" * 80)

# 建立綜合比較表格
comparison_summary = []
for model_name in ['KNN_原始', 'KNN_標準化', 'CART', 'C4.5']:
    for seed in SEEDS:
        result = all_results[model_name][seed]
        comparison_summary.append({
            'Model': model_name,
            'Seed': seed,
            'Test_ACC (%)': result['test_acc'] * 100,
            'Test_F1 (%)': result['test_f1'] * 100,
            'Test_Precision (%)': result['test_precision'] * 100,
            'Test_Sensitivity (%)': result['test_recall'] * 100,
            'Test_Specificity (%)': result['test_specificity'] * 100
        })

comparison_df = pd.DataFrame(comparison_summary)

# 添加平均值
avg_comparison = []
for model_name in ['KNN_原始', 'KNN_標準化', 'CART', 'C4.5']:
    model_results = comparison_df[comparison_df['Model'] == model_name]
    avg_comparison.append({
        'Model': model_name,
        'Seed': 'Average',
        'Test_ACC (%)': model_results['Test_ACC (%)'].mean(),
        'Test_F1 (%)': model_results['Test_F1 (%)'].mean(),
        'Test_Precision (%)': model_results['Test_Precision (%)'].mean(),
        'Test_Sensitivity (%)': model_results['Test_Sensitivity (%)'].mean(),
        'Test_Specificity (%)': model_results['Test_Specificity (%)'].mean()
    })

avg_comparison_df = pd.DataFrame(avg_comparison)
full_comparison_df = pd.concat([comparison_df, avg_comparison_df], ignore_index=True)

print("\n完整比較表格:")
print(full_comparison_df.to_string(index=False))

# 儲存 CSV
full_comparison_df.to_csv('complete_results_all_seeds.csv', 
                          index=False, encoding='utf-8-sig')
print("\n✓ 完整結果已儲存: complete_results_all_seeds.csv")

# ============================================================
# 8. 最終總結
# ============================================================
print("\n" + "=" * 80)
print("最終總結")
print("=" * 80)

print("\n各模型平均測試集準確率:")
avg_accs = {}
for model_name in ['KNN_原始', 'KNN_標準化', 'CART', 'C4.5']:
    avg_acc = np.mean([all_results[model_name][s]['test_acc'] for s in SEEDS])
    avg_accs[model_name] = avg_acc
    print(f"  {model_name}: {avg_acc*100:.2f}%")

best_model = max(avg_accs, key=avg_accs.get)
print(f"\n平均測試集準確率最佳模型: {best_model} ({avg_accs[best_model]*100:.2f}%)")

print("\n重要發現:")
print(f"1. KNN 標準化效果: 從 {avg_accs['KNN_原始']*100:.2f}% 提升到 {avg_accs['KNN_標準化']*100:.2f}% (提升 {(avg_accs['KNN_標準化']-avg_accs['KNN_原始'])*100:+.2f}%)")
print(f"2. 決策樹方法（CART & C4.5）不受特徵尺度影響，表現穩定")
print(f"3. 標準化後的 KNN {'已接近' if avg_accs['KNN_標準化'] > 0.70 else '仍低於'} 決策樹的表現")

print("\n" + "=" * 80)
print("Assignment 4 完成!")
print("=" * 80)