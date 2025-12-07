"""
步驟2: 模型訓練
直接載入處理好的資料，快速訓練模型
"""

import pandas as pd
import numpy as np
import pickle
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
print("步驟2: 模型訓練 (使用已處理的資料)".center(80))
print("="*80)

# ============================================================================
# 1. 載入處理好的資料
# ============================================================================
print("\n[1] 載入處理好的資料")

try:
    # 方式1: 從pickle載入 (推薦，更快)
    with open('processed_data.pkl', 'rb') as f:
        data = pickle.load(f)
    
    X = data['X']
    y = data['y']
    
    print("✓ 從 processed_data.pkl 載入成功")
    print(f"✓ 特徵數: {X.shape[1]}, 樣本數: {X.shape[0]}")
    print(f"✓ 目標分布: {data['class_counts']}")
    
except FileNotFoundError:
    # 方式2: 從CSV載入 (備用)
    print("找不到 processed_data.pkl，嘗試從CSV載入...")
    X = pd.read_csv('X_processed.csv')
    y = pd.read_csv('y_processed.csv')['Productivity_Category']
    
    print("✓ 從 CSV 載入成功")
    print(f"✓ 特徵數: {X.shape[1]}, 樣本數: {X.shape[0]}")

# ============================================================================
# 2. 定義評估函數
# ============================================================================
print("\n[2] 準備訓練")

# 定義F1 scorer
f1_high_scorer = make_scorer(f1_score, pos_label='High', average='binary')

# ============================================================================
# 3. 模型訓練 (多個random seeds)
# ============================================================================
print("\n[3] 開始訓練模型")
print("="*80)

random_seeds = [4, 40, 400]
results = []

for seed in random_seeds:
    print(f"\n{'─'*80}")
    print(f"Random Seed: {seed}".center(80))
    print(f"{'─'*80}")
    
    # 資料分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    
    print(f"訓練集: {X_train.shape[0]}, 測試集: {X_test.shape[0]}")
    
    # ========================================================================
    # CART
    # ========================================================================
    print("\n[CART - class_weight='balanced']")
    
    cart = GridSearchCV(
        DecisionTreeClassifier(random_state=seed, class_weight='balanced'),
        {
            'max_depth': [3, 5, 7, 10, 15],
            'min_samples_leaf': [2, 4, 8, 16],
            'min_samples_split': [2, 5, 10],
            'criterion': ['gini', 'entropy']
        },
        cv=5, scoring=f1_high_scorer, n_jobs=-1, verbose=0
    )
    
    cart.fit(X_train, y_train)
    cart_pred = cart.predict(X_test)
    
    cart_results = {
        'Model': 'CART', 'Seed': seed,
        'Accuracy': accuracy_score(y_test, cart_pred),
        'Precision': precision_score(y_test, cart_pred, pos_label='High', zero_division=0),
        'Recall': recall_score(y_test, cart_pred, pos_label='High', zero_division=0),
        'Specificity': recall_score(y_test, cart_pred, pos_label='Not_High', zero_division=0),
        'F1': f1_score(y_test, cart_pred, pos_label='High', zero_division=0),
        'Best_Params': str(cart.best_params_)
    }
    results.append(cart_results)
    
    print(f"最佳參數: {cart.best_params_}")
    print(f"CV F1(High): {cart.best_score_:.4f}")
    print(f"測試集 - Acc:{cart_results['Accuracy']:.4f}, F1:{cart_results['F1']:.4f}, "
          f"Recall:{cart_results['Recall']:.4f}, Precision:{cart_results['Precision']:.4f}")
    
    # ========================================================================
    # SVM
    # ========================================================================
    print("\n[SVM - 標準化 + class_weight='balanced']")
    
    # 處理可能的NaN/inf值
    from sklearn.impute import SimpleImputer
    
    X_train_svm_clean = X_train.replace([np.inf, -np.inf], np.nan)
    X_test_svm_clean = X_test.replace([np.inf, -np.inf], np.nan)
    
    # 檢查NaN
    if X_train_svm_clean.isna().any().any():
        print("  ⚠️ 發現NaN值，使用中位數填補")
        imputer = SimpleImputer(strategy='median')
        X_train_svm_clean = pd.DataFrame(imputer.fit_transform(X_train_svm_clean), 
                                          columns=X_train.columns, index=X_train.index)
        X_test_svm_clean = pd.DataFrame(imputer.transform(X_test_svm_clean), 
                                         columns=X_test.columns, index=X_test.index)
    
    # 標準化
    scaler_svm = StandardScaler()
    X_train_svm = scaler_svm.fit_transform(X_train_svm_clean)
    X_test_svm = scaler_svm.transform(X_test_svm_clean)
    
    svm = GridSearchCV(
        SVC(random_state=seed, class_weight='balanced'),
        {
            'C': [0.1, 1, 10, 100],
            'kernel': ['linear', 'rbf'],
            'gamma': ['scale', 'auto']
        },
        cv=5, scoring=f1_high_scorer, n_jobs=-1, verbose=0
    )
    
    svm.fit(X_train_svm, y_train)
    svm_pred = svm.predict(X_test_svm)
    
    svm_results = {
        'Model': 'SVM', 'Seed': seed,
        'Accuracy': accuracy_score(y_test, svm_pred),
        'Precision': precision_score(y_test, svm_pred, pos_label='High', zero_division=0),
        'Recall': recall_score(y_test, svm_pred, pos_label='High', zero_division=0),
        'Specificity': recall_score(y_test, svm_pred, pos_label='Not_High', zero_division=0),
        'F1': f1_score(y_test, svm_pred, pos_label='High', zero_division=0),
        'Best_Params': str(svm.best_params_)
    }
    results.append(svm_results)
    
    print(f"最佳參數: {svm.best_params_}")
    print(f"CV F1(High): {svm.best_score_:.4f}")
    print(f"測試集 - Acc:{svm_results['Accuracy']:.4f}, F1:{svm_results['F1']:.4f}, "
          f"Recall:{svm_results['Recall']:.4f}, Precision:{svm_results['Precision']:.4f}")
    
    # ========================================================================
    # KNN
    # ========================================================================
    print("\n[KNN - 標準化 ★作業4經驗★]")
    
    # 處理可能的NaN/inf值
    X_train_knn_clean = X_train.replace([np.inf, -np.inf], np.nan)
    X_test_knn_clean = X_test.replace([np.inf, -np.inf], np.nan)
    
    # 檢查NaN
    if X_train_knn_clean.isna().any().any():
        print("  ⚠️ 發現NaN值，使用中位數填補")
        imputer = SimpleImputer(strategy='median')
        X_train_knn_clean = pd.DataFrame(imputer.fit_transform(X_train_knn_clean), 
                                          columns=X_train.columns, index=X_train.index)
        X_test_knn_clean = pd.DataFrame(imputer.transform(X_test_knn_clean), 
                                         columns=X_test.columns, index=X_test.index)
    
    # 標準化
    scaler_knn = StandardScaler()
    X_train_knn = scaler_knn.fit_transform(X_train_knn_clean)
    X_test_knn = scaler_knn.transform(X_test_knn_clean)
    
    knn = GridSearchCV(
        KNeighborsClassifier(),
        {
            'n_neighbors': [3, 5, 7, 9, 11, 15, 20, 25],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan']
        },
        cv=5, scoring=f1_high_scorer, n_jobs=-1, verbose=0
    )
    
    knn.fit(X_train_knn, y_train)
    knn_pred = knn.predict(X_test_knn)
    
    knn_results = {
        'Model': 'KNN', 'Seed': seed,
        'Accuracy': accuracy_score(y_test, knn_pred),
        'Precision': precision_score(y_test, knn_pred, pos_label='High', zero_division=0),
        'Recall': recall_score(y_test, knn_pred, pos_label='High', zero_division=0),
        'Specificity': recall_score(y_test, knn_pred, pos_label='Not_High', zero_division=0),
        'F1': f1_score(y_test, knn_pred, pos_label='High', zero_division=0),
        'Best_Params': str(knn.best_params_)
    }
    results.append(knn_results)
    
    print(f"最佳參數: {knn.best_params_}")
    print(f"CV F1(High): {knn.best_score_:.4f}")
    print(f"測試集 - Acc:{knn_results['Accuracy']:.4f}, F1:{knn_results['F1']:.4f}, "
          f"Recall:{knn_results['Recall']:.4f}, Precision:{knn_results['Precision']:.4f}")
    
    # ========================================================================
    # 混淆矩陣
    # ========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for idx, (pred, name) in enumerate([(cart_pred,'CART'), (svm_pred,'SVM'), (knn_pred,'KNN')]):
        cm = confusion_matrix(y_test, pred, labels=['Not_High', 'High'])
        
        # 計算百分比
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        # 顯示數量和百分比
        annot = np.empty_like(cm).astype(str)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                annot[i, j] = f'{cm[i,j]}\n({cm_percent[i,j]:.1f}%)'
        
        sns.heatmap(cm, annot=annot, fmt='', cmap='Blues', ax=axes[idx],
                   xticklabels=['Not_High', 'High'],
                   yticklabels=['Not_High', 'High'])
        axes[idx].set_title(f'{name} (Seed={seed})', fontweight='bold')
        axes[idx].set_ylabel('實際')
        axes[idx].set_xlabel('預測')
    
    plt.tight_layout()
    plt.savefig(f'confusion_matrices_seed_{seed}.png', dpi=300)
    plt.close()
    print(f"\n✓ 混淆矩陣已儲存: confusion_matrices_seed_{seed}.png")

# ============================================================================
# 4. 結果彙整
# ============================================================================
print(f"\n{'='*80}")
print("結果彙整".center(80))
print(f"{'='*80}")

results_df = pd.DataFrame(results)

# 平均
avg_results = results_df.groupby('Model')[['Accuracy', 'Precision', 'Recall', 
                                            'Specificity', 'F1']].mean()
# 標準差
std_results = results_df.groupby('Model')[['Accuracy', 'Precision', 'Recall', 
                                           'Specificity', 'F1']].std()

print("\n平均性能 (跨3個seeds):")
print(avg_results.round(4))

print("\n標準差:")
print(std_results.round(4))

# 儲存
results_df.to_csv('modeling_results_detailed.csv', index=False, encoding='utf-8-sig')
avg_results.to_csv('modeling_results_average.csv', encoding='utf-8-sig')
std_results.to_csv('modeling_results_std.csv', encoding='utf-8-sig')

print("\n✓ 結果已儲存:")
print("  - modeling_results_detailed.csv")
print("  - modeling_results_average.csv")
print("  - modeling_results_std.csv")

# ============================================================================
# 5. 視覺化比較
# ============================================================================
print("\n[5] 生成模型比較圖")

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.ravel()

metrics = ['Accuracy', 'Precision', 'Recall', 'Specificity', 'F1']
metric_names = ['準確率', '精確率(High)', '召回率(High)', '特異度(Not_High)', 'F1分數']
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

for idx, (metric, name) in enumerate(zip(metrics, metric_names)):
    ax = axes[idx]
    
    avg_results[metric].plot(kind='bar', ax=ax, color=colors, 
                            edgecolor='black', linewidth=1.5)
    ax.set_title(f'{name}', fontsize=12, fontweight='bold')
    ax.set_ylabel(name, fontsize=10)
    ax.set_xlabel('模型', fontsize=10)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim([0, 1.0])
    
    # 添加數值標籤
    for i, v in enumerate(avg_results[metric]):
        ax.text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom', 
               fontsize=9, fontweight='bold')
    
    # 添加標準差error bars
    if metric in std_results.columns:
        ax.errorbar(range(len(avg_results)), avg_results[metric], 
                   yerr=std_results[metric], fmt='none', 
                   ecolor='black', capsize=5, linewidth=1.5)

axes[5].axis('off')

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

print("✓ 模型比較圖已儲存: model_comparison.png")

# ============================================================================
# 6. 總結
# ============================================================================
print(f"\n{'='*80}")
print("訓練完成！".center(80))
print(f"{'='*80}")

best_model = avg_results['F1'].idxmax()
best_f1 = avg_results.loc[best_model, 'F1']
best_recall = avg_results.loc[best_model, 'Recall']

print(f"\n✓ 最佳模型: {best_model}")
print(f"✓ F1-Score: {best_f1:.4f}")
print(f"✓ Recall(High): {best_recall:.4f} (能找出{best_recall*100:.1f}%的高績效員工)")

print("\n✓ 生成的檔案:")
print("  1. confusion_matrices_seed_*.png (3個)")
print("  2. model_comparison.png")
print("  3. modeling_results_detailed.csv")
print("  4. modeling_results_average.csv")
print("  5. modeling_results_std.csv")

print("\n✓ 優點:")
print("  • 不需要重新前處理資料")
print("  • 可以快速嘗試不同參數")
print("  • 結果自動儲存")

print("\n如果要調整參數:")
print("  → 直接修改本檔案的GridSearchCV參數")
print("  → 重新執行即可，超快！")