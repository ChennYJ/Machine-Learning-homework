# Assignment 4 - KNN、CART、C4.5 完整比較

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
import seaborn as sns

# 資料預處理
from sklearn.preprocessing import StandardScaler

# 模型相關
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree

# 評估指標
from sklearn.metrics import (accuracy_score, f1_score, precision_score, 
                             recall_score, roc_auc_score, confusion_matrix,
                             classification_report, roc_curve)

# 設定中文字型
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Microsoft JhengHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 80)
print("Assignment 4: KNN、CART、C4.5 模型比較分析（改進版 - 加入標準化）")
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
# ★ 新增：檢查特徵的數值範圍
# ============================================================
print("\n" + "=" * 80)
print("特徵數值範圍分析")
print("=" * 80)
print("\n未標準化前的特徵統計:")
print(X.describe().loc[['min', 'max', 'mean', 'std']])

# ============================================================
# 2. 定義三個 Random Seeds
# ============================================================
SEEDS = [4, 40, 400]
print(f"\n使用的 Random Seeds: {SEEDS}")

# 儲存所有結果
all_results = {
    'KNN_原始': {},      # 不標準化
    'KNN_標準化': {},    # 標準化
    'CART': {},
    'C4.5': {}
}

# ============================================================
# 3. 對每個 Seed 進行實驗
# ============================================================

for seed in SEEDS:
    print("\n" + "=" * 80)
    print(f"Random Seed = {seed}")
    print("=" * 80)
    
    # 資料分割 (80% 訓練, 20% 測試)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    
    print(f"訓練集大小: {X_train.shape}")
    print(f"測試集大小: {X_test.shape}")
    
    # ========================================
    # ★ 標準化處理（只針對 KNN）
    # ========================================
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 轉回 DataFrame 以保留欄位名稱（方便檢查）
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X.columns, index=X_train.index)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X.columns, index=X_test.index)
    
    print("\n標準化後的特徵統計 (前 5 個特徵):")
    print(X_train_scaled_df.iloc[:, :5].describe().loc[['mean', 'std']])
    
    # ========================================
    # KNN 模型 - 原始數據（不標準化）
    # ========================================
    print("\n[模型 1a] K-Nearest Neighbors (原始數據 - 不標準化)")
    print("-" * 40)
    
    knn_param_grid = {
        'n_neighbors': [3, 5, 7, 9, 11, 15, 20],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }
    
    knn_grid_original = GridSearchCV(
        KNeighborsClassifier(),
        knn_param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )
    
    knn_grid_original.fit(X_train, y_train)
    knn_best_model_original = knn_grid_original.best_estimator_
    
    print(f"最佳參數: {knn_grid_original.best_params_}")
    
    # 評估
    y_train_pred_knn_orig = knn_best_model_original.predict(X_train)
    y_test_pred_knn_orig = knn_best_model_original.predict(X_test)
    y_test_proba_knn_orig = knn_best_model_original.predict_proba(X_test)[:, 1]
    
    knn_orig_train_acc = accuracy_score(y_train, y_train_pred_knn_orig)
    knn_orig_test_acc = accuracy_score(y_test, y_test_pred_knn_orig)
    knn_orig_test_f1 = f1_score(y_test, y_test_pred_knn_orig)
    knn_orig_test_pre = precision_score(y_test, y_test_pred_knn_orig)
    knn_orig_test_rec = recall_score(y_test, y_test_pred_knn_orig)
    knn_orig_test_auc = roc_auc_score(y_test, y_test_proba_knn_orig)
    
    cm_knn_orig = confusion_matrix(y_test, y_test_pred_knn_orig)
    tn, fp, fn, tp = cm_knn_orig.ravel()
    knn_orig_test_spe = tn / (tn + fp)
    
    print(f"測試集準確率: {knn_orig_test_acc:.4f} ({knn_orig_test_acc*100:.2f}%)")
    print(f"測試集 F1-Score: {knn_orig_test_f1:.4f} ({knn_orig_test_f1*100:.2f}%)")
    print(f"測試集 AUC: {knn_orig_test_auc:.4f}")
    
    all_results['KNN_原始'][seed] = {
        'train_acc': knn_orig_train_acc,
        'test_acc': knn_orig_test_acc,
        'test_f1': knn_orig_test_f1,
        'test_precision': knn_orig_test_pre,
        'test_recall': knn_orig_test_rec,
        'test_specificity': knn_orig_test_spe,
        'test_auc': knn_orig_test_auc,
        'best_params': knn_grid_original.best_params_,
        'confusion_matrix': cm_knn_orig
    }
    
    # ========================================
    # KNN 模型 - 標準化數據
    # ========================================
    print("\n[模型 1b] K-Nearest Neighbors (標準化數據) ★")
    print("-" * 40)
    
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
    
    # 評估
    y_train_pred_knn_scaled = knn_best_model_scaled.predict(X_train_scaled)
    y_test_pred_knn_scaled = knn_best_model_scaled.predict(X_test_scaled)
    y_test_proba_knn_scaled = knn_best_model_scaled.predict_proba(X_test_scaled)[:, 1]
    
    knn_scaled_train_acc = accuracy_score(y_train, y_train_pred_knn_scaled)
    knn_scaled_test_acc = accuracy_score(y_test, y_test_pred_knn_scaled)
    knn_scaled_test_f1 = f1_score(y_test, y_test_pred_knn_scaled)
    knn_scaled_test_pre = precision_score(y_test, y_test_pred_knn_scaled)
    knn_scaled_test_rec = recall_score(y_test, y_test_pred_knn_scaled)
    knn_scaled_test_auc = roc_auc_score(y_test, y_test_proba_knn_scaled)
    
    cm_knn_scaled = confusion_matrix(y_test, y_test_pred_knn_scaled)
    tn, fp, fn, tp = cm_knn_scaled.ravel()
    knn_scaled_test_spe = tn / (tn + fp)
    
    print(f"測試集準確率: {knn_scaled_test_acc:.4f} ({knn_scaled_test_acc*100:.2f}%)")
    print(f"測試集 F1-Score: {knn_scaled_test_f1:.4f} ({knn_scaled_test_f1*100:.2f}%)")
    print(f"測試集 AUC: {knn_scaled_test_auc:.4f}")
    
    # 計算改進幅度
    improvement = (knn_scaled_test_acc - knn_orig_test_acc) * 100
    print(f"\n★ 標準化後準確率提升: {improvement:+.2f}%")
    
    all_results['KNN_標準化'][seed] = {
        'train_acc': knn_scaled_train_acc,
        'test_acc': knn_scaled_test_acc,
        'test_f1': knn_scaled_test_f1,
        'test_precision': knn_scaled_test_pre,
        'test_recall': knn_scaled_test_rec,
        'test_specificity': knn_scaled_test_spe,
        'test_auc': knn_scaled_test_auc,
        'best_params': knn_grid_scaled.best_params_,
        'confusion_matrix': cm_knn_scaled,
        'y_test': y_test,
        'y_test_pred': y_test_pred_knn_scaled,
        'y_test_proba': y_test_proba_knn_scaled
    }
    
    # ========================================
    # CART 模型（不需要標準化）
    # ========================================
    print("\n[模型 2] CART (不需標準化)")
    print("-" * 40)
    
    cart_param_grid = {
        'criterion': ['gini'],
        'max_depth': [3, 4, 5, 6, 7],
        'min_samples_leaf': [1, 2, 5, 10, 15],
        'min_samples_split': [2, 5, 10]
    }
    
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
    
    # 評估
    y_train_pred_cart = cart_best_model.predict(X_train)
    y_test_pred_cart = cart_best_model.predict(X_test)
    y_test_proba_cart = cart_best_model.predict_proba(X_test)[:, 1]
    
    cart_train_acc = accuracy_score(y_train, y_train_pred_cart)
    cart_test_acc = accuracy_score(y_test, y_test_pred_cart)
    cart_test_f1 = f1_score(y_test, y_test_pred_cart)
    cart_test_pre = precision_score(y_test, y_test_pred_cart)
    cart_test_rec = recall_score(y_test, y_test_pred_cart)
    cart_test_auc = roc_auc_score(y_test, y_test_proba_cart)
    
    cm_cart = confusion_matrix(y_test, y_test_pred_cart)
    tn, fp, fn, tp = cm_cart.ravel()
    cart_test_spe = tn / (tn + fp)
    
    print(f"測試集準確率: {cart_test_acc:.4f} ({cart_test_acc*100:.2f}%)")
    print(f"測試集 F1-Score: {cart_test_f1:.4f} ({cart_test_f1*100:.2f}%)")
    
    # 特徵重要性
    feature_importance_cart = pd.DataFrame({
        'Feature': X.columns,
        'Importance': cart_best_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    all_results['CART'][seed] = {
        'train_acc': cart_train_acc,
        'test_acc': cart_test_acc,
        'test_f1': cart_test_f1,
        'test_precision': cart_test_pre,
        'test_recall': cart_test_rec,
        'test_specificity': cart_test_spe,
        'test_auc': cart_test_auc,
        'best_params': cart_grid.best_params_,
        'feature_importance': feature_importance_cart,
        'confusion_matrix': cm_cart,
        'model': cart_best_model
    }
    
    # ========================================
    # C4.5 模型（不需要標準化）
    # ========================================
    print("\n[模型 3] C4.5 (不需標準化)")
    print("-" * 40)
    
    c45_param_grid = {
        'criterion': ['entropy'],
        'max_depth': [3, 4, 5, 6, 7],
        'min_samples_leaf': [1, 2, 5, 10, 15],
        'min_samples_split': [2, 5, 10]
    }
    
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
    
    # 評估
    y_train_pred_c45 = c45_best_model.predict(X_train)
    y_test_pred_c45 = c45_best_model.predict(X_test)
    y_test_proba_c45 = c45_best_model.predict_proba(X_test)[:, 1]
    
    c45_train_acc = accuracy_score(y_train, y_train_pred_c45)
    c45_test_acc = accuracy_score(y_test, y_test_pred_c45)
    c45_test_f1 = f1_score(y_test, y_test_pred_c45)
    c45_test_pre = precision_score(y_test, y_test_pred_c45)
    c45_test_rec = recall_score(y_test, y_test_pred_c45)
    c45_test_auc = roc_auc_score(y_test, y_test_proba_c45)
    
    cm_c45 = confusion_matrix(y_test, y_test_pred_c45)
    tn, fp, fn, tp = cm_c45.ravel()
    c45_test_spe = tn / (tn + fp)
    
    print(f"測試集準確率: {c45_test_acc:.4f} ({c45_test_acc*100:.2f}%)")
    print(f"測試集 F1-Score: {c45_test_f1:.4f} ({c45_test_f1*100:.2f}%)")
    
    # 特徵重要性
    feature_importance_c45 = pd.DataFrame({
        'Feature': X.columns,
        'Importance': c45_best_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    all_results['C4.5'][seed] = {
        'train_acc': c45_train_acc,
        'test_acc': c45_test_acc,
        'test_f1': c45_test_f1,
        'test_precision': c45_test_pre,
        'test_recall': c45_test_rec,
        'test_specificity': c45_test_spe,
        'test_auc': c45_test_auc,
        'best_params': c45_grid.best_params_,
        'feature_importance': feature_importance_c45,
        'confusion_matrix': cm_c45,
        'model': c45_best_model
    }

# ============================================================
# 4. 跨 Seed 結果彙整
# ============================================================
print("\n" + "=" * 80)
print("跨 Seed 結果彙整")
print("=" * 80)

# 建立彙整表格
summary_data = []

for model_name in ['KNN_原始', 'KNN_標準化', 'CART', 'C4.5']:
    for seed in SEEDS:
        result = all_results[model_name][seed]
        summary_data.append({
            'Model': model_name,
            'Seed': seed,
            'Test_ACC': result['test_acc'] * 100,
            'Test_F1': result['test_f1'] * 100,
            'Test_AUC': result['test_auc'] * 100
        })

summary_df = pd.DataFrame(summary_data)

# 計算平均值
avg_data = []
for model_name in ['KNN_原始', 'KNN_標準化', 'CART', 'C4.5']:
    model_results = summary_df[summary_df['Model'] == model_name]
    avg_data.append({
        'Model': model_name,
        'Seed': 'Average',
        'Test_ACC': model_results['Test_ACC'].mean(),
        'Test_F1': model_results['Test_F1'].mean(),
        'Test_AUC': model_results['Test_AUC'].mean()
    })

avg_df = pd.DataFrame(avg_data)
full_summary_df = pd.concat([summary_df, avg_df], ignore_index=True)

print("\n完整結果表格:")
print(full_summary_df.to_string(index=False))

# ============================================================
# 5. 標準化效果分析
# ============================================================
print("\n" + "=" * 80)
print("★ 標準化效果分析")
print("=" * 80)

knn_orig_avg = avg_df[avg_df['Model'] == 'KNN_原始']['Test_ACC'].values[0]
knn_scaled_avg = avg_df[avg_df['Model'] == 'KNN_標準化']['Test_ACC'].values[0]
improvement_avg = knn_scaled_avg - knn_orig_avg

print(f"\nKNN 原始數據平均準確率: {knn_orig_avg:.2f}%")
print(f"KNN 標準化數據平均準確率: {knn_scaled_avg:.2f}%")
print(f"平均提升幅度: {improvement_avg:+.2f}%")

print("\n各 Seed 的提升幅度:")
for seed in SEEDS:
    orig = all_results['KNN_原始'][seed]['test_acc'] * 100
    scaled = all_results['KNN_標準化'][seed]['test_acc'] * 100
    print(f"  Seed {seed}: {orig:.2f}% → {scaled:.2f}% (提升 {scaled-orig:+.2f}%)")

# ============================================================
# 6. 視覺化比較
# ============================================================
print("\n" + "=" * 80)
print("產生視覺化圖表")
print("=" * 80)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 圖1: 平均準確率比較
ax1 = axes[0, 0]
models = ['KNN_原始', 'KNN_標準化', 'CART', 'C4.5']
avg_accs = [avg_df[avg_df['Model'] == m]['Test_ACC'].values[0] for m in models]
colors = ['lightcoral', 'lightgreen', 'skyblue', 'gold']
bars = ax1.bar(models, avg_accs, color=colors, alpha=0.8, edgecolor='black')
ax1.set_ylabel('Average Test Accuracy (%)', fontsize=12)
ax1.set_title('模型平均準確率比較', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3, axis='y')
ax1.set_ylim([60, 80])
# 在柱狀圖上標註數值
for bar in bars:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.2f}%',
             ha='center', va='bottom', fontsize=10)

# 圖2: KNN 標準化前後對比
ax2 = axes[0, 1]
x_pos = np.arange(len(SEEDS))
width = 0.35
orig_accs = [all_results['KNN_原始'][s]['test_acc'] * 100 for s in SEEDS]
scaled_accs = [all_results['KNN_標準化'][s]['test_acc'] * 100 for s in SEEDS]
ax2.bar(x_pos - width/2, orig_accs, width, label='原始數據', alpha=0.8, color='lightcoral')
ax2.bar(x_pos + width/2, scaled_accs, width, label='標準化數據', alpha=0.8, color='lightgreen')
ax2.set_xlabel('Random Seed', fontsize=12)
ax2.set_ylabel('Test Accuracy (%)', fontsize=12)
ax2.set_title('KNN 標準化前後對比', fontsize=14, fontweight='bold')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(SEEDS)
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

# 圖3: 所有模型跨 Seed 準確率
ax3 = axes[1, 0]
for model_name, color in zip(['KNN_原始', 'KNN_標準化', 'CART', 'C4.5'], 
                              ['lightcoral', 'lightgreen', 'skyblue', 'gold']):
    accs = [all_results[model_name][s]['test_acc'] * 100 for s in SEEDS]
    ax3.plot(SEEDS, accs, marker='o', linewidth=2, markersize=8, 
             label=model_name, color=color)
ax3.set_xlabel('Random Seed', fontsize=12)
ax3.set_ylabel('Test Accuracy (%)', fontsize=12)
ax3.set_title('所有模型跨 Seed 準確率變化', fontsize=14, fontweight='bold')
ax3.set_xticks(SEEDS)
ax3.legend()
ax3.grid(True, alpha=0.3)

# 圖4: F1-Score 和 AUC 比較
ax4 = axes[1, 1]
model_names = ['KNN\n原始', 'KNN\n標準化', 'CART', 'C4.5']
avg_f1s = [avg_df[avg_df['Model'] == m]['Test_F1'].values[0] 
           for m in ['KNN_原始', 'KNN_標準化', 'CART', 'C4.5']]
avg_aucs = [avg_df[avg_df['Model'] == m]['Test_AUC'].values[0] 
            for m in ['KNN_原始', 'KNN_標準化', 'CART', 'C4.5']]
x = np.arange(len(model_names))
width = 0.35
ax4.bar(x - width/2, avg_f1s, width, label='F1-Score', alpha=0.8)
ax4.bar(x + width/2, avg_aucs, width, label='AUC', alpha=0.8)
ax4.set_ylabel('Score (%)', fontsize=12)
ax4.set_title('F1-Score 與 AUC 比較', fontsize=14, fontweight='bold')
ax4.set_xticks(x)
ax4.set_xticklabels(model_names)
ax4.legend()
ax4.grid(True, alpha=0.3, axis='y')
ax4.set_ylim([50, 85])

plt.tight_layout()
plt.savefig('model_comparison_with_scaling.png', dpi=300, bbox_inches='tight')
print("\n✓ 視覺化圖表已儲存: model_comparison_with_scaling.png")
plt.close()

# ============================================================
# 7. 最終總結
# ============================================================
print("\n" + "=" * 80)
print("最終總結")
print("=" * 80)

print("\n各模型平均測試集準確率:")
for model_name in ['KNN_原始', 'KNN_標準化', 'CART', 'C4.5']:
    avg_acc = avg_df[avg_df['Model'] == model_name]['Test_ACC'].values[0]
    print(f"  {model_name}: {avg_acc:.2f}%")

print(f"\n★ 結論:")
print(f"1. KNN 在標準化後平均提升 {improvement_avg:.2f}%")
print(f"2. 標準化後的 KNN ({knn_scaled_avg:.2f}%) {'已接近' if knn_scaled_avg > 70 else '仍低於'} 決策樹的表現")
print(f"3. 特徵尺度差異確實是影響 KNN 性能的關鍵因素")

print("\n" + "=" * 80)
print("分析完成!")
print("=" * 80)