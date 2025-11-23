# Assignment 4 - KNN、CART、C4.5 完整比較

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
import seaborn as sns

# 資料預處理
# from sklearn.preprocessing import LabelEncoder 

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
print("Assignment 4: KNN、CART、C4.5 模型比較分析")
print("使用三個 Random Seeds: 4, 40, 400")
print("=" * 80)

# ============================================================
# 1. 資料載入與預處理
# ============================================================
print("\n[步驟 1] 資料載入與預處理")

df = pd.read_csv('new_df.csv')

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
# 檢查是否有 'Survived' 欄位或是 'Survived_yes'
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
    'KNN': {},
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
    # KNN 模型
    # ========================================
    print("\n[模型 1] K-Nearest Neighbors (KNN)")
    print("-" * 40)
    
    knn_param_grid = {
        'n_neighbors': [3, 5, 7, 9, 11, 15, 20],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }
    
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
    y_train_pred_knn = knn_best_model.predict(X_train)
    y_test_pred_knn = knn_best_model.predict(X_test)
    y_test_proba_knn = knn_best_model.predict_proba(X_test)[:, 1]
    
    knn_train_acc = accuracy_score(y_train, y_train_pred_knn)
    knn_test_acc = accuracy_score(y_test, y_test_pred_knn)
    knn_test_f1 = f1_score(y_test, y_test_pred_knn)
    knn_test_pre = precision_score(y_test, y_test_pred_knn)
    knn_test_rec = recall_score(y_test, y_test_pred_knn)
    knn_test_auc = roc_auc_score(y_test, y_test_proba_knn)
    
    # 計算 Specificity (TN / (TN + FP))
    cm_knn = confusion_matrix(y_test, y_test_pred_knn)
    tn_knn, fp_knn, fn_knn, tp_knn = cm_knn.ravel()
    knn_test_spe = tn_knn / (tn_knn + fp_knn)
    
    print(f"\n訓練集準確率: {knn_train_acc:.4f}")
    print(f"測試集準確率: {knn_test_acc:.4f} ({knn_test_acc*100:.2f}%)")
    print(f"測試集 F1-Score: {knn_test_f1:.4f} ({knn_test_f1*100:.2f}%)")
    print(f"測試集 Precision: {knn_test_pre:.4f} ({knn_test_pre*100:.2f}%)")
    print(f"測試集 Recall (Sensitivity): {knn_test_rec:.4f} ({knn_test_rec*100:.2f}%)")
    print(f"測試集 Specificity: {knn_test_spe:.4f} ({knn_test_spe*100:.2f}%)")
    print(f"測試集 AUC: {knn_test_auc:.4f}")
    
    all_results['KNN'][seed] = {
        'train_acc': knn_train_acc,
        'test_acc': knn_test_acc,
        'test_f1': knn_test_f1,
        'test_precision': knn_test_pre,
        'test_recall': knn_test_rec,
        'test_specificity': knn_test_spe,
        'test_auc': knn_test_auc,
        'best_params': knn_grid.best_params_,
        'confusion_matrix': cm_knn,
        'y_test': y_test,
        'y_test_pred': y_test_pred_knn,
        'y_test_proba': y_test_proba_knn
    }
    
    # ========================================
    # CART 模型
    # ========================================
    print("\n[模型 2] CART (Classification and Regression Trees)")
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
    print(f"最佳 CV 分數: {cart_grid.best_score_:.4f}")
    
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
    tn_cart, fp_cart, fn_cart, tp_cart = cm_cart.ravel()
    cart_test_spe = tn_cart / (tn_cart + fp_cart)
    
    print(f"\n訓練集準確率: {cart_train_acc:.4f}")
    print(f"測試集準確率: {cart_test_acc:.4f} ({cart_test_acc*100:.2f}%)")
    print(f"測試集 F1-Score: {cart_test_f1:.4f} ({cart_test_f1*100:.2f}%)")
    print(f"測試集 Precision: {cart_test_pre:.4f} ({cart_test_pre*100:.2f}%)")
    print(f"測試集 Recall (Sensitivity): {cart_test_rec:.4f} ({cart_test_rec*100:.2f}%)")
    print(f"測試集 Specificity: {cart_test_spe:.4f} ({cart_test_spe*100:.2f}%)")
    print(f"測試集 AUC: {cart_test_auc:.4f}")
    
    # 特徵重要性
    feature_importance_cart = pd.DataFrame({
        'Feature': X.columns,
        'Importance': cart_best_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print(f"\nCART 特徵重要性 (Top 5):")
    print(feature_importance_cart.head().to_string())
    
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
        'model': cart_best_model,
        'y_test': y_test,
        'y_test_pred': y_test_pred_cart,
        'y_test_proba': y_test_proba_cart
    }
    
    # ========================================
    # C4.5 模型
    # ========================================
    print("\n[模型 3] C4.5 Decision Tree")
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
    print(f"最佳 CV 分數: {c45_grid.best_score_:.4f}")
    
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
    tn_c45, fp_c45, fn_c45, tp_c45 = cm_c45.ravel()
    c45_test_spe = tn_c45 / (tn_c45 + fp_c45)
    
    print(f"\n訓練集準確率: {c45_train_acc:.4f}")
    print(f"測試集準確率: {c45_test_acc:.4f} ({c45_test_acc*100:.2f}%)")
    print(f"測試集 F1-Score: {c45_test_f1:.4f} ({c45_test_f1*100:.2f}%)")
    print(f"測試集 Precision: {c45_test_pre:.4f} ({c45_test_pre*100:.2f}%)")
    print(f"測試集 Recall (Sensitivity): {c45_test_rec:.4f} ({c45_test_rec*100:.2f}%)")
    print(f"測試集 Specificity: {c45_test_spe:.4f} ({c45_test_spe*100:.2f}%)")
    print(f"測試集 AUC: {c45_test_auc:.4f}")
    
    # 特徵重要性
    feature_importance_c45 = pd.DataFrame({
        'Feature': X.columns,
        'Importance': c45_best_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print(f"\nC4.5 特徵重要性 (Top 5):")
    print(feature_importance_c45.head().to_string())
    
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
        'model': c45_best_model,
        'y_test': y_test,
        'y_test_pred': y_test_pred_c45,
        'y_test_proba': y_test_proba_c45
    }

# ============================================================
# 4. 跨 Seed 結果彙整
# ============================================================
print("\n" + "=" * 80)
print("跨 Seed 結果彙整")
print("=" * 80)

# 建立彙整表格
summary_data = []

for model_name in ['KNN', 'CART', 'C4.5']:
    for seed in SEEDS:
        result = all_results[model_name][seed]
        summary_data.append({
            'Model': model_name,
            'Seed': seed,
            'Test_ACC': result['test_acc'] * 100,
            'Test_F1': result['test_f1'] * 100,
            'Test_Precision': result['test_precision'] * 100,
            'Test_Sensitivity': result['test_recall'] * 100,
            'Test_Specificity': result['test_specificity'] * 100
        })

summary_df = pd.DataFrame(summary_data)

# 計算平均值
avg_data = []
for model_name in ['KNN', 'CART', 'C4.5']:
    model_results = summary_df[summary_df['Model'] == model_name]
    avg_data.append({
        'Model': model_name,
        'Seed': 'Average',
        'Test_ACC': model_results['Test_ACC'].mean(),
        'Test_F1': model_results['Test_F1'].mean(),
        'Test_Precision': model_results['Test_Precision'].mean(),
        'Test_Sensitivity': model_results['Test_Sensitivity'].mean(),
        'Test_Specificity': model_results['Test_Specificity'].mean()
    })

avg_df = pd.DataFrame(avg_data)
full_summary_df = pd.concat([summary_df, avg_df], ignore_index=True)

print("\n完整結果表格:")
print(full_summary_df.to_string(index=False))

# 儲存 CSV
output_dir = 'C:/Users/fiona/iCloudDrive/114/機器學習/knn/'
full_summary_df.to_csv(output_dir + 'complete_results_all_seeds.csv', 
                       index=False, encoding='utf-8-sig')
print("\n✓ 完整結果已儲存: complete_results_all_seeds.csv")

# ============================================================
# 5. 視覺化比較 (使用 Seed=400 的結果)
# ============================================================
print("\n" + "=" * 80)
print("產生視覺化圖表 (使用 Seed=400)")
print("=" * 80)

seed_for_viz = 400

# 準備 Seed 400 的數據
viz_data = []
for model_name in ['KNN', 'CART', 'C4.5']:
    result = all_results[model_name][seed_for_viz]
    viz_data.append({
        'Model': model_name,
        'Train_Accuracy': result['train_acc'],
        'Test_Accuracy': result['test_acc'],
        'Test_Precision': result['test_precision'],
        'Test_Recall': result['test_recall'],
        'Test_F1': result['test_f1'],
        'Test_AUC': result['test_auc']
    })

viz_df = pd.DataFrame(viz_data)

# 繪製比較圖表
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 圖1: 準確率比較
ax1 = axes[0, 0]
x_pos = np.arange(len(viz_df))
width = 0.35
ax1.bar(x_pos - width/2, viz_df['Train_Accuracy'], width, label='Train Accuracy', alpha=0.8)
ax1.bar(x_pos + width/2, viz_df['Test_Accuracy'], width, label='Test Accuracy', alpha=0.8)
ax1.set_xlabel('Model', fontsize=12)
ax1.set_ylabel('Accuracy', fontsize=12)
ax1.set_title(f'Model Accuracy Comparison (Seed={seed_for_viz})', fontsize=14, fontweight='bold')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(viz_df['Model'])
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_ylim([0, 1])

# 圖2: 多項指標比較
ax2 = axes[0, 1]
metrics = ['Test_Precision', 'Test_Recall', 'Test_F1', 'Test_AUC']
x = np.arange(len(viz_df))
width = 0.2
for i, metric in enumerate(metrics):
    ax2.bar(x + i*width - width*1.5, viz_df[metric], width, 
            label=metric.replace('Test_', ''))
ax2.set_xlabel('Model', fontsize=12)
ax2.set_ylabel('Score', fontsize=12)
ax2.set_title(f'Model Performance Metrics (Seed={seed_for_viz})', fontsize=14, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(viz_df['Model'])
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_ylim([0, 1])

# 圖3: ROC 曲線
ax3 = axes[1, 0]
for model_name in ['KNN', 'CART', 'C4.5']:
    result = all_results[model_name][seed_for_viz]
    fpr, tpr, _ = roc_curve(result['y_test'], result['y_test_proba'])
    ax3.plot(fpr, tpr, label=f'{model_name} (AUC={result["test_auc"]:.3f})', linewidth=2)
ax3.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
ax3.set_xlabel('False Positive Rate', fontsize=12)
ax3.set_ylabel('True Positive Rate', fontsize=12)
ax3.set_title(f'ROC Curve Comparison (Seed={seed_for_viz})', fontsize=14, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 圖4: 跨 Seed 準確率比較
ax4 = axes[1, 1]
for model_name in ['KNN', 'CART', 'C4.5']:
    accs = [all_results[model_name][s]['test_acc'] * 100 for s in SEEDS]
    ax4.plot(SEEDS, accs, marker='o', linewidth=2, markersize=8, label=model_name)
ax4.set_xlabel('Random Seed', fontsize=12)
ax4.set_ylabel('Test Accuracy (%)', fontsize=12)
ax4.set_title('Test Accuracy Across Different Seeds', fontsize=14, fontweight='bold')
ax4.set_xticks(SEEDS)
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir + 'model_comparison_seed400.png', dpi=300, bbox_inches='tight')
print("\n✓ 模型比較圖表已儲存: model_comparison_seed400.png")
plt.close()

# 繪製決策樹 (Seed 400)
fig, axes = plt.subplots(1, 2, figsize=(24, 10))

cart_model = all_results['CART'][seed_for_viz]['model']
tree.plot_tree(cart_model, 
               filled=True, 
               rounded=True, 
               class_names=['Not Survived', 'Survived'],
               feature_names=X.columns,
               ax=axes[0],
               fontsize=8)
axes[0].set_title(f'CART Decision Tree (Seed={seed_for_viz})', fontsize=14, fontweight='bold')

c45_model = all_results['C4.5'][seed_for_viz]['model']
tree.plot_tree(c45_model, 
               filled=True, 
               rounded=True, 
               class_names=['Not Survived', 'Survived'],
               feature_names=X.columns,
               ax=axes[1],
               fontsize=8)
axes[1].set_title(f'C4.5 Decision Tree (Seed={seed_for_viz})', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir + 'decision_trees_seed400.png', dpi=300, bbox_inches='tight')
print("✓ 決策樹視覺化已儲存: decision_trees_seed400.png")
plt.close()

# ============================================================
# 6. 詳細結果輸出
# ============================================================
print("\n" + "=" * 80)
print("儲存詳細結果")
print("=" * 80)

with open(output_dir + 'detailed_results_all_seeds.txt', 'w', encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write("Assignment 4 - 完整實驗結果\n")
    f.write("Random Seeds: 4, 40, 400\n")
    f.write("=" * 80 + "\n\n")
    
    for model_name in ['KNN', 'CART', 'C4.5']:
        f.write(f"\n{'=' * 80}\n")
        f.write(f"{model_name} 模型結果\n")
        f.write(f"{'=' * 80}\n\n")
        
        for seed in SEEDS:
            result = all_results[model_name][seed]
            f.write(f"\nRandom Seed = {seed}\n")
            f.write("-" * 40 + "\n")
            f.write(f"最佳參數: {result['best_params']}\n")
            f.write(f"訓練集準確率: {result['train_acc']:.4f} ({result['train_acc']*100:.2f}%)\n")
            f.write(f"測試集準確率: {result['test_acc']:.4f} ({result['test_acc']*100:.2f}%)\n")
            f.write(f"測試集 F1-Score: {result['test_f1']:.4f} ({result['test_f1']*100:.2f}%)\n")
            f.write(f"測試集 Precision: {result['test_precision']:.4f} ({result['test_precision']*100:.2f}%)\n")
            f.write(f"測試集 Sensitivity (Recall): {result['test_recall']:.4f} ({result['test_recall']*100:.2f}%)\n")
            f.write(f"測試集 Specificity: {result['test_specificity']:.4f} ({result['test_specificity']*100:.2f}%)\n")
            f.write(f"測試集 AUC: {result['test_auc']:.4f}\n")
            
            cm = result['confusion_matrix']
            f.write(f"\n混淆矩陣:\n")
            f.write(f"  TN={cm[0,0]}, FP={cm[0,1]}\n")
            f.write(f"  FN={cm[1,0]}, TP={cm[1,1]}\n")
            
            if model_name in ['CART', 'C4.5']:
                f.write(f"\n特徵重要性 (Top 5):\n")
                f.write(result['feature_importance'].head().to_string())
                f.write("\n")
        
        # 計算平均值
        avg_acc = np.mean([all_results[model_name][s]['test_acc'] for s in SEEDS])
        avg_f1 = np.mean([all_results[model_name][s]['test_f1'] for s in SEEDS])
        f.write(f"\n{model_name} 平均測試集準確率: {avg_acc:.4f} ({avg_acc*100:.2f}%)\n")
        f.write(f"{model_name} 平均測試集 F1-Score: {avg_f1:.4f} ({avg_f1*100:.2f}%)\n")

print("✓ 詳細結果已儲存: detailed_results_all_seeds.txt")

# ============================================================
# 7. 最終總結
# ============================================================
print("\n" + "=" * 80)
print("最終總結")
print("=" * 80)

# 找出平均表現最佳的模型
avg_accs = {}
for model_name in ['KNN', 'CART', 'C4.5']:
    avg_accs[model_name] = np.mean([all_results[model_name][s]['test_acc'] for s in SEEDS])

best_model = max(avg_accs, key=avg_accs.get)
print(f"\n平均測試集準確率最佳模型: {best_model} ({avg_accs[best_model]*100:.2f}%)")

print("\n各模型平均表現:")
for model_name in ['KNN', 'CART', 'C4.5']:
    print(f"  {model_name}: {avg_accs[model_name]*100:.2f}%")

print("\n" + "=" * 80)
print("Assignment 4 完成!")
print("=" * 80)
print("\n產生的檔案:")
print("1. complete_results_all_seeds.csv - 所有 Seed 的完整結果表格")
print("2. model_comparison_seed400.png - 模型比較圖表 (Seed=400)")
print("3. decision_trees_seed400.png - 決策樹視覺化 (Seed=400)")
print("4. detailed_results_all_seeds.txt - 所有 Seed 的詳細結果")