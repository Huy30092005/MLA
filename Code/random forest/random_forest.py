import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score, RocCurveDisplay
)
import matplotlib.pyplot as plt
import seaborn as sns


file_path = 'dataset/data.csv'

df = pd.read_csv(file_path, sep=';')

# Strip stray whitespace/tabs from column names
df.columns = df.columns.str.strip()

# Preprocessing: remove all 2nd-semester columns (predict before the 2nd semester starts)
sem2_cols = [c for c in df.columns if '2nd sem' in c.lower()]
print("Dropping 2nd-semester columns:", sem2_cols)
df = df.drop(columns=sem2_cols)

print("Dataset shape:", df.shape)
print("\nTarget distribution:")
print(df['Target'].value_counts())

x = df.drop(columns=['Target'])
y = df['Target']

y_binary = y.map({
    'Dropout': 1,
    'Graduate': 0,
    'Enrolled': 0
})

X_train, X_test, y_train, y_test = train_test_split(
    x, y_binary,
    test_size=0.2,
    random_state=42,
    stratify=y_binary
)


model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)


y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]


print("\n ==============Results==========")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"ROC AUC Score: {roc_auc_score(y_test, y_proba):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Not Dropout', 'Dropout']))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Not Dropout', 'Dropout'], 
            yticklabels=['Not Dropout', 'Dropout'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()


RocCurveDisplay.from_predictions(y_test, y_proba)
plt.title('ROC Curve')
plt.show()

importance_df = pd.DataFrame({
    'Feature': x.columns,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)


print("\nFeature Importances:")
print(importance_df.head(15).to_string(index=False))

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title('Feature Importances')
plt.tight_layout()
plt.show()

# Ablation study: Full features vs Removed Debtor and Tuition Fees vs Removed all financial features
from sklearn.metrics import precision_score, recall_score, f1_score

SEEDS = [42, 100, 2024, 99, 7]
K_VALUES = [0.10, 0.20]

def precision_at_k(y_true, y_proba, k):
    """Precision among the top k% of samples ranked by predicted probability (k is a fraction)."""
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    n_top = int(np.ceil(k * len(y_true)))
    top_idx = np.argsort(y_proba)[::-1][:n_top]   # highest probabilities first
    tp = int(y_true[top_idx].sum())
    return tp, n_top, tp / n_top

debtor_tuition = ['Debtor', 'Tuition fees up to date']
all_financial = debtor_tuition + ['Scholarship holder']

# Fail loudly if a column name doesn't match the dataset
missing = [c for c in all_financial if c not in x.columns]
assert not missing, f"Columns not found: {missing}"

feature_sets = {
    'Full features': list(x.columns),
    'No Debtor/Tuition': [c for c in x.columns if c not in debtor_tuition],
    'No financial': [c for c in x.columns if c not in all_financial],
}

rows = []
for name, cols in feature_sets.items():
    print(f"{name}: {len(cols)} features")
    for seed in SEEDS:
        X_tr, X_te, y_tr, y_te = train_test_split(
            x[cols], y_binary,
            test_size=0.2,
            random_state=seed,
            stratify=y_binary
        )
        rf = RandomForestClassifier(
            n_estimators=200,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=seed,
            n_jobs=-1
        )
        rf.fit(X_tr, y_tr)
        pred = rf.predict(X_te)
        proba = rf.predict_proba(X_te)[:, 1]

        y_te_arr = np.asarray(y_te)
        row = {
            'Feature set': name,
            'Seed': seed,
            'Accuracy': accuracy_score(y_te, pred),
            'ROC AUC': roc_auc_score(y_te, proba),
            'Precision': precision_score(y_te, pred, zero_division=0),
            'Recall': recall_score(y_te, pred, zero_division=0),
            'F1': f1_score(y_te, pred, zero_division=0),
            'Total dropouts': int(y_te_arr.sum()),
            'TP (predicted)': int(((pred == 1) & (y_te_arr == 1)).sum()),
        }
        for k in K_VALUES:
            tp, n_top, prec = precision_at_k(y_te, proba, k)
            row[f'P@{int(k*100)}%'] = prec
            row[f'TP@{int(k*100)}%'] = tp
        rows.append(row)

ablation = pd.DataFrame(rows)
metrics = ['Accuracy', 'ROC AUC', 'Precision', 'Recall', 'F1'] + [f'P@{int(k*100)}%' for k in K_VALUES]
order = list(feature_sets.keys())

mean_df = ablation.groupby('Feature set')[metrics].mean().loc[order]
std_df = ablation.groupby('Feature set')[metrics].std(ddof=0).loc[order]

print("\n========== Per-seed P@K ==========")
print(ablation[['Feature set', 'Seed'] + [f'P@{int(k*100)}%' for k in K_VALUES]]
      .round(4).to_string(index=False))


tp_cols = ['Total dropouts', 'TP (predicted)'] + [f'TP@{int(k*100)}%' for k in K_VALUES]

print("\n========== Per-seed true positives ==========")
print(ablation[['Feature set', 'Seed'] + tp_cols].to_string(index=False))

print("\n========== True positives: mean over 5 seeds ==========")
tp_mean = ablation.groupby('Feature set')[tp_cols].mean().loc[order]
print(tp_mean.round(1).to_string())

print("\n========== True positives: change vs. full features ==========")
print((tp_mean - tp_mean.loc['Full features']).round(1).to_string())



print("\n========== Ablation: mean over 5 seeds ==========")
print(mean_df.round(4).to_string())
print("\n========== Ablation: std over 5 seeds ==========")
print(std_df.round(4).to_string())

print("\n========== Change vs. full features ==========")
print((mean_df - mean_df.loc['Full features']).round(4).to_string())

ax = mean_df.T.plot(kind='bar', yerr=std_df.T, capsize=3, figsize=(12, 5), rot=0)
ax.set_ylim(0, 1)
ax.set_ylabel('Score (mean ± std over 5 seeds)')
ax.set_title('Ablation: effect of removing financial features')
plt.tight_layout()
plt.show()