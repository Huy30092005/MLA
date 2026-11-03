import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ( accuracy_score, confusion_matrix, classification_report, roc_auc_score, RocCurveDisplay )


import matplotlib.pyplot as plt
import seaborn as sns

# Preprocessing: Remove all the columns relating to the 2nd semester, as we want to predict the dropout before the 2nd semester starts.
file_path = 'dataset/data.csv'

df = pd.read_csv(file_path, sep=';')

# Strip stray whitespace/tabs from column names (some versions of this dataset have them)
df.columns = df.columns.str.strip()

# Preprocessing: remove all 2nd-semester columns, since we want to predict
# dropout BEFORE the 2nd semester starts.
sem2_cols = [c for c in df.columns if '2nd sem' in c.lower()]
print("\nDropping 2nd-semester columns:")
for c in sem2_cols:
    print("  -", c)
df = df.drop(columns=sem2_cols)

print("\nDataset shape after dropping:", df.shape)
print("\nColumns:")
print(df.columns.tolist())
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


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


model = LogisticRegression(
    max_iter=1000,
    class_weight='balanced',
    random_state=42,
    solver='lbfgs'
)

model.fit(X_train_scaled, y_train)


y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)[:, 1]

print("\n ==============Results==========")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC AUC Score:", roc_auc_score(y_test, y_proba))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Not Dropout', 'Dropout']))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Not Dropout', 'Dropout'], 
            yticklabels=['Not Dropout', 'Dropout'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.show()

# Print out
# Precision@K (10%, 20%) through 5 random seeds (42, 100, 2024, 99, 7)
# Precision@K average
# Recall and recall average
# Number of True Positives in the top K% of predicted probabilities
def metrics_at_k(y_true, y_proba, k):
    """
    Rank samples by predicted probability and look at the top k% (k is a fraction).
    Returns (true_positives_in_top_k, n_top, precision@k, recall@k)
      precision@k = TP in top k / n_top
      recall@k    = TP in top k / total positives in y_true
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)

    n_top = int(np.ceil(k * len(y_true)))
    top_idx = np.argsort(y_proba)[::-1][:n_top]   # highest probabilities first

    tp = int(y_true[top_idx].sum())
    total_pos = int(y_true.sum())

    precision = tp / n_top
    recall = tp / total_pos if total_pos > 0 else 0.0
    return tp, n_top, precision, recall


seeds = [42, 100, 2024, 99, 7]
k_values = [0.10, 0.20]

# results[k] is a list of dicts, one per seed
results = {k: [] for k in k_values}

for seed in seeds:
    X_tr, X_te, y_tr, y_te = train_test_split(
        x, y_binary,
        test_size=0.2,
        random_state=seed,
        stratify=y_binary
    )

    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr)
    X_te_s = sc.transform(X_te)

    clf = LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        random_state=seed,
        solver='lbfgs'
    )
    clf.fit(X_tr_s, y_tr)
    proba = clf.predict_proba(X_te_s)[:, 1]

    total_pos = int(np.asarray(y_te).sum())
    print(f"\n--- Seed {seed} (test size = {len(y_te)}, total dropouts = {total_pos}) ---")

    for k in k_values:
        tp, n_top, prec, rec = metrics_at_k(y_te, proba, k)
        results[k].append({'seed': seed, 'tp': tp, 'n_top': n_top,
                           'precision': prec, 'recall': rec})
        print(f"  Top {int(k*100):>2}% ({n_top} students) | "
              f"TP: {tp:>3} | Precision@K: {prec:.4f} | Recall@K: {rec:.4f}")

print("\n========== Summary over 5 seeds ==========")
for k in k_values:
    df_k = pd.DataFrame(results[k])
    print(f"\nTop {int(k*100)}%:")
    print(f"  Avg True Positives : {df_k['tp'].mean():.2f}")
    print(f"  Precision@K        : mean = {df_k['precision'].mean():.4f}, "
          f"std = {df_k['precision'].std(ddof=0):.4f}")
    print(f"  Recall@K           : mean = {df_k['recall'].mean():.4f}, "
          f"std = {df_k['recall'].std(ddof=0):.4f}")

# Optional: visualize precision and recall per seed
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, metric in zip(axes, ['precision', 'recall']):
    plot_df = pd.DataFrame({
        'Seed': seeds,
        'Top 10%': [r[metric] for r in results[0.10]],
        'Top 20%': [r[metric] for r in results[0.20]],
    }).set_index('Seed')
    plot_df.plot(kind='bar', ax=ax, rot=0)
    ax.set_title(f'{metric.capitalize()}@K across seeds')
    ax.set_ylabel(metric.capitalize())
    ax.set_ylim(0, 1)
plt.tight_layout()
plt.show()