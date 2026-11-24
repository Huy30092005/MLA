# comparison.py
# Compare Logistic Regression vs. the approval-ratio heuristic on the SAME test splits,
# using only information available before the 2nd semester starts.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

# ---------------------------------------------------------------- config
FILE_PATH = 'dataset/data.csv'
SEEDS = [42, 100, 2024, 99, 7]
K_VALUES = [0.10, 0.20]
RATIO_THRESHOLD = 0.5

ENROLLED_COL = 'Curricular units 1st sem (enrolled)'
APPROVED_COL = 'Curricular units 1st sem (approved)'

# ---------------------------------------------------------------- data
df = pd.read_csv(FILE_PATH, sep=';')
df.columns = df.columns.str.strip()

# Remove all 2nd-semester columns (predict before the 2nd semester starts)
sem2_cols = [c for c in df.columns if '2nd sem' in c.lower()]
df = df.drop(columns=sem2_cols)
print("Dropped 2nd-semester columns:", sem2_cols)

x = df.drop(columns=['Target'])
y = df['Target'].map({'Dropout': 1, 'Graduate': 0, 'Enrolled': 0})


# ---------------------------------------------------------------- helpers
def approval_ratio(frame):
    """approved / enrolled (1st semester). 0 enrolled units -> ratio 0."""
    enrolled = frame[ENROLLED_COL].to_numpy(dtype=float)
    approved = frame[APPROVED_COL].to_numpy(dtype=float)
    return np.where(enrolled > 0, approved / np.where(enrolled > 0, enrolled, 1), 0.0)


def metrics_at_k(y_true, score, k):
    """Top k% by score (higher = more likely dropout). Returns (tp, n_top, precision, recall)."""
    y_true = np.asarray(y_true)
    score = np.asarray(score)
    n_top = int(np.ceil(k * len(y_true)))
    top_idx = np.argsort(-score, kind='stable')[:n_top]
    tp = int(y_true[top_idx].sum())
    total_pos = int(y_true.sum())
    return tp, n_top, tp / n_top, (tp / total_pos if total_pos else 0.0)


def evaluate(y_true, y_pred, score):
    """All metrics for one model on one split."""
    row = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1': f1_score(y_true, y_pred, zero_division=0),
        'ROC AUC': roc_auc_score(y_true, score),
    }
    for k in K_VALUES:
        tp, _, prec, rec = metrics_at_k(y_true, score, k)
        pct = int(k * 100)
        row[f'TP@{pct}%'] = tp
        row[f'Precision@{pct}%'] = prec
        row[f'Recall@{pct}%'] = rec
    return row


# ---------------------------------------------------------------- run
rows = []
for seed in SEEDS:
    X_tr, X_te, y_tr, y_te = train_test_split(
        x, y, test_size=0.2, random_state=seed, stratify=y
    )

    # --- Logistic regression
    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr)
    X_te_s = sc.transform(X_te)

    clf = LogisticRegression(max_iter=1000, class_weight='balanced',
                             random_state=seed, solver='lbfgs')
    clf.fit(X_tr_s, y_tr)
    lr_pred = clf.predict(X_te_s)
    lr_score = clf.predict_proba(X_te_s)[:, 1]

    # --- Heuristic (no fitting; scored on the same test rows)
    ratio = approval_ratio(X_te)
    h_pred = (ratio < RATIO_THRESHOLD).astype(int)
    h_score = -ratio  # lower ratio -> higher dropout risk

    for name, pred, score in [('Logistic Regression', lr_pred, lr_score),
                              ('Heuristic', h_pred, h_score)]:
        r = evaluate(y_te, pred, score)
        r.update({'Model': name, 'Seed': seed})
        rows.append(r)

res = pd.DataFrame(rows)

# ---------------------------------------------------------------- report
pd.set_option('display.width', 200)
pd.set_option('display.max_columns', None)

metric_cols = [c for c in res.columns if c not in ('Model', 'Seed')]

print("\n========== Per-seed results ==========")
print(res.sort_values(['Seed', 'Model'])[['Seed', 'Model'] + metric_cols]
      .round(4).to_string(index=False))

summary_mean = res.groupby('Model')[metric_cols].mean()
summary_std = res.groupby('Model')[metric_cols].std(ddof=0)

print("\n========== Mean over seeds ==========")
print(summary_mean.round(4).T.to_string())

print("\n========== Std over seeds ==========")
print(summary_std.round(4).T.to_string())

print("\n========== Difference (Logistic Regression - Heuristic), mean ==========")
diff = summary_mean.loc['Logistic Regression'] - summary_mean.loc['Heuristic']
print(diff.round(4).to_string())

# ---------------------------------------------------------------- plot
plot_metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC AUC']
plot_metrics += [f'Precision@{int(k*100)}%' for k in K_VALUES]
plot_metrics += [f'Recall@{int(k*100)}%' for k in K_VALUES]

means = summary_mean[plot_metrics].T
errs = summary_std[plot_metrics].T

ax = means.plot(kind='bar', yerr=errs, capsize=3, figsize=(12, 5), rot=30)
ax.set_ylim(0, 1)
ax.set_ylabel('Score (mean ± std over 5 seeds)')
ax.set_title('Logistic Regression vs. Heuristic (approved/enrolled < 0.5)')
plt.tight_layout()
plt.show()