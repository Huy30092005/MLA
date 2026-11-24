# predict_h.py
# Predict dropout for the H set using the binary-classification solution
# (Dropout = 1 vs Graduate/Enrolled = 0, scaled + class-balanced Logistic Regression).
#
# H = students with 1st-semester approved units >= 4 AND 1st-semester grade >= 12.
# All 2nd-semester columns are dropped (prediction is made before semester 2 starts).
#
# Pipeline:
#   1. Student information (approved, enrolled, grade, age, financial, ...)
#   2. LR model -> dropout probability
#   3. Ranking  -> students sorted by probability, highest first
#   4. Compare  -> LR prediction and heuristic prediction (approved/enrolled < 0.5)
#                  against the actual label
#
# Every student in H gets an OUT-OF-FOLD probability: it comes from models that
# never saw that student during training (5-fold stratified x 5 repeats, the same
# splits as silent_risk.py / H_lr_gb.py, averaged over the 5 repeats).
# Predicting on the data the model was fitted on would inflate the results.
# The heuristic has nothing to fit, so it is scored on the same students directly.
#
# Run from anywhere:  python silent_risk/predict_h.py
# Output:             h_set_predictions.csv (repo root)

import argparse
from pathlib import Path
from pprint import pprint

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------- config
ROOT = Path(__file__).resolve().parents[1]
FILE_PATH = ROOT / 'dataset' / 'data.csv'
OUT_PATH = ROOT / 'h_set_predictions.csv'

APPROVED_COL = 'Curricular units 1st sem (approved)'
ENROLLED_COL = 'Curricular units 1st sem (enrolled)'
GRADE_COL = 'Curricular units 1st sem (grade)'
RATIO_THRESHOLD = 0.5   # heuristic: approved / enrolled < 0.5 -> predicted dropout
K_VALUES = [0.10, 0.20]
N_SPLITS, N_REPEATS, SEED = 5, 5, 42


# ---------------------------------------------------------------- data
def load_raw():
    """The full dataset exactly as in dataset/data.csv (all columns, all students)."""
    df = pd.read_csv(FILE_PATH, sep=';')
    df.columns = df.columns.str.strip()
    return df


def _native(d):
    """numpy scalars -> plain Python values (cleaner to print / serialize)."""
    return {k: (v.item() if hasattr(v, 'item') else v) for k, v in d.items()}


def get_student_info(row_id, include_prediction=True):
    """Return everything known about one student.

    row_id: 0-based row index in dataset/data.csv (the same `row_id` as in
            h_set_predictions.csv).

    Returns a dict:
      row_id, target (Dropout/Graduate/Enrolled), actual_dropout (1/0),
      in_H (bool), approval_ratio, heuristic_prediction (1 = ratio < 0.5),
      features (every column of the row except Target, 2nd-semester ones included),
      prediction (LR output from h_set_predictions.csv for students in H, else None).
    """
    df = load_raw()
    if not 0 <= row_id < len(df):
        raise IndexError(f"row_id must be between 0 and {len(df) - 1}, got {row_id}")

    row = df.loc[row_id]
    approved, enrolled, grade = row[APPROVED_COL], row[ENROLLED_COL], row[GRADE_COL]
    ratio = approved / enrolled if enrolled > 0 else 0.0   # same rule as heuristic.py

    info = {
        'row_id': int(row_id),
        'target': row['Target'],
        'actual_dropout': int(row['Target'] == 'Dropout'),
        'in_H': bool(approved >= 4 and grade >= 12),
        'approval_ratio': float(ratio),
        'heuristic_prediction': int(ratio < RATIO_THRESHOLD),
        'features': _native(row.drop('Target').to_dict()),
        'prediction': None,
    }

    if include_prediction and info['in_H'] and OUT_PATH.exists():
        preds = pd.read_csv(OUT_PATH).set_index('row_id')
        if row_id in preds.index:
            p = preds.loc[row_id]
            info['prediction'] = _native({
                'dropout_probability': p['dropout_probability'],
                'risk_rank': int(p['risk_rank']),
                'students_ranked': len(preds),
                'predicted_dropout': int(p['predicted_dropout']),
                'flag_top_10pct': int(p['flag_top_10pct']),
                'flag_top_20pct': int(p['flag_top_20pct']),
            })
    return info


def load_h_set():
    df = load_raw()

    sem2_cols = [c for c in df.columns if '2nd sem' in c.lower()]
    df = df.drop(columns=sem2_cols)
    assert APPROVED_COL in df.columns and GRADE_COL in df.columns, "Check column names"

    H = df[(df[APPROVED_COL] >= 4) & (df[GRADE_COL] >= 12)].copy()
    X = H.drop(columns=['Target'])
    y = H['Target'].map({'Dropout': 1, 'Graduate': 0, 'Enrolled': 0})
    return df, X, y


# ---------------------------------------------------------------- model + helpers
def make_lr():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight='balanced', solver='lbfgs'),
    )


def top_k_mask(proba, k):
    """Boolean mask of the top k fraction of students by predicted probability."""
    proba = np.asarray(proba)
    n_top = int(np.ceil(k * len(proba)))
    top_idx = np.argsort(-proba, kind='stable')[:n_top]
    mask = np.zeros(len(proba), dtype=bool)
    mask[top_idx] = True
    return mask


def precision_at_k(y_true, proba, k):
    y_true = np.asarray(y_true)
    mask = top_k_mask(proba, k)
    return y_true[mask].sum() / mask.sum()


def approval_ratio(frame):
    """approved / enrolled (1st semester). 0 enrolled units -> ratio 0 (same as heuristic.py)."""
    enrolled = frame[ENROLLED_COL].to_numpy(dtype=float)
    approved = frame[APPROVED_COL].to_numpy(dtype=float)
    return np.where(enrolled > 0, approved / np.where(enrolled > 0, enrolled, 1), 0.0)


def compare_row(name, y_true, pred, score):
    """Confusion counts and metrics for one set of predictions against the actual label."""
    y_true, pred = np.asarray(y_true), np.asarray(pred)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    row = {'Model': name, 'Flagged': tp + fp, 'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn,
           'Accuracy': (tp + tn) / len(y_true), 'Precision': prec, 'Recall': rec,
           'F1': f1, 'ROC AUC': roc_auc_score(y_true, score)}
    for k in K_VALUES:
        row[f'P@{int(k*100)}%'] = precision_at_k(y_true, score, k)
    return row


# ---------------------------------------------------------------- predict
def predict_h():
    df, X, y = load_h_set()
    base_rate = y.mean()
    print(f"H set: {len(X)} / {len(df)} students, dropouts = {int(y.sum())} "
          f"({base_rate:.2%})")

    splits = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS,
                                     random_state=SEED).split(X, y)

    proba_sum = np.zeros(len(X))
    n_preds = np.zeros(len(X))
    fold_p_at_k = {k: [] for k in K_VALUES}

    for tr, te in splits:
        model = make_lr().fit(X.iloc[tr], y.iloc[tr])
        p = model.predict_proba(X.iloc[te])[:, 1]
        proba_sum[te] += p
        n_preds[te] += 1
        for k in K_VALUES:
            fold_p_at_k[k].append(precision_at_k(y.iloc[te], p, k))

    proba = proba_sum / n_preds   # each student: mean over N_REPEATS out-of-fold predictions

    ratio = approval_ratio(X)
    out = pd.DataFrame({
        'row_id': X.index,                          # row index in dataset/data.csv
        'actual_dropout': y.to_numpy(),
        'dropout_probability': proba,
        'predicted_dropout': (proba >= 0.5).astype(int),
        'approval_ratio': ratio,
        'heuristic_prediction': (ratio < RATIO_THRESHOLD).astype(int),
    })
    out['lr_correct'] = (out['predicted_dropout'] == out['actual_dropout']).astype(int)
    out['heuristic_correct'] = (out['heuristic_prediction'] == out['actual_dropout']).astype(int)
    out['risk_rank'] = out['dropout_probability'].rank(ascending=False, method='first').astype(int)
    for k in K_VALUES:
        out[f'flag_top_{int(k*100)}pct'] = top_k_mask(proba, k).astype(int)
    out = out.sort_values('risk_rank').reset_index(drop=True)
    out.to_csv(OUT_PATH, index=False)

    # ------------------------------------------------------------ report
    print(f"\nOut-of-fold ROC AUC: {roc_auc_score(y, proba):.4f}")

    print(f"\nRanking quality (mean ± std over {N_SPLITS * N_REPEATS} test folds, "
          f"same protocol as H_lr_gb.py):")
    for k in K_VALUES:
        p = np.array(fold_p_at_k[k])
        print(f"  P@{int(k*100)}%: {p.mean() * 100:.2f}% ± {p.std(ddof=0) * 100:.2f}"
              f"  | lift vs base rate {base_rate:.2%}: {p.mean() / base_rate:.2f}x")

    print("\nFlagged groups (ranking over all of H, averaged out-of-fold probabilities):")
    for k in K_VALUES:
        mask = top_k_mask(proba, k)
        tp = int(y.to_numpy()[mask].sum())
        print(f"  Top {int(k*100)}% ({mask.sum()} students): "
              f"{tp} dropouts caught | precision {tp / mask.sum():.2%} | "
              f"recall {tp / int(y.sum()):.2%}")

    pred = out['predicted_dropout'].to_numpy()
    act = out['actual_dropout'].to_numpy()
    tp = int(((pred == 1) & (act == 1)).sum())
    print(f"\nThreshold 0.5: {int(pred.sum())} flagged, {tp} true dropouts "
          f"(precision {tp / max(pred.sum(), 1):.2%}, recall {tp / int(act.sum()):.2%})")

    # ------------------------------------------------------------ step 4: compare with heuristic and actual
    y_arr = out['actual_dropout'].to_numpy()
    h_pred = out['heuristic_prediction'].to_numpy()
    n_heur = int(h_pred.sum())
    lr_score = out['dropout_probability']
    h_score = -out['approval_ratio']   # lower ratio -> higher risk

    # 1) Fixed rules: LR at probability 0.5, heuristic at ratio < 0.5.
    rows = [
        compare_row('LR (probability >= 0.5)', y_arr, out['predicted_dropout'], lr_score),
        compare_row(f'Heuristic (ratio < {RATIO_THRESHOLD})', y_arr, h_pred, h_score),
    ]
    # 2) Same budget: flag the top k% of H by each model's own risk score.
    #    The heuristic has many tied ratios, so its top-k% depends on tie-breaking
    #    (stable order = original row order here).
    for k in K_VALUES:
        pct = int(k * 100)
        rows.append(compare_row(f'LR (top {pct}%)', y_arr, top_k_mask(lr_score, k), lr_score))
        rows.append(compare_row(f'Heuristic (top {pct}%, lowest ratio)', y_arr,
                                top_k_mask(h_score, k), h_score))
    comp = pd.DataFrame(rows).set_index('Model')

    pd.set_option('display.width', 220)
    pd.set_option('display.max_columns', None)
    print("\nCompare with actual label (same students, all of H):")
    print(comp.round(4).to_string())
    print(f"\nNote: the ratio < {RATIO_THRESHOLD} rule flags only {n_heur} of {len(out)} students in H "
          f"(H already requires >= 4 approved units), so compare the models at equal budgets "
          f"(top 10% / 20%) rather than at their default thresholds.")

    both = int(((out['predicted_dropout'] == 1) & (h_pred == 1)).sum())
    print(f"LR (0.5) and heuristic both flag {both} students; "
          f"LR only: {int(out['predicted_dropout'].sum()) - both}, heuristic only: {n_heur - both}")

    print(f"\nSaved {len(out)} predictions to {OUT_PATH}")
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Dropout predictions for the H set.")
    parser.add_argument('--student', type=int, metavar='ROW_ID',
                        help="print one student's information and exit "
                             "(prediction fields need h_set_predictions.csv, "
                             "created by running this script without arguments)")
    args = parser.parse_args()

    if args.student is not None:
        pprint(get_student_info(args.student), sort_dicts=False)
    else:
        predict_h()