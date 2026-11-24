# Create a script that will take the data of an index from data.csv, then give the user options to choose the processing model in the codebase, then return the index's full data and the prediction result from the chosen model.


import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------- config
DEFAULT_DATA = Path(__file__).resolve().parent / 'dataset' / 'data.csv'
APPROVED_COL = 'Curricular units 1st sem (approved)'
ENROLLED_COL = 'Curricular units 1st sem (enrolled)'
RATIO_THRESHOLD = 0.5      # same rule as heuristic.py
PROB_THRESHOLD = 0.5       # probability cut-off for the label of a fitted model
SEED = 42

FINANCIAL_FIELDS = ['Debtor', 'Tuition fees up to date', 'Scholarship holder']


def make_lr():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight='balanced', solver='lbfgs'),
    )


def make_rf():
    return RandomForestClassifier(
        n_estimators=200, max_depth=None, min_samples_split=5, min_samples_leaf=2,
        class_weight='balanced', random_state=SEED, n_jobs=-1,
    )


def make_gb():
    return GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=3, subsample=0.8,
        random_state=SEED,
    )


# key -> (menu label, factory or None for the rule)
MODELS = {
    'heuristic': ('Heuristic rule: approved / enrolled < 0.5', None),
    'lr': ('Logistic Regression (scaled, class_weight=balanced)', make_lr),
    'rf': ('Random Forest (200 trees, class_weight=balanced)', make_rf),
    'gb': ('Gradient Boosting (200 trees, lr 0.05, depth 3)', make_gb),
}
MENU_ORDER = ['heuristic', 'lr', 'rf', 'gb']


# ---------------------------------------------------------------- data
def load_data(path):
    path = Path(path)
    if not path.exists():
        sys.exit(f"Data file not found: {path}\n"
                 f"Put the dataset at dataset/data.csv or pass --data PATH.")
    df = pd.read_csv(path, sep=';')
    df.columns = df.columns.str.strip()
    for col in (APPROVED_COL, ENROLLED_COL, 'Target'):
        if col not in df.columns:
            sys.exit(f"Column not found in {path}: {col!r}. Is this the UCI dropout file?")
    return df


def feature_columns(df):
    """Semester-1 prediction: drop every '2nd sem' column and the target."""
    sem2 = [c for c in df.columns if '2nd sem' in c.lower()]
    return [c for c in df.columns if c not in sem2 and c != 'Target']


def approval_ratio(approved, enrolled):
    """approved / enrolled; zero enrolled units -> ratio 0 (same as heuristic.py)."""
    return float(approved) / float(enrolled) if float(enrolled) > 0 else 0.0


# ---------------------------------------------------------------- user input
def ask_index(n_rows):
    while True:
        raw = input(f"Student index (0 to {n_rows - 1}): ").strip()
        try:
            idx = int(raw)
        except ValueError:
            print("  Please enter a whole number.")
            continue
        if 0 <= idx < n_rows:
            return idx
        print(f"  Out of range. Valid indices are 0 to {n_rows - 1}.")


def ask_model():
    print("\nChoose a model:")
    for i, key in enumerate(MENU_ORDER, start=1):
        print(f"  {i}. {MODELS[key][0]}   [{key}]")
    while True:
        raw = input("Model number or name: ").strip().lower()
        if raw in MODELS:
            return raw
        if raw.isdigit() and 1 <= int(raw) <= len(MENU_ORDER):
            return MENU_ORDER[int(raw) - 1]
        print("  Not a valid choice.")


# ---------------------------------------------------------------- prediction
def predict(df, idx, model_key):
    """Return a dict with the prediction for row `idx` from the chosen model."""
    feats = feature_columns(df)
    row = df.iloc[idx]
    y_all = (df['Target'] == 'Dropout').astype(int)

    if model_key == 'heuristic':
        ratio = approval_ratio(row[APPROVED_COL], row[ENROLLED_COL])
        return {
            'model': model_key,
            'predicted_dropout': int(ratio < RATIO_THRESHOLD),
            'score': 1.0 - ratio,   # higher = more risk; not a probability
            'score_meaning': 'rule score = 1 - approval ratio (not a probability)',
            'details': {
                'approved_units_sem1': float(row[APPROVED_COL]),
                'enrolled_units_sem1': float(row[ENROLLED_COL]),
                'approval_ratio': ratio,
                'threshold': RATIO_THRESHOLD,
            },
        }

    # fitted models: leave the chosen student out of training
    train_mask = np.arange(len(df)) != idx
    model = MODELS[model_key][1]()
    model.fit(df.loc[train_mask, feats], y_all[train_mask])
    x_row = df.iloc[[idx]][feats]
    proba = float(model.predict_proba(x_row)[0, 1])

    out = {
        'model': model_key,
        'predicted_dropout': int(proba >= PROB_THRESHOLD),
        'score': proba,
        'score_meaning': ('model score for dropout (balanced class weights, so it is not '
                          'calibrated to the 32% base rate)' if model_key != 'gb' else
                          'model score for dropout (not calibrated)'),
        'details': {'trained_on_students': int(train_mask.sum()), 'n_features': len(feats)},
    }

    if model_key == 'lr':
        scaler, clf = model[0], model[-1]
        z = scaler.transform(x_row)[0]
        contrib = z * clf.coef_[0]
        order = np.argsort(-np.abs(contrib))[:8]
        out['details']['intercept'] = float(clf.intercept_[0])
        out['details']['logit'] = float(clf.intercept_[0] + contrib.sum())
        out['details']['top_contributions'] = [
            {'feature': feats[i], 'value': float(x_row.iloc[0, i]), 'z': float(z[i]),
             'coef': float(clf.coef_[0][i]), 'contribution': float(contrib[i])}
            for i in order
        ]
    else:
        imp = model.feature_importances_
        order = np.argsort(-imp)[:8]
        out['details']['top_feature_importances'] = [
            {'feature': feats[i], 'importance': float(imp[i]), 'value': float(x_row.iloc[0, i])}
            for i in order
        ]
    return out


# ---------------------------------------------------------------- output
def print_report(df, idx, result):
    row = df.iloc[idx]
    actual = row['Target']
    actual_bin = int(actual == 'Dropout')
    pred = result['predicted_dropout']

    print("\n" + "=" * 72)
    print(f"STUDENT RECORD  (row index {idx} of {len(df) - 1})")
    print("=" * 72)
    width = max(len(c) for c in df.columns)
    for col in df.columns:
        mark = '   <- not used for prediction (after semester 1)' if '2nd sem' in col.lower() else ''
        print(f"  {col:<{width}} : {row[col]}{mark}")

    print("\n" + "=" * 72)
    print(f"PREDICTION  ({MODELS[result['model']][0]})")
    print("=" * 72)
    print(f"  Predicted class : {'Dropout' if pred else 'Non-Dropout (Graduate / Enrolled)'}")
    print(f"  Score           : {result['score']:.4f}  ({result['score_meaning']})")
    print(f"  Actual outcome  : {actual}  ->  prediction is "
          f"{'CORRECT' if pred == actual_bin else 'WRONG'}")

    d = result['details']
    if result['model'] == 'heuristic':
        print(f"  Rule            : {d['approved_units_sem1']:.0f} approved / "
              f"{d['enrolled_units_sem1']:.0f} enrolled = {d['approval_ratio']:.3f} "
              f"(flag if < {d['threshold']})")
    else:
        print(f"  Trained on      : {d['trained_on_students']} other students, "
              f"{d['n_features']} semester-1 features (this student excluded)")
    if 'top_contributions' in d:
        print(f"\n  Logistic regression logit = intercept {d['intercept']:+.3f} + sum of contributions "
              f"= {d['logit']:+.3f}")
        print("  Largest contributions (standardised value x coefficient):")
        for c in d['top_contributions']:
            print(f"    {c['feature']:<42} value {c['value']:>9.3f}  z {c['z']:>+7.2f}  "
                  f"coef {c['coef']:>+7.3f}  -> {c['contribution']:>+7.3f}")
    if 'top_feature_importances' in d:
        print("\n  Most important features of this model (global, not specific to this student):")
        for c in d['top_feature_importances']:
            print(f"    {c['feature']:<42} importance {c['importance']:.3f}  (student's value {c['value']:.3f})")

    print("\n  Notes")
    print("   - Ranking tool for advisors, not a label for the student.")
    print(f"   - Availability at the end of semester 1 of {', '.join(FINANCIAL_FIELDS)} "
          "is unconfirmed in the dataset paper.")
    if result['model'] != 'heuristic':
        print("   - Categorical codes (e.g. Nacionality, International) enter the model as numbers.")
    print()


def to_json(df, idx, result):
    row = df.iloc[idx]
    record = {c: (v.item() if hasattr(v, 'item') else v) for c, v in row.items()}
    return json.dumps({'index': idx, 'record': record, 'prediction': result,
                       'actual_outcome': row['Target']}, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description="Predict dropout for one student of data.csv.")
    ap.add_argument('--index', type=int, help="0-based row index in data.csv")
    ap.add_argument('--model', choices=list(MODELS), help="model to use")
    ap.add_argument('--data', default=str(DEFAULT_DATA), help="path to data.csv (default: dataset/data.csv)")
    ap.add_argument('--json', action='store_true', help="print JSON instead of the text report")
    args = ap.parse_args(argv)

    df = load_data(args.data)

    idx = args.index
    if idx is None:
        if not sys.stdin.isatty():
            sys.exit("No --index given and no interactive terminal available.")
        idx = ask_index(len(df))
    elif not 0 <= idx < len(df):
        sys.exit(f"Index {idx} is out of range. Valid indices are 0 to {len(df) - 1}.")

    model_key = args.model
    if model_key is None:
        if not sys.stdin.isatty():
            sys.exit("No --model given and no interactive terminal available.")
        model_key = ask_model()

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        result = predict(df, idx, model_key)

    if args.json:
        print(to_json(df, idx, result))
    else:
        print_report(df, idx, result)


if __name__ == '__main__':
    main()
