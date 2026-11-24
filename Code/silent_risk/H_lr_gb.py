import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
 
# ---------------------------------------------------------------- cấu hình
FILE_PATH = 'dataset/data.csv'
BASE_RATE = 0.1104
K_VALUES = [0.10, 0.20]
APPROVED_COL = 'Curricular units 1st sem (approved)'
GRADE_COL = 'Curricular units 1st sem (grade)'
 
FINANCIAL = ['Debtor', 'Tuition fees up to date', 'Scholarship holder']
AGE = ['Age at enrollment']
MACRO = ['Unemployment rate', 'Inflation rate', 'GDP']
 
# ---------------------------------------------------------------- dữ liệu
df = pd.read_csv(FILE_PATH, sep=';')
df.columns = df.columns.str.strip()
 
sem2_cols = [c for c in df.columns if '2nd sem' in c.lower()]
print("Bỏ các cột HK2:", sem2_cols)
df = df.drop(columns=sem2_cols)
 
assert APPROVED_COL in df.columns and GRADE_COL in df.columns, "Kiểm tra lại tên cột"
 
H = df[(df[APPROVED_COL] >= 4) & (df[GRADE_COL] >= 12)].copy()
x_H = H.drop(columns=['Target'])
y_H = H['Target'].map({'Dropout': 1, 'Graduate': 0, 'Enrolled': 0})
 
print(f"Tập H: {len(H)} / {len(df)} sinh viên, số dropout = {int(y_H.sum())}")
print(f"Tỷ lệ dropout thực tế trong H = {y_H.mean():.4%} (mốc so sánh: {BASE_RATE:.2%})")
 
missing = [c for c in FINANCIAL + AGE + MACRO if c not in x_H.columns]
assert not missing, f"Không tìm thấy cột: {missing}"
 
# ---------------------------------------------------------------- hàm phụ
def precision_at_k(y_true, y_proba, k):
    y_true = np.asarray(y_true)
    n_top = int(np.ceil(k * len(y_true)))
    top_idx = np.argsort(np.asarray(y_proba))[::-1][:n_top]
    return y_true[top_idx].sum() / n_top
 
 
# Chia fold MỘT LẦN, dùng chung cho mọi cấu hình và mô hình (so sánh cặp công bằng)
splits = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)
              .split(x_H, y_H))
N_SPLITS = len(splits)   # 25
 
 
def cv_eval(make_model, X, y):
    rows = []
    for tr, te in splits:
        model = make_model()
        model.fit(X.iloc[tr], y.iloc[tr])
        proba = model.predict_proba(X.iloc[te])[:, 1]
        rows.append({f'P@{int(k*100)}%': precision_at_k(y.iloc[te], proba, k)
                     for k in K_VALUES})
    return pd.DataFrame(rows)
 
 
# ---------------------------------------------------------------- mô hình
models = {
    'LR (mốc)': lambda: make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight='balanced', solver='lbfgs')
    ),
    'Gradient Boosting': lambda: GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=3, subsample=0.8,
        random_state=42
    ),
}
 
# ---------------------------------------------------------------- cấu hình đặc trưng
FULL = 'Đầy đủ đặc trưng'
feature_sets = {
    FULL: list(x_H.columns),
    'Bỏ nhóm tài chính': [c for c in x_H.columns if c not in FINANCIAL],
    'Bỏ tuổi': [c for c in x_H.columns if c not in AGE],
    'Bỏ 3 biến vĩ mô': [c for c in x_H.columns if c not in MACRO],
}
 
# ---------------------------------------------------------------- chạy
raw = {}     # (cấu hình, mô hình) -> DataFrame 25 dòng
cells = {}   # (cấu hình, mô hình, metric) -> (mean, std)
for fs_name, cols in feature_sets.items():
    for model_name, make_model in models.items():
        print(f"Đang chạy: {fs_name} | {model_name} ({len(cols)} đặc trưng)")
        res = cv_eval(make_model, x_H[cols], y_H)
        raw[(fs_name, model_name)] = res
        for metric in res.columns:
            cells[(fs_name, model_name, metric)] = (res[metric].mean(),
                                                    res[metric].std(ddof=0))
 
metrics = [f'P@{int(k*100)}%' for k in K_VALUES]
pd.set_option('display.width', 220)
pd.set_option('display.max_columns', None)
 
 
def fmt(m, s):
    return f"{m*100:.2f}% ± {s*100:.2f}"
 
 
# ---------------------------------------------------------------- bảng chính
table = pd.DataFrame(
    {(mn, met): [fmt(*cells[(fs, mn, met)]) for fs in feature_sets]
     for mn in models for met in metrics},
    index=list(feature_sets.keys())
)
table.columns = pd.MultiIndex.from_tuples(table.columns)
table.index.name = 'Cấu hình đặc trưng'
 
print(f"\n===== BẢNG CHÍNH: P@K trong H, 5-fold phân tầng x 5 lần lặp (mean ± std trên {N_SPLITS} lượt) =====")
print(f"Mốc so sánh (tỷ lệ dropout trong H): {BASE_RATE:.2%}\n")
print(table.to_string())
table.to_csv('h_set_results.csv')
 
# ---------------------------------------------------------------- báo cáo ablation
print("\n===== ABLATION: chênh lệch so với 'Đầy đủ đặc trưng' (điểm phần trăm) =====")
print(f"Mỗi ô: mean ± std của (cấu hình bỏ - đầy đủ) trên {N_SPLITS} lượt | số lượt kém hơn đầy đủ\n")
abl_rows = []
for mn in models:
    for fs in feature_sets:
        if fs == FULL:
            continue
        row = {'Mô hình': mn, 'Cấu hình': fs}
        for met in metrics:
            d = (raw[(fs, mn)][met] - raw[(FULL, mn)][met]) * 100
            row[met] = f"{d.mean():+.2f} ± {d.std(ddof=0):.2f} | {int((d < 0).sum())}/{N_SPLITS} kém hơn"
        abl_rows.append(row)
abl_table = pd.DataFrame(abl_rows).set_index(['Mô hình', 'Cấu hình'])
print(abl_table.to_string())
abl_table.to_csv('h_set_ablation.csv')
 
# ---------------------------------------------------------------- lift so với mốc
print(f"\n===== Mức nâng so với mốc {BASE_RATE:.2%} (P@K / mốc) =====")
lift_rows = []
for fs in feature_sets:
    row = {'Cấu hình': fs}
    for mn in models:
        for met in metrics:
            row[f'{mn} {met}'] = f"{cells[(fs, mn, met)][0] / BASE_RATE:.2f}x"
    lift_rows.append(row)
print(pd.DataFrame(lift_rows).set_index('Cấu hình').to_string())
 
# ---------------------------------------------------------------- GB so với LR
print(f"\n===== Gradient Boosting - LR (điểm phần trăm, cặp trên {N_SPLITS} lượt) =====")
gb_rows = []
for fs in feature_sets:
    row = {'Cấu hình': fs}
    for met in metrics:
        d = (raw[(fs, 'Gradient Boosting')][met] - raw[(fs, 'LR (mốc)')][met]) * 100
        row[met] = f"{d.mean():+.2f} ± {d.std(ddof=0):.2f} | GB thắng {int((d > 0).sum())}/{N_SPLITS}"
    gb_rows.append(row)
print(pd.DataFrame(gb_rows).set_index('Cấu hình').to_string())
 