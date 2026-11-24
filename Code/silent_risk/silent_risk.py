# h_lr_report.py
# Bước 1: Logistic Regression trên tập H + báo cáo
# H = sinh viên có số môn đạt (HK1) >= 4 và điểm (HK1) >= 12
# Đánh giá: 5-fold phân tầng lặp 5 lần, so với tỷ lệ nền 11,04%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# ---------------------------------------------------------------- cấu hình
FILE_PATH = 'dataset/data.csv'
BASE_RATE = 0.1104            # mốc so sánh: tỷ lệ dropout trong H
K_VALUES = [0.10, 0.20]
APPROVED_COL = 'Curricular units 1st sem (approved)'
GRADE_COL = 'Curricular units 1st sem (grade)'

# ---------------------------------------------------------------- dữ liệu
df = pd.read_csv(FILE_PATH, sep=';')
df.columns = df.columns.str.strip()

# Bỏ toàn bộ cột HK2 (dự đoán trước khi HK2 bắt đầu)
sem2_cols = [c for c in df.columns if '2nd sem' in c.lower()]
print("Bỏ các cột HK2:", sem2_cols)
df = df.drop(columns=sem2_cols)

assert APPROVED_COL in df.columns and GRADE_COL in df.columns, "Kiểm tra lại tên cột"

# ---------------------------------------------------------------- tập H
H = df[(df[APPROVED_COL] >= 4) & (df[GRADE_COL] >= 12)].copy()
x_H = H.drop(columns=['Target'])
y_H = H['Target'].map({'Dropout': 1, 'Graduate': 0, 'Enrolled': 0})

print(f"\nTập H: {len(H)} / {len(df)} sinh viên")
print(f"Số dropout trong H: {int(y_H.sum())}")
print(f"Tỷ lệ dropout thực tế trong H: {y_H.mean():.4%}  (mốc so sánh: {BASE_RATE:.2%})")


# ---------------------------------------------------------------- hàm phụ
def top_k_stats(y_true, y_proba, k):
    """(TP trong top k%, kích thước top k%, precision@k, recall@k). k là tỷ lệ (0.1 = 10%)."""
    y_true = np.asarray(y_true)
    n_top = int(np.ceil(k * len(y_true)))
    top_idx = np.argsort(np.asarray(y_proba))[::-1][:n_top]   # xác suất cao nhất trước
    tp = int(y_true[top_idx].sum())
    total_pos = int(y_true.sum())
    return tp, n_top, tp / n_top, (tp / total_pos if total_pos else 0.0)


def make_lr():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight='balanced', solver='lbfgs')
    )


# ---------------------------------------------------------------- CV
rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)

rows = []
for i, (tr, te) in enumerate(rskf.split(x_H, y_H), start=1):
    model = make_lr()
    model.fit(x_H.iloc[tr], y_H.iloc[tr])
    proba = model.predict_proba(x_H.iloc[te])[:, 1]
    y_te = y_H.iloc[te].to_numpy()

    row = {
        'Lượt': i,
        'Lần lặp': (i - 1) // 5 + 1,
        'Fold': (i - 1) % 5 + 1,
        'Test size': len(te),
        'Dropout trong test': int(y_te.sum()),
        'ROC AUC': roc_auc_score(y_te, proba),
    }
    for k in K_VALUES:
        pct = int(k * 100)
        tp, n_top, prec, rec = top_k_stats(y_te, proba, k)
        row[f'Top {pct}% (n)'] = n_top
        row[f'TP@{pct}%'] = tp
        row[f'P@{pct}%'] = prec
        row[f'R@{pct}%'] = rec
    rows.append(row)

cv = pd.DataFrame(rows)

# ---------------------------------------------------------------- báo cáo
pd.set_option('display.width', 220)
pd.set_option('display.max_columns', None)

print("\n========== Kết quả từng lượt (25 lượt) ==========")
print(cv.round(4).to_string(index=False))

print("\n========== Tóm tắt: LR trên tập H (mean ± std trên 25 lượt) ==========")
summary = []
for name in ['ROC AUC'] + [f'{m}@{int(k*100)}%' for k in K_VALUES for m in ('P', 'R')] \
        + [f'TP@{int(k*100)}%' for k in K_VALUES]:
    s = cv[name]
    summary.append({'Chỉ số': name, 'Mean': s.mean(), 'Std': s.std(ddof=0),
                    'Min': s.min(), 'Max': s.max()})
summary_df = pd.DataFrame(summary).set_index('Chỉ số')
print(summary_df.round(4).to_string())

print(f"\n========== So với tỷ lệ nền {BASE_RATE:.2%} ==========")
for k in K_VALUES:
    pct = int(k * 100)
    p = cv[f'P@{pct}%']
    lift = p / BASE_RATE
    print(f"P@{pct}%: {p.mean():.2%} ± {p.std(ddof=0):.2%}  |  "
          f"hơn mốc {(p.mean() - BASE_RATE) * 100:+.2f} điểm %  |  "
          f"lift = {lift.mean():.2f}x ± {lift.std(ddof=0):.2f}  |  "
          f"cao hơn mốc ở {int((p > BASE_RATE).sum())}/{len(cv)} lượt")

# ---------------------------------------------------------------- hệ số LR (fit trên toàn bộ H, để tham khảo)
final = make_lr().fit(x_H, y_H)
coefs = pd.DataFrame({
    'Feature': x_H.columns,
    'Hệ số (đã chuẩn hóa)': final[-1].coef_[0],
}).assign(abs_coef=lambda d: d['Hệ số (đã chuẩn hóa)'].abs()) \
 .sort_values('abs_coef', ascending=False).drop(columns='abs_coef')

print("\n========== 15 đặc trưng có hệ số lớn nhất (fit trên toàn H) ==========")
print(coefs.head(15).round(4).to_string(index=False))

# ---------------------------------------------------------------- biểu đồ
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, k in zip(axes, K_VALUES):
    pct = int(k * 100)
    ax.boxplot(cv[f'P@{pct}%'], widths=0.5)
    ax.axhline(BASE_RATE, color='red', linestyle='--', label=f'Mốc {BASE_RATE:.2%}')
    ax.set_title(f'P@{pct}% trong H (25 lượt)')
    ax.set_xticks([])
    ax.set_ylim(0, max(0.5, cv[f'P@{pct}%'].max() * 1.1))
    ax.legend()
plt.tight_layout()
plt.show()

cv.to_csv('h_lr_per_split.csv', index=False)
summary_df.to_csv('h_lr_summary.csv')