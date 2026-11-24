#heuristic.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import ( accuracy_score, confusion_matrix, classification_report, roc_auc_score, RocCurveDisplay )


import matplotlib.pyplot as plt
import seaborn as sns

file_path = 'dataset/data.csv'

df = pd.read_csv(file_path, sep=';')



#Flag if one record has ( (number of approved courses / number of enrolled courses) < 0.5, then it is a dropout )

df.columns = df.columns.str.strip()

enrolled_col = 'Curricular units 1st sem (enrolled)'
approved_col = 'Curricular units 1st sem (approved)'

# Ground truth: same binary mapping as the model script
y_true = df['Target'].map({'Dropout': 1, 'Graduate': 0, 'Enrolled': 0})

# Heuristic: approval ratio = approved / enrolled
enrolled = df[enrolled_col]
approved = df[approved_col]

# Students enrolled in 0 units have an undefined ratio -> treat as ratio 0 (flagged)
ratio = np.where(enrolled > 0, approved / enrolled.replace(0, np.nan), 0.0)
df['approval_ratio'] = ratio

# Flag as dropout if ratio < 0.5
y_pred = (df['approval_ratio'] < 0.5).astype(int)

print("Students with 0 enrolled units:", int((enrolled == 0).sum()))
print("Flagged as dropout:", int(y_pred.sum()), "of", len(df))

print("\n========== Heuristic results ==========")
print("Accuracy:", accuracy_score(y_true, y_pred))

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=['Not Dropout', 'Dropout']))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Not Dropout', 'Dropout'],
            yticklabels=['Not Dropout', 'Dropout'])
plt.title('Confusion Matrix (heuristic: approved/enrolled < 0.5)')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.show()
