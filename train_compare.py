"""
Parte 1: Entrenamiento y comparación de modelos
Random Forest vs SVM, validados con Leave-One-Subject-Out CV
sobre S1-S6. S7 queda reservado como sujeto de prueba final (Parte 2).
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

TEST_SUBJECT = "S7"

df = pd.read_csv("features_df_7subjects.csv")

meta_cols = ["subject_id", "label", "start_idx", "end_idx"]
feature_cols = [c for c in df.columns if c not in meta_cols]

train_df = df[df["subject_id"] != TEST_SUBJECT].reset_index(drop=True)
test_df = df[df["subject_id"] == TEST_SUBJECT].reset_index(drop=True)

X_train_full = train_df[feature_cols]
y_train_full = train_df["label"]
groups = train_df["subject_id"]

print(f"Sujeto de prueba reservado: {TEST_SUBJECT} ({len(test_df)} ventanas, no se toca en esta parte)")
print(f"Sujetos de entrenamiento/CV: {sorted(train_df['subject_id'].unique())} ({len(train_df)} ventanas)")
print(f"Features utilizadas ({len(feature_cols)}): {feature_cols}\n")

models = {
    "Random Forest": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced")),
    ]),
    "SVM (RBF)": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", C=10, gamma="scale", class_weight="balanced")),
    ]),
}

logo = LeaveOneGroupOut()
results = {name: {"acc": [], "f1_macro": [], "precision_macro": [], "recall_macro": []} for name in models}
fold_subjects = sorted(train_df["subject_id"].unique())

for name, pipe in models.items():
    print(f"=== {name} (LOSO-CV sobre S1-S6) ===")
    for fold_idx, (tr_idx, val_idx) in enumerate(logo.split(X_train_full, y_train_full, groups)):
        val_subject = groups.iloc[val_idx].iloc[0]
        X_tr, X_val = X_train_full.iloc[tr_idx], X_train_full.iloc[val_idx]
        y_tr, y_val = y_train_full.iloc[tr_idx], y_train_full.iloc[val_idx]

        pipe.fit(X_tr, y_tr)
        y_pred = pipe.predict(X_val)

        acc = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, average="macro")
        prec = precision_score(y_val, y_pred, average="macro", zero_division=0)
        rec = recall_score(y_val, y_pred, average="macro", zero_division=0)

        results[name]["acc"].append(acc)
        results[name]["f1_macro"].append(f1)
        results[name]["precision_macro"].append(prec)
        results[name]["recall_macro"].append(rec)

        print(f"  Fold (val={val_subject}): acc={acc:.3f}  f1_macro={f1:.3f}  precision={prec:.3f}  recall={rec:.3f}")
    print()

print("=== Resumen (media ± desviación estándar sobre 6 folds) ===")
summary_rows = []
for name in models:
    row = {"Modelo": name}
    for metric in ["acc", "f1_macro", "precision_macro", "recall_macro"]:
        vals = results[name][metric]
        row[metric] = f"{np.mean(vals):.3f} ± {np.std(vals):.3f}"
    summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows)
summary_df.columns = ["Modelo", "Accuracy", "F1-macro", "Precision-macro", "Recall-macro"]
print(summary_df.to_string(index=False))
summary_df.to_csv("comparacion_modelos.csv", index=False)
print("\nGuardado en comparacion_modelos.csv")
