"""
Parte 2: Selección y evaluación del modelo final
Random Forest entrenado con S1-S6, evaluado sobre S7 (reservado).
"""
import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report,
)

TEST_SUBJECT = "S7"

df = pd.read_csv("features_df_7subjects.csv")
meta_cols = ["subject_id", "label", "start_idx", "end_idx"]
feature_cols = [c for c in df.columns if c not in meta_cols]

train_df = df[df["subject_id"] != TEST_SUBJECT].reset_index(drop=True)
test_df = df[df["subject_id"] == TEST_SUBJECT].reset_index(drop=True)

X_train = train_df[feature_cols]
y_train = train_df["label"]
X_test = test_df[feature_cols]
y_test = test_df["label"]

final_model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced")),
])

final_model.fit(X_train, y_train)
y_pred = final_model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="macro")
prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
rec = recall_score(y_test, y_pred, average="macro", zero_division=0)

print(f"=== Evaluación final sobre sujeto de prueba {TEST_SUBJECT} ===")
print(f"Accuracy:         {acc:.3f}")
print(f"F1-macro:         {f1:.3f}")
print(f"Precision-macro:  {prec:.3f}")
print(f"Recall-macro:     {rec:.3f}\n")

labels = sorted(y_test.unique())
cm = confusion_matrix(y_test, y_pred, labels=labels)
cm_df = pd.DataFrame(cm, index=[f"Real {l}" for l in labels], columns=[f"Pred {l}" for l in labels])
print("Matriz de confusión:")
print(cm_df.to_string())
print()

print("Classification report:")
print(classification_report(y_test, y_pred, zero_division=0))

cm_df.to_csv("matriz_confusion_S7.csv")

# Exportar modelo final (pipeline con preprocesamiento incluido)
joblib.dump(final_model, "modelo_gripper_rf.joblib")
print("Modelo exportado en modelo_gripper_rf.joblib")

# Verificar que el modelo exportado sea compatible con 03_GripperVirtual.py
# (requiere feature_names_in_)
loaded = joblib.load("modelo_gripper_rf.joblib")
print("\nfeature_names_in_ presente:", hasattr(loaded, "feature_names_in_"))
print("Features:", list(loaded.feature_names_in_))

# Generar un CSV de prueba con las features de S7 para usar con --model en la HMI
test_export = test_df[feature_cols + ["label", "subject_id", "start_idx", "end_idx"]]
test_export.to_csv("S7_features_para_hmi.csv", index=False)
print("\nCSV de prueba para la HMI (modo modelo) guardado en S7_features_para_hmi.csv")
