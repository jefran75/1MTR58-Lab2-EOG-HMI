"""
Replica el pipeline de 02_EOG_Procesamiento.ipynb para los 7 sujetos
y arma un features_df combinado.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

DATA_DIR = Path("data")
FS = 256
WINDOW_SEC = 1
OVERLAP = 0.5
WIN_SIZE = int(WINDOW_SEC * FS)
STEP = int(WIN_SIZE * (1 - OVERLAP))

SUBJECTS = [f"S{i}" for i in range(1, 8)]


def bandpass_filter(x, fs=256, lowcut=0.1, highcut=30.0, order=4):
    x = np.asarray(x, dtype=float)
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype="band")
    return filtfilt(b, a, x)


def extract_features(x, prefix):
    x = np.asarray(x, dtype=float)
    return {
        f"{prefix}_mean": np.mean(x),
        f"{prefix}_mav": np.mean(np.abs(x)),
        f"{prefix}_rms": np.sqrt(np.mean(x ** 2)),
        f"{prefix}_std": np.std(x),
        f"{prefix}_max": np.max(x),
        f"{prefix}_min": np.min(x),
        f"{prefix}_ptp": np.ptp(x),
        f"{prefix}_wl": np.sum(np.abs(np.diff(x))),
    }


def process_subject(subject_id):
    eog_path = DATA_DIR / f"{subject_id}-EOG.csv"
    cs_path = DATA_DIR / f"{subject_id}-ControlSignal.csv"

    df_eog = pd.read_csv(eog_path)
    df_cs = pd.read_csv(cs_path)

    df_eog = df_eog.loc[:, ~df_eog.columns.str.contains(r"^Unnamed")].copy()
    df_cs = df_cs.loc[:, ~df_cs.columns.str.contains(r"^Unnamed")].copy()

    df_eog.columns = [c.strip() for c in df_eog.columns]
    df_cs.columns = [c.strip() for c in df_cs.columns]

    if "ControlSignal" not in df_cs.columns:
        if len(df_cs.columns) == 1:
            df_cs.columns = ["ControlSignal"]
        else:
            raise ValueError(f"[{subject_id}] No se encontró la columna ControlSignal.")

    required = {"V1", "V2", "V3", "V4"}
    missing = required - set(df_eog.columns)
    if missing:
        raise ValueError(f"[{subject_id}] Faltan columnas EOG: {sorted(missing)}")

    n = min(len(df_eog), len(df_cs))
    df = pd.concat(
        [
            df_eog.iloc[:n].reset_index(drop=True),
            df_cs[["ControlSignal"]].iloc[:n].reset_index(drop=True),
        ],
        axis=1,
    )

    df["subject_id"] = subject_id
    df["Tiempo"] = np.arange(len(df)) / FS
    df["EOGh"] = df["V3"] - df["V4"]
    df["EOGv"] = df["V1"] - df["V2"]

    df["EOGh_filt"] = bandpass_filter(df["EOGh"], fs=FS)
    df["EOGv_filt"] = bandpass_filter(df["EOGv"], fs=FS)

    # Solo se conservan las clases válidas (1, 2, 3)
    df = df[df["ControlSignal"].isin([1, 2, 3])].copy()

    df["block_id"] = (df["ControlSignal"] != df["ControlSignal"].shift()).cumsum()

    segments = (
        df.groupby("block_id")
        .agg(
            label=("ControlSignal", "first"),
            start_idx=("ControlSignal", lambda x: x.index[0]),
            end_idx=("ControlSignal", lambda x: x.index[-1]),
            n_samples=("ControlSignal", "size"),
        )
        .reset_index(drop=True)
    )

    records = []
    for _, seg in segments.iterrows():
        label = int(seg["label"])
        start_seg = int(seg["start_idx"])
        end_seg = int(seg["end_idx"]) + 1

        if end_seg - start_seg < WIN_SIZE:
            continue

        for start in range(start_seg, end_seg - WIN_SIZE + 1, STEP):
            end = start + WIN_SIZE
            if df.loc[start:end - 1, "ControlSignal"].nunique() != 1:
                continue

            records.append({
                "subject_id": subject_id,
                "label": label,
                "start_idx": start,
                "end_idx": end - 1,
                "EOGh_window": df.loc[start:end - 1, "EOGh_filt"].to_numpy(),
                "EOGv_window": df.loc[start:end - 1, "EOGv_filt"].to_numpy(),
            })

    windows_df = pd.DataFrame(records)

    feature_rows = []
    for _, row in windows_df.iterrows():
        feats = {}
        feats.update(extract_features(row["EOGh_window"], "EOGh"))
        feats.update(extract_features(row["EOGv_window"], "EOGv"))
        feats["subject_id"] = row["subject_id"]
        feats["label"] = row["label"]
        feats["start_idx"] = row["start_idx"]
        feats["end_idx"] = row["end_idx"]
        feature_rows.append(feats)

    return pd.DataFrame(feature_rows)


def main():
    all_dfs = []
    for subject_id in SUBJECTS:
        print(f"Procesando {subject_id}...")
        feats = process_subject(subject_id)
        print(f"  -> {feats.shape[0]} ventanas, distribución: {feats['label'].value_counts().sort_index().to_dict()}")
        all_dfs.append(feats)

    features_df = pd.concat(all_dfs, ignore_index=True)

    # Reordenar columnas: features primero, metadata al final
    meta_cols = ["subject_id", "label", "start_idx", "end_idx"]
    feature_cols = [c for c in features_df.columns if c not in meta_cols]
    features_df = features_df[feature_cols + meta_cols]

    out_path = "features_df_7subjects.csv"
    features_df.to_csv(out_path, index=False)
    print(f"\nTotal: {features_df.shape}")
    print(f"Guardado en {out_path}")
    print("\nVentanas por sujeto y clase:")
    print(features_df.groupby(["subject_id", "label"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()
