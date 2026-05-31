# -*- coding: utf-8 -*-
"""
analysis_core.py
================
Colab not defterindeki (Untitled9.ipynb) anomali / fraud tespit
mantiginin BIREBIR korunmus halidir.

Modeller, parametreler, on isleme adimlari, sentetik fraud testi ve
tum hesaplamalar orijinal kod ile AYNIDIR. Yalnizca Colab'a ozgu
girdi/cikti cagrilari (files.upload, files.download, display) masaustu
arayuzunden cagrilabilen fonksiyonlara donusturulmustur.

Grafikler, PyQt arayuzune gomulebilmesi icin pyplot yerine acik Figure
nesneleri uzerinde ayni icerikle (ayni veriler, ayni baslik/etiketler)
yeniden uretilir.
"""

import os
import random
import time
import unicodedata
import warnings

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import EarlyStopping, Callback
from tensorflow.keras.optimizers import Adam

warnings.filterwarnings("ignore")

# ============================================================
# 1. SABIT PARAMETRELER VE TEKRAR URETILEBILIRLIK
# ============================================================

RANDOM_STATE = 42
CONTAMINATION = 0.01
SYNTHETIC_FRAUD_RATE = 0.01


def set_global_seeds():
    """Orijinal koddaki tekrar uretilebilirlik ayarlarini uygular."""
    os.environ["PYTHONHASHSEED"] = str(RANDOM_STATE)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    tf.random.set_seed(RANDOM_STATE)


set_global_seeds()


# ============================================================
# 2. SUTUN ADLARINI TEMIZLEME
# ============================================================

def normalize_column_name(col):
    """
    Sutun adlarini sadelestirir:
    - Turkce karakterleri donusturur
    - Buyuk harfe cevirir
    - Bosluk ve ozel karakterleri alt cizgi yapar
    """
    col = str(col).strip()

    replacements = {
        "ı": "i", "İ": "I",
        "ğ": "g", "Ğ": "G",
        "ü": "u", "Ü": "U",
        "ş": "s", "Ş": "S",
        "ö": "o", "Ö": "O",
        "ç": "c", "Ç": "C"
    }

    for tr_char, en_char in replacements.items():
        col = col.replace(tr_char, en_char)

    col = unicodedata.normalize("NFKD", col)
    col = "".join([c for c in col if not unicodedata.combining(c)])

    col = col.upper()

    for ch in [" ", "-", "/", "\\", ".", ",", "(", ")", "[", "]"]:
        col = col.replace(ch, "_")

    while "__" in col:
        col = col.replace("__", "_")

    col = col.strip("_")
    return col


def clean_columns(df_raw):
    """Ham veri uzerinde sutun adi temizleme uygular (orijinal bolum 3)."""
    df = df_raw.copy()
    old_columns = df.columns.tolist()
    df.columns = [normalize_column_name(c) for c in df.columns]

    column_mapping = pd.DataFrame({
        "ORIJINAL_SUTUN_ADI": old_columns,
        "YENI_SUTUN_ADI": df.columns
    })
    return df, column_mapping


# ============================================================
# 3. SAYISAL DONUSUM
# ============================================================

def to_numeric_frame(df):
    """Orijinal bolum 4: tum sutunlari sayisala donusturur."""
    df_numeric = df.copy()

    for col in df_numeric.columns:
        if df_numeric[col].dtype == "object":
            df_numeric[col] = (
                df_numeric[col]
                .astype(str)
                .str.strip()
                .str.replace(",", ".", regex=False)
                .str.replace(" ", "", regex=False)
            )
            df_numeric[col] = pd.to_numeric(df_numeric[col], errors="coerce")
        else:
            df_numeric[col] = pd.to_numeric(df_numeric[col], errors="coerce")

    df_numeric = df_numeric.replace([np.inf, -np.inf], np.nan)
    return df_numeric


# ============================================================
# 4. DEGISKEN GRUPLARI (orijinal bolum 6)
# ============================================================

id_columns = [
    "POLICE_NO",
    "PROVIZYON_NO"
]

numeric_features = [
    "YENILEME_NO",
    "POL_BAS_TAR",
    "POL_BAS_TAN_FRK",
    "SIGORTALI_YASI",
    "POL_BAS_HASAR_SURE",
    "SKYT_HASAR_SURE",
    "HASAR_IHBAR_SURE",
    "POL_BAS_ODEME_SURE",
    "TALEP_TUTAR",
    "ODENEN_TUTAR"
]

categorical_features = [
    "POLICE_TIPI",
    "ACENTE_NO",
    "ACENTE_BOLGE",
    "CINSIYET",
    "BRY_TIP",
    "PAKET_ID",
    "HASAR_STATU",
    "HASAR_KAYNAGI",
    "ICMAL_STATUSU",
    "ANA_TEMINAT_ADI",
    "ALT_TEMINAT",
    "ISLEM_TIPI",
    "DOKTOR_BRANS",
    "DOKTOR_ID",
    "ICD_KODU",
    "KURUM_TIPI",
    "KURUM_KODU",
    "KURUM_IL",
    "KURUM_ILCE",
    "BANKA_ADI",
    "SUBE_ADI",
    "IBAN_NO",
    "ANA_TEMINAT_KODU",
    "TEMINAT_KODU",
    "VAKA_TIPI"
]


# ============================================================
# 5. MODEL MATRISI HAZIRLAMA (orijinal bolum 7)
# ============================================================

def prepare_model_matrix(
    input_df,
    numeric_features,
    categorical_features,
    scale_data=True
):
    """Modelleme icin veri matrisi hazirlar (orijinal kod ile birebir)."""

    df_prep = input_df.copy()
    df_prep = df_prep.replace([np.inf, -np.inf], np.nan)

    numeric_existing = [c for c in numeric_features if c in df_prep.columns]
    categorical_existing = [c for c in categorical_features if c in df_prep.columns]

    for col in numeric_existing:
        median_value = df_prep[col].median()
        df_prep[col] = df_prep[col].fillna(median_value)

    for col in categorical_existing:
        df_prep[col] = df_prep[col].fillna(-999999)

    df_model = pd.DataFrame(index=df_prep.index)

    for col in numeric_existing:
        df_model[col] = df_prep[col]

    freq_encoded_columns = []
    for col in categorical_existing:
        freq_map = df_prep[col].value_counts(normalize=True, dropna=False)
        new_col = col + "_FREQ"
        df_model[new_col] = df_prep[col].map(freq_map)
        freq_encoded_columns.append(new_col)

    df_model = df_model.replace([np.inf, -np.inf], np.nan)

    if df_model.isna().sum().sum() > 0:
        df_model = df_model.fillna(df_model.median())

    scaler = None
    if scale_data:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_model)
        X_scaled_df = pd.DataFrame(
            X_scaled,
            columns=df_model.columns,
            index=df_model.index
        )
    else:
        X_scaled_df = df_model.copy()

    return df_prep, df_model, X_scaled_df, scaler


# ============================================================
# 6. YARDIMCI MODELLEME FONKSIYONLARI (orijinal bolum 9)
# ============================================================

def mark_top_anomalies(score_array, contamination=0.01):
    """En yuksek skorlu kayitlari anomali isaretler (yuksek skor = anomalik)."""
    score_array = np.asarray(score_array)
    n = len(score_array)
    n_top = int(np.ceil(n * contamination))

    anomaly_indices = np.argsort(score_array)[-n_top:]

    labels = np.zeros(n, dtype=int)
    labels[anomaly_indices] = 1

    threshold = np.min(score_array[anomaly_indices])
    return labels, threshold


def anomaly_summary(labels, model_name):
    """Model bazinda anomali sayisi ve oranini ozetler."""
    count = int(np.sum(labels))
    ratio = count / len(labels)

    return pd.DataFrame({
        "MODEL": [model_name],
        "ANOMALI_SAYISI": [count],
        "ANOMALI_ORANI": [ratio],
        "ANOMALI_ORANI_YUZDE": [ratio * 100]
    })


# ============================================================
# 7. MODELLER (orijinal bolum 10.1 - 10.3)
# ============================================================

def run_isolation_forest(X, contamination=0.01, random_state=42):
    """Isolation Forest (yuksek skor = anomalik)."""
    start_time = time.time()

    model = IsolationForest(
        n_estimators=300,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(X)

    scores = -model.decision_function(X)
    labels, threshold = mark_top_anomalies(scores, contamination=contamination)
    elapsed_time = time.time() - start_time

    return {
        "model": model,
        "scores": scores,
        "labels": labels,
        "threshold": threshold,
        "elapsed_time": elapsed_time
    }


def run_lof(X, contamination=0.01):
    """Local Outlier Factor (yuksek skor = anomalik)."""
    start_time = time.time()

    model = LocalOutlierFactor(
        n_neighbors=20,
        contamination=contamination,
        metric="euclidean",
        n_jobs=-1
    )
    model.fit_predict(X)

    scores = -model.negative_outlier_factor_
    labels, threshold = mark_top_anomalies(scores, contamination=contamination)
    elapsed_time = time.time() - start_time

    return {
        "model": model,
        "scores": scores,
        "labels": labels,
        "threshold": threshold,
        "elapsed_time": elapsed_time
    }


class _EpochProgress(Callback):
    """Egitim ilerlemesini arayuze bildirmek icin yardimci callback.

    Modelin mimarisini/egitimini DEGISTIRMEZ; yalnizca her epoch sonunda
    bir ilerleme mesaji iletir (orijinaldeki verbose ciktisinin yerine).
    """
    def __init__(self, total_epochs, progress_callback=None,
                 base=0.0, span=1.0, phase=""):
        super().__init__()
        self.total_epochs = total_epochs
        self.progress_callback = progress_callback
        self.base = base
        self.span = span
        self.phase = phase

    def on_epoch_end(self, epoch, logs=None):
        if self.progress_callback is None:
            return
        logs = logs or {}
        frac = (epoch + 1) / float(self.total_epochs)
        pct = self.base + self.span * frac
        loss = logs.get("loss", 0.0)
        val_loss = logs.get("val_loss", 0.0)
        self.progress_callback(
            pct,
            "%sAutoencoder egitimi - epoch %d (loss=%.5f, val_loss=%.5f)"
            % (self.phase, epoch + 1, loss, val_loss)
        )


def run_autoencoder(X, contamination=0.01, random_state=42,
                    progress_callback=None, prog_base=0.0,
                    prog_span=1.0, phase=""):
    """
    Autoencoder (orijinal bolum 10.3) - mimari ve egitim birebir korunmustur.
    input_dim -> 24 -> 12 -> 6 -> 12 -> 24 -> input_dim
    Skor: rekonstruksiyon MSE hatasi (yuksek skor = anomalik).
    """
    start_time = time.time()

    random.seed(random_state)
    np.random.seed(random_state)
    tf.random.set_seed(random_state)

    X_ae = X.values.astype("float32") if isinstance(X, pd.DataFrame) else X.astype("float32")
    input_dim = X_ae.shape[1]

    input_layer = Input(shape=(input_dim,), name="input_layer")

    encoded = Dense(24, activation="relu", name="encoder_dense_1")(input_layer)
    encoded = Dense(12, activation="relu", name="encoder_dense_2")(encoded)
    bottleneck = Dense(6, activation="relu", name="bottleneck")(encoded)

    decoded = Dense(12, activation="relu", name="decoder_dense_1")(bottleneck)
    decoded = Dense(24, activation="relu", name="decoder_dense_2")(decoded)
    output_layer = Dense(input_dim, activation="linear", name="output_layer")(decoded)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )

    callbacks = [early_stop]
    if progress_callback is not None:
        callbacks.append(_EpochProgress(
            total_epochs=50,
            progress_callback=progress_callback,
            base=prog_base,
            span=prog_span,
            phase=phase
        ))

    history = model.fit(
        X_ae,
        X_ae,
        epochs=50,
        batch_size=256,
        validation_split=0.10,
        shuffle=False,
        callbacks=callbacks,
        verbose=0
    )

    X_pred = model.predict(X_ae, batch_size=256, verbose=0)
    scores = np.mean(np.square(X_ae - X_pred), axis=1)

    labels, threshold = mark_top_anomalies(scores, contamination=contamination)
    elapsed_time = time.time() - start_time

    return {
        "model": model,
        "history": history,
        "scores": scores,
        "labels": labels,
        "threshold": threshold,
        "elapsed_time": elapsed_time
    }


# ============================================================
# 8. JACCARD (orijinal bolum 19)
# ============================================================

def jaccard_similarity(set_a, set_b):
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    if union == 0:
        return 0
    return intersection / union


def _emit(progress_callback, pct, msg):
    if progress_callback is not None:
        progress_callback(pct, msg)


# ============================================================
# 9. GERCEK VERI ANALIZ HATTI (orijinal bolum 2-24)
# ============================================================

def run_full_analysis(df_raw, progress_callback=None):
    """
    Ham veri cercevesi (Excel'den okunmus) alir; orijinal not defterindeki
    tum gercek veri analizini calistirir ve sonuclari sozluk olarak dondurur.
    """
    set_global_seeds()

    _emit(progress_callback, 0.02, "Sutun adlari temizleniyor...")
    df, column_mapping = clean_columns(df_raw)

    _emit(progress_callback, 0.05, "Sayisal donusum yapiliyor...")
    df_numeric = to_numeric_frame(df)

    # Bolum 5: boyut / eksik deger kontrolu
    expected_rows = 164155
    expected_cols = 37
    missing_summary = pd.DataFrame({
        "EKSIK_SAYISI": df_numeric.isna().sum(),
        "EKSIK_ORANI": df_numeric.isna().mean()
    }).sort_values("EKSIK_SAYISI", ascending=False).reset_index().rename(
        columns={"index": "SUTUN"}
    )

    # Bolum 6: sutun uyum kontrolu
    all_defined_columns = id_columns + numeric_features + categorical_features
    missing_defined_columns = [c for c in all_defined_columns if c not in df_numeric.columns]
    extra_columns = [c for c in df_numeric.columns if c not in all_defined_columns]

    _emit(progress_callback, 0.10, "Model matrisi hazirlaniyor...")
    df_prep, df_model, X_scaled_df, scaler = prepare_model_matrix(
        input_df=df_numeric,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        scale_data=True
    )
    analysis_df = df_prep.copy()

    # ---- Isolation Forest (bolum 11.1) ----
    _emit(progress_callback, 0.15, "Isolation Forest calisiyor...")
    if_result = run_isolation_forest(
        X=X_scaled_df, contamination=CONTAMINATION, random_state=RANDOM_STATE
    )
    analysis_df["IF_SCORE"] = if_result["scores"]
    analysis_df["IF_ANOMALY"] = if_result["labels"]

    # ---- Local Outlier Factor (bolum 11.2) ----
    _emit(progress_callback, 0.30, "Local Outlier Factor calisiyor...")
    lof_result = run_lof(X=X_scaled_df, contamination=CONTAMINATION)
    analysis_df["LOF_SCORE"] = lof_result["scores"]
    analysis_df["LOF_ANOMALY"] = lof_result["labels"]

    # ---- Autoencoder (bolum 11.3) ----
    _emit(progress_callback, 0.45, "Autoencoder egitiliyor...")
    ae_result = run_autoencoder(
        X=X_scaled_df, contamination=CONTAMINATION, random_state=RANDOM_STATE,
        progress_callback=progress_callback, prog_base=0.45, prog_span=0.25,
        phase="Gercek veri - "
    )
    analysis_df["AE_SCORE"] = ae_result["scores"]
    analysis_df["AE_ANOMALY"] = ae_result["labels"]

    _emit(progress_callback, 0.72, "Ozet tablolar olusturuluyor...")

    # ---- Model ozeti (bolum 12) ----
    model_summary = pd.concat([
        anomaly_summary(analysis_df["IF_ANOMALY"].values, "Isolation Forest"),
        anomaly_summary(analysis_df["LOF_ANOMALY"].values, "Local Outlier Factor"),
        anomaly_summary(analysis_df["AE_ANOMALY"].values, "Autoencoder")
    ], ignore_index=True)

    # ---- Skor ozeti (bolum 14) ----
    score_summary = analysis_df[["IF_SCORE", "LOF_SCORE", "AE_SCORE"]].describe().T
    score_summary["SKEWNESS"] = analysis_df[["IF_SCORE", "LOF_SCORE", "AE_SCORE"]].skew()
    score_summary["KURTOSIS"] = analysis_df[["IF_SCORE", "LOF_SCORE", "AE_SCORE"]].kurtosis()
    score_summary = score_summary.reset_index().rename(columns={"index": "SKOR"})

    # ---- Cakisma degiskenleri (bolum 16) ----
    analysis_df["ANOMALY_MODEL_COUNT"] = (
        analysis_df["IF_ANOMALY"] + analysis_df["LOF_ANOMALY"] + analysis_df["AE_ANOMALY"]
    )

    def anomaly_combination(row):
        models = []
        if row["IF_ANOMALY"] == 1:
            models.append("IF")
        if row["LOF_ANOMALY"] == 1:
            models.append("LOF")
        if row["AE_ANOMALY"] == 1:
            models.append("AE")
        if len(models) == 0:
            return "Normal"
        return " + ".join(models)

    analysis_df["ANOMALY_COMBINATION"] = analysis_df.apply(anomaly_combination, axis=1)

    # ---- Genel cakisma ozeti (bolum 17) ----
    n_total = len(analysis_df)
    if_set = set(analysis_df.index[analysis_df["IF_ANOMALY"] == 1])
    lof_set = set(analysis_df.index[analysis_df["LOF_ANOMALY"] == 1])
    ae_set = set(analysis_df.index[analysis_df["AE_ANOMALY"] == 1])

    if_lof = if_set & lof_set
    if_ae = if_set & ae_set
    lof_ae = lof_set & ae_set
    all_three = if_set & lof_set & ae_set
    at_least_two = (if_set & lof_set) | (if_set & ae_set) | (lof_set & ae_set)
    any_model = if_set | lof_set | ae_set

    overlap_summary = pd.DataFrame({
        "KATEGORI": [
            "IF anomalileri", "LOF anomalileri", "AE anomalileri",
            "IF n LOF", "IF n AE", "LOF n AE", "IF n LOF n AE",
            "En az iki model", "En az bir model"
        ],
        "KAYIT_SAYISI": [
            len(if_set), len(lof_set), len(ae_set),
            len(if_lof), len(if_ae), len(lof_ae), len(all_three),
            len(at_least_two), len(any_model)
        ]
    })
    overlap_summary["TOPLAM_VERIYE_ORANI"] = overlap_summary["KAYIT_SAYISI"] / n_total
    overlap_summary["TOPLAM_VERIYE_ORANI_YUZDE"] = overlap_summary["TOPLAM_VERIYE_ORANI"] * 100

    # ---- Kombinasyon ozeti (bolum 18) ----
    combination_summary = (
        analysis_df["ANOMALY_COMBINATION"]
        .value_counts()
        .rename_axis("ANOMALI_KOMBINASYONU")
        .reset_index(name="KAYIT_SAYISI")
    )
    combination_summary["ORAN"] = combination_summary["KAYIT_SAYISI"] / len(analysis_df)
    combination_summary["ORAN_YUZDE"] = combination_summary["ORAN"] * 100

    # ---- Jaccard (bolum 19) ----
    jaccard_summary = pd.DataFrame({
        "MODEL_CIFTI": [
            "Isolation Forest - LOF",
            "Isolation Forest - Autoencoder",
            "LOF - Autoencoder"
        ],
        "JACCARD_BENZERLIGI": [
            jaccard_similarity(if_set, lof_set),
            jaccard_similarity(if_set, ae_set),
            jaccard_similarity(lof_set, ae_set)
        ]
    })

    # ---- Cakisma & Jaccard matrisleri (bolum 20, 22) ----
    models = ["IF", "LOF", "AE"]
    sets = {"IF": if_set, "LOF": lof_set, "AE": ae_set}

    overlap_matrix = pd.DataFrame(index=models, columns=models, dtype=int)
    jaccard_matrix = pd.DataFrame(index=models, columns=models, dtype=float)
    for m1 in models:
        for m2 in models:
            overlap_matrix.loc[m1, m2] = len(sets[m1] & sets[m2])
            jaccard_matrix.loc[m1, m2] = jaccard_similarity(sets[m1], sets[m2])

    # ---- Guclu anomali adaylari (bolum 23) ----
    strong_anomalies = analysis_df[analysis_df["ANOMALY_MODEL_COUNT"] >= 2].copy()
    strong_anomalies = strong_anomalies.sort_values(
        ["ANOMALY_MODEL_COUNT", "IF_SCORE", "LOF_SCORE", "AE_SCORE"],
        ascending=[False, False, False, False]
    )

    # ---- Calisma sureleri (bolum 24) ----
    runtime_summary = pd.DataFrame({
        "MODEL": ["Isolation Forest", "Local Outlier Factor", "Autoencoder"],
        "CALISMA_SURESI_SANIYE": [
            round(if_result["elapsed_time"], 2),
            round(lof_result["elapsed_time"], 2),
            round(ae_result["elapsed_time"], 2)
        ]
    })

    # En anomalik ilk 20 kayit icin gosterilecek sutunlar (bolum 15)
    show_columns = [c for c in [
        "POLICE_NO", "PROVIZYON_NO", "TALEP_TUTAR", "ODENEN_TUTAR",
        "POL_BAS_HASAR_SURE", "SKYT_HASAR_SURE", "HASAR_IHBAR_SURE",
        "POL_BAS_ODEME_SURE", "DOKTOR_ID", "ICD_KODU", "KURUM_KODU"
    ] if c in analysis_df.columns]

    top_if = analysis_df.sort_values("IF_SCORE", ascending=False)[
        show_columns + ["IF_SCORE", "IF_ANOMALY"]].head(20)
    top_lof = analysis_df.sort_values("LOF_SCORE", ascending=False)[
        show_columns + ["LOF_SCORE", "LOF_ANOMALY"]].head(20)
    top_ae = analysis_df.sort_values("AE_SCORE", ascending=False)[
        show_columns + ["AE_SCORE", "AE_ANOMALY"]].head(20)

    _emit(progress_callback, 0.78, "Gercek veri analizi tamamlandi.")

    return {
        "df_numeric": df_numeric,
        "column_mapping": column_mapping,
        "missing_summary": missing_summary,
        "shape": df_numeric.shape,
        "expected_shape": (expected_rows, expected_cols),
        "missing_defined_columns": missing_defined_columns,
        "extra_columns": extra_columns,
        "analysis_df": analysis_df,
        "if_result": if_result,
        "lof_result": lof_result,
        "ae_result": ae_result,
        "model_summary": model_summary,
        "score_summary": score_summary,
        "overlap_summary": overlap_summary,
        "combination_summary": combination_summary,
        "jaccard_summary": jaccard_summary,
        "overlap_matrix": overlap_matrix,
        "jaccard_matrix": jaccard_matrix,
        "strong_anomalies": strong_anomalies,
        "runtime_summary": runtime_summary,
        "top_if": top_if,
        "top_lof": top_lof,
        "top_ae": top_ae,
        "show_columns": show_columns,
    }


# ============================================================
# 10. SENTETIK FRAUD TESTI YARDIMCILARI (orijinal bolum 27-28)
# ============================================================

def get_rare_values(df, column, n_values=20):
    counts = df[column].value_counts(dropna=False)
    return counts.sort_values(ascending=True).head(n_values).index.tolist()


def safe_quantile(df, column, q, fallback_value=0):
    if column in df.columns:
        return df[column].quantile(q)
    return fallback_value


def safe_rare_values(df, column, n_values=20, fallback_value=-999999):
    if column in df.columns:
        return get_rare_values(df, column, n_values=n_values)
    return [fallback_value]


def apply_synthetic_scenarios(synthetic_df, talep_q95, talep_q99, talep_q995,
                              rare_doctors, rare_institutions, rare_icd,
                              random_state=42):
    """Sentetik fraud senaryolarini uygular (orijinal bolum 30, birebir)."""
    df_synth = synthetic_df.copy()
    rng = np.random.default_rng(random_state)

    # Senaryo 1: Erken hasar + yuksek talep
    mask_s1 = df_synth["SENTETIK_SENARYO"] == "S1_EARLY_HIGH_CLAIM"
    n_mask_s1 = mask_s1.sum()
    if n_mask_s1 > 0:
        if "POL_BAS_HASAR_SURE" in df_synth.columns:
            df_synth.loc[mask_s1, "POL_BAS_HASAR_SURE"] = rng.uniform(0, 2, n_mask_s1)
        if "SKYT_HASAR_SURE" in df_synth.columns:
            df_synth.loc[mask_s1, "SKYT_HASAR_SURE"] = rng.uniform(0, 2, n_mask_s1)
        if "HASAR_IHBAR_SURE" in df_synth.columns:
            df_synth.loc[mask_s1, "HASAR_IHBAR_SURE"] = rng.uniform(0, 1, n_mask_s1)
        if "TALEP_TUTAR" in df_synth.columns:
            df_synth.loc[mask_s1, "TALEP_TUTAR"] = rng.uniform(
                talep_q99, talep_q995 * 1.25, n_mask_s1)
        if "ODENEN_TUTAR" in df_synth.columns and "TALEP_TUTAR" in df_synth.columns:
            payment_ratio_s1 = rng.uniform(0.90, 1.00, n_mask_s1)
            df_synth.loc[mask_s1, "ODENEN_TUTAR"] = (
                df_synth.loc[mask_s1, "TALEP_TUTAR"].values * payment_ratio_s1)

    # Senaryo 2: Yuksek odeme orani + hizli odeme
    mask_s2 = df_synth["SENTETIK_SENARYO"] == "S2_HIGH_PAYMENT_FAST"
    n_mask_s2 = mask_s2.sum()
    if n_mask_s2 > 0:
        if "TALEP_TUTAR" in df_synth.columns:
            df_synth.loc[mask_s2, "TALEP_TUTAR"] = rng.uniform(
                talep_q95, talep_q99 * 1.50, n_mask_s2)
        if "ODENEN_TUTAR" in df_synth.columns and "TALEP_TUTAR" in df_synth.columns:
            df_synth.loc[mask_s2, "ODENEN_TUTAR"] = df_synth.loc[mask_s2, "TALEP_TUTAR"]
        if "POL_BAS_ODEME_SURE" in df_synth.columns:
            df_synth.loc[mask_s2, "POL_BAS_ODEME_SURE"] = rng.uniform(0, 3, n_mask_s2)
        if "HASAR_IHBAR_SURE" in df_synth.columns:
            df_synth.loc[mask_s2, "HASAR_IHBAR_SURE"] = rng.uniform(0, 1, n_mask_s2)

    # Senaryo 3: Nadir kombinasyon + yuksek tutar
    mask_s3 = df_synth["SENTETIK_SENARYO"] == "S3_RARE_COMBINATION_HIGH_AMOUNT"
    n_mask_s3 = mask_s3.sum()
    if n_mask_s3 > 0:
        if "DOKTOR_ID" in df_synth.columns:
            df_synth.loc[mask_s3, "DOKTOR_ID"] = rng.choice(
                rare_doctors, size=n_mask_s3, replace=True)
        if "KURUM_KODU" in df_synth.columns:
            df_synth.loc[mask_s3, "KURUM_KODU"] = rng.choice(
                rare_institutions, size=n_mask_s3, replace=True)
        if "ICD_KODU" in df_synth.columns:
            df_synth.loc[mask_s3, "ICD_KODU"] = rng.choice(
                rare_icd, size=n_mask_s3, replace=True)
        if "TALEP_TUTAR" in df_synth.columns:
            df_synth.loc[mask_s3, "TALEP_TUTAR"] = rng.uniform(
                talep_q99, talep_q995 * 1.50, n_mask_s3)
        if "ODENEN_TUTAR" in df_synth.columns and "TALEP_TUTAR" in df_synth.columns:
            payment_ratio_s3 = rng.uniform(0.80, 1.00, n_mask_s3)
            df_synth.loc[mask_s3, "ODENEN_TUTAR"] = (
                df_synth.loc[mask_s3, "TALEP_TUTAR"].values * payment_ratio_s3)

    return df_synth


def synthetic_detection_summary(df, anomaly_col, model_name):
    """Model bazinda sentetik fraud tespit ozeti (orijinal bolum 35)."""
    synthetic_records = df[df["SENTETIK_FRAUD"] == 1]
    original_records = df[df["SENTETIK_FRAUD"] == 0]

    total_synthetic = len(synthetic_records)
    detected_synthetic = int(synthetic_records[anomaly_col].sum())
    detection_rate = detected_synthetic / total_synthetic if total_synthetic > 0 else 0

    original_flagged = int(original_records[anomaly_col].sum())
    original_flagged_rate = original_flagged / len(original_records) if len(original_records) > 0 else 0

    return pd.DataFrame({
        "MODEL": [model_name],
        "TOPLAM_SENTETIK_FRAUD": [total_synthetic],
        "TESPIT_EDILEN_SENTETIK_FRAUD": [detected_synthetic],
        "SENTETIK_FRAUD_TESPIT_ORANI": [detection_rate],
        "ORIJINAL_KAYITLARDA_ANOMALI_SAYISI": [original_flagged],
        "ORIJINAL_KAYITLARDA_ANOMALI_ORANI": [original_flagged_rate],
        "SENTETIK_FRAUD_TESPIT_ORANI_YUZDE": [detection_rate * 100],
        "ORIJINAL_KAYITLARDA_ANOMALI_ORANI_YUZDE": [original_flagged_rate * 100]
    })


def scenario_detection_summary(df, anomaly_col, model_name):
    """Senaryo bazinda sentetik fraud tespit ozeti (orijinal bolum 35)."""
    synthetic_records = df[df["SENTETIK_FRAUD"] == 1].copy()
    summary = (
        synthetic_records
        .groupby("SENTETIK_SENARYO")
        .agg(
            TOPLAM_SENTETIK_FRAUD=("SENTETIK_FRAUD", "count"),
            TESPIT_EDILEN_SENTETIK_FRAUD=(anomaly_col, "sum")
        )
        .reset_index()
    )
    summary["MODEL"] = model_name
    summary["SENTETIK_FRAUD_TESPIT_ORANI"] = (
        summary["TESPIT_EDILEN_SENTETIK_FRAUD"] / summary["TOPLAM_SENTETIK_FRAUD"])
    summary["SENTETIK_FRAUD_TESPIT_ORANI_YUZDE"] = (
        summary["SENTETIK_FRAUD_TESPIT_ORANI"] * 100)
    return summary[[
        "MODEL", "SENTETIK_SENARYO", "TOPLAM_SENTETIK_FRAUD",
        "TESPIT_EDILEN_SENTETIK_FRAUD", "SENTETIK_FRAUD_TESPIT_ORANI",
        "SENTETIK_FRAUD_TESPIT_ORANI_YUZDE"
    ]]


# ============================================================
# 11. SENTETIK FRAUD TESTI HATTI (orijinal bolum 26-46)
# ============================================================

def run_synthetic_test(df_numeric, progress_callback=None):
    """
    Orijinal not defterindeki sentetik fraud testini birebir calistirir.
    df_numeric: gercek veri analizinde uretilen sayisal veri cercevesi.
    """
    set_global_seeds()

    df_original_for_synth = df_numeric.copy()
    n_original = len(df_original_for_synth)
    n_synthetic = int(np.ceil(n_original * SYNTHETIC_FRAUD_RATE))

    _emit(progress_callback, 0.80, "Sentetik fraud kayitlari olusturuluyor...")

    synthetic_sample_indices = (
        df_original_for_synth.sample(n=n_synthetic, random_state=RANDOM_STATE).index
    )
    synthetic_df = df_original_for_synth.loc[synthetic_sample_indices].copy()
    synthetic_df["ORIGINAL_INDEX"] = synthetic_sample_indices

    # Yardimci degerler (bolum 28)
    talep_q95 = safe_quantile(df_original_for_synth, "TALEP_TUTAR", 0.95)
    talep_q99 = safe_quantile(df_original_for_synth, "TALEP_TUTAR", 0.99)
    talep_q995 = safe_quantile(df_original_for_synth, "TALEP_TUTAR", 0.995)

    if pd.isna(talep_q95):
        talep_q95 = 0
    if pd.isna(talep_q99):
        talep_q99 = talep_q95
    if pd.isna(talep_q995):
        talep_q995 = talep_q99
    if talep_q995 <= talep_q99:
        talep_q995 = talep_q99 * 1.10 if talep_q99 > 0 else 1

    rare_doctors = safe_rare_values(df_original_for_synth, "DOKTOR_ID", n_values=20)
    rare_institutions = safe_rare_values(df_original_for_synth, "KURUM_KODU", n_values=20)
    rare_icd = safe_rare_values(df_original_for_synth, "ICD_KODU", n_values=20)

    # Senaryolara bolme (bolum 29)
    synthetic_df = synthetic_df.reset_index(drop=True)
    n_s1 = n_synthetic // 3
    n_s2 = n_synthetic // 3
    n_s3 = n_synthetic - n_s1 - n_s2
    scenario_labels = (
        ["S1_EARLY_HIGH_CLAIM"] * n_s1 +
        ["S2_HIGH_PAYMENT_FAST"] * n_s2 +
        ["S3_RARE_COMBINATION_HIGH_AMOUNT"] * n_s3
    )
    synthetic_df["SENTETIK_SENARYO"] = scenario_labels

    # Senaryolari uygula (bolum 31)
    synthetic_df = apply_synthetic_scenarios(
        synthetic_df=synthetic_df,
        talep_q95=talep_q95, talep_q99=talep_q99, talep_q995=talep_q995,
        rare_doctors=rare_doctors, rare_institutions=rare_institutions,
        rare_icd=rare_icd, random_state=RANDOM_STATE
    )
    synthetic_df["SENTETIK_FRAUD"] = 1

    # Orijinal + sentetik birlestirme (bolum 32)
    original_for_synthetic_test = df_original_for_synth.copy()
    original_for_synthetic_test["SENTETIK_FRAUD"] = 0
    original_for_synthetic_test["SENTETIK_SENARYO"] = "ORIGINAL"
    original_for_synthetic_test["ORIGINAL_INDEX"] = original_for_synthetic_test.index

    for col in original_for_synthetic_test.columns:
        if col not in synthetic_df.columns:
            synthetic_df[col] = np.nan
    for col in synthetic_df.columns:
        if col not in original_for_synthetic_test.columns:
            original_for_synthetic_test[col] = np.nan
    synthetic_df = synthetic_df[original_for_synthetic_test.columns]

    df_synthetic_test = pd.concat(
        [original_for_synthetic_test, synthetic_df], axis=0, ignore_index=True
    )

    # Model matrisi (bolum 33)
    _emit(progress_callback, 0.83, "Sentetik test model matrisi hazirlaniyor...")
    df_synth_prep, df_synth_model, X_synth_scaled_df, synth_scaler = prepare_model_matrix(
        input_df=df_synthetic_test,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        scale_data=True
    )

    # IF (bolum 34.1)
    _emit(progress_callback, 0.85, "Sentetik test - Isolation Forest...")
    if_synth_result = run_isolation_forest(
        X=X_synth_scaled_df, contamination=CONTAMINATION, random_state=RANDOM_STATE)
    df_synth_prep["IF_SCORE_SYNTH"] = if_synth_result["scores"]
    df_synth_prep["IF_ANOMALY_SYNTH"] = if_synth_result["labels"]

    # LOF (bolum 34.2)
    _emit(progress_callback, 0.88, "Sentetik test - Local Outlier Factor...")
    lof_synth_result = run_lof(X=X_synth_scaled_df, contamination=CONTAMINATION)
    df_synth_prep["LOF_SCORE_SYNTH"] = lof_synth_result["scores"]
    df_synth_prep["LOF_ANOMALY_SYNTH"] = lof_synth_result["labels"]

    # AE (bolum 34.3)
    _emit(progress_callback, 0.90, "Sentetik test - Autoencoder egitiliyor...")
    ae_synth_result = run_autoencoder(
        X=X_synth_scaled_df, contamination=CONTAMINATION, random_state=RANDOM_STATE,
        progress_callback=progress_callback, prog_base=0.90, prog_span=0.08,
        phase="Sentetik test - ")
    df_synth_prep["AE_SCORE_SYNTH"] = ae_synth_result["scores"]
    df_synth_prep["AE_ANOMALY_SYNTH"] = ae_synth_result["labels"]

    _emit(progress_callback, 0.98, "Sentetik test ozetleri olusturuluyor...")

    # Model bazinda tespit (bolum 36)
    synthetic_detection_results = pd.concat([
        synthetic_detection_summary(df_synth_prep, "IF_ANOMALY_SYNTH", "Isolation Forest"),
        synthetic_detection_summary(df_synth_prep, "LOF_ANOMALY_SYNTH", "Local Outlier Factor"),
        synthetic_detection_summary(df_synth_prep, "AE_ANOMALY_SYNTH", "Autoencoder")
    ], ignore_index=True)

    # Senaryo bazinda tespit (bolum 37)
    scenario_detection_results = pd.concat([
        scenario_detection_summary(df_synth_prep, "IF_ANOMALY_SYNTH", "Isolation Forest"),
        scenario_detection_summary(df_synth_prep, "LOF_ANOMALY_SYNTH", "Local Outlier Factor"),
        scenario_detection_summary(df_synth_prep, "AE_ANOMALY_SYNTH", "Autoencoder")
    ], ignore_index=True)

    # Sentetik kayitlarda cakisma (bolum 38)
    synthetic_only = df_synth_prep[df_synth_prep["SENTETIK_FRAUD"] == 1].copy()
    synthetic_only["SYNTH_ANOMALY_MODEL_COUNT"] = (
        synthetic_only["IF_ANOMALY_SYNTH"] +
        synthetic_only["LOF_ANOMALY_SYNTH"] +
        synthetic_only["AE_ANOMALY_SYNTH"]
    )

    def synth_anomaly_combination(row):
        models = []
        if row["IF_ANOMALY_SYNTH"] == 1:
            models.append("IF")
        if row["LOF_ANOMALY_SYNTH"] == 1:
            models.append("LOF")
        if row["AE_ANOMALY_SYNTH"] == 1:
            models.append("AE")
        if len(models) == 0:
            return "Yakalanmadi"
        return " + ".join(models)

    synthetic_only["SYNTH_ANOMALY_COMBINATION"] = synthetic_only.apply(
        synth_anomaly_combination, axis=1)

    synthetic_overlap_summary = (
        synthetic_only["SYNTH_ANOMALY_COMBINATION"]
        .value_counts()
        .rename_axis("MODEL_KOMBINASYONU")
        .reset_index(name="KAYIT_SAYISI")
    )
    synthetic_overlap_summary["ORAN"] = (
        synthetic_overlap_summary["KAYIT_SAYISI"] / len(synthetic_only))
    synthetic_overlap_summary["ORAN_YUZDE"] = synthetic_overlap_summary["ORAN"] * 100

    synth_caught_at_least_one = int((synthetic_only["SYNTH_ANOMALY_MODEL_COUNT"] >= 1).sum())
    synth_caught_at_least_two = int((synthetic_only["SYNTH_ANOMALY_MODEL_COUNT"] >= 2).sum())
    synth_caught_all_three = int((synthetic_only["SYNTH_ANOMALY_MODEL_COUNT"] == 3).sum())
    synth_not_caught = int((synthetic_only["SYNTH_ANOMALY_MODEL_COUNT"] == 0).sum())

    # Genel cakisma (bolum 39)
    synthetic_overlap_general = pd.DataFrame({
        "KATEGORI": [
            "En az bir model tarafindan yakalanan",
            "En az iki model tarafindan yakalanan",
            "Uc model tarafindan yakalanan",
            "Hicbir model tarafindan yakalanmayan"
        ],
        "KAYIT_SAYISI": [
            synth_caught_at_least_one, synth_caught_at_least_two,
            synth_caught_all_three, synth_not_caught
        ]
    })
    synthetic_overlap_general["ORAN"] = (
        synthetic_overlap_general["KAYIT_SAYISI"] / len(synthetic_only))
    synthetic_overlap_general["ORAN_YUZDE"] = synthetic_overlap_general["ORAN"] * 100

    # Skor ozeti (bolum 40)
    synth_score_summary = df_synth_prep[
        ["IF_SCORE_SYNTH", "LOF_SCORE_SYNTH", "AE_SCORE_SYNTH"]].describe().T
    synth_score_summary["SKEWNESS"] = df_synth_prep[
        ["IF_SCORE_SYNTH", "LOF_SCORE_SYNTH", "AE_SCORE_SYNTH"]].skew()
    synth_score_summary["KURTOSIS"] = df_synth_prep[
        ["IF_SCORE_SYNTH", "LOF_SCORE_SYNTH", "AE_SCORE_SYNTH"]].kurtosis()
    synth_score_summary = synth_score_summary.reset_index().rename(columns={"index": "SKOR"})

    # Log donusumlu skorlar (bolum 45)
    df_synth_prep["LOG_LOF_SCORE_SYNTH"] = np.log1p(df_synth_prep["LOF_SCORE_SYNTH"])
    df_synth_prep["LOG_AE_SCORE_SYNTH"] = np.log1p(df_synth_prep["AE_SCORE_SYNTH"])

    # Calisma sureleri (bolum 46)
    synthetic_runtime_summary = pd.DataFrame({
        "MODEL": ["Isolation Forest", "Local Outlier Factor", "Autoencoder"],
        "CALISMA_SURESI_SANIYE": [
            round(if_synth_result["elapsed_time"], 2),
            round(lof_synth_result["elapsed_time"], 2),
            round(ae_synth_result["elapsed_time"], 2)
        ]
    })

    _emit(progress_callback, 1.0, "Sentetik fraud testi tamamlandi.")

    return {
        "df_synth_prep": df_synth_prep,
        "synthetic_only": synthetic_only,
        "if_synth_result": if_synth_result,
        "lof_synth_result": lof_synth_result,
        "ae_synth_result": ae_synth_result,
        "synthetic_detection_results": synthetic_detection_results,
        "scenario_detection_results": scenario_detection_results,
        "synthetic_overlap_summary": synthetic_overlap_summary,
        "synthetic_overlap_general": synthetic_overlap_general,
        "synth_score_summary": synth_score_summary,
        "synthetic_runtime_summary": synthetic_runtime_summary,
    }


# ============================================================
# 12. EXCEL DISA AKTARMA (orijinal bolum 25 ve 47)
# ============================================================

def export_real_results_to_excel(results, output_file):
    """Gercek veri analiz sonuclarini Excel'e yazar (orijinal bolum 25)."""
    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        results["analysis_df"].to_excel(writer, sheet_name="Tum_Kayitlar", index=False)
        results["model_summary"].to_excel(writer, sheet_name="Model_Ozeti", index=False)
        results["score_summary"].to_excel(writer, sheet_name="Skor_Ozeti", index=False)
        results["overlap_summary"].to_excel(writer, sheet_name="Cakisma_Ozeti", index=False)
        results["combination_summary"].to_excel(writer, sheet_name="Kombinasyon_Ozeti", index=False)
        results["jaccard_summary"].to_excel(writer, sheet_name="Jaccard_Ozeti", index=False)
        results["overlap_matrix"].to_excel(writer, sheet_name="Cakisma_Matrisi")
        results["jaccard_matrix"].to_excel(writer, sheet_name="Jaccard_Matrisi")
        results["strong_anomalies"].to_excel(writer, sheet_name="Guclu_Anomali_Adaylari", index=False)
        results["runtime_summary"].to_excel(writer, sheet_name="Calisma_Sureleri", index=False)
    return output_file


def export_synthetic_results_to_excel(synth_results, output_file):
    """Sentetik fraud test sonuclarini Excel'e yazar (orijinal bolum 47)."""
    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        synth_results["df_synth_prep"].to_excel(
            writer, sheet_name="Tum_Kayitlar_Sentetik_Test", index=False)
        synth_results["synthetic_detection_results"].to_excel(
            writer, sheet_name="Model_Bazinda_Tespit", index=False)
        synth_results["scenario_detection_results"].to_excel(
            writer, sheet_name="Senaryo_Bazinda_Tespit", index=False)
        synth_results["synthetic_overlap_summary"].to_excel(
            writer, sheet_name="Sentetik_Cakisma_Detay", index=False)
        synth_results["synthetic_overlap_general"].to_excel(
            writer, sheet_name="Sentetik_Cakisma_Genel", index=False)
        synth_results["synthetic_only"].to_excel(
            writer, sheet_name="Sentetik_Kayitlar", index=False)
        synth_results["synth_score_summary"].to_excel(
            writer, sheet_name="Sentetik_Skor_Ozeti", index=False)
        synth_results["synthetic_runtime_summary"].to_excel(
            writer, sheet_name="Sentetik_Calisma_Sureleri", index=False)
    return output_file
