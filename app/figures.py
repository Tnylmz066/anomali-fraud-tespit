# -*- coding: utf-8 -*-
"""
figures.py
==========
Orijinal not defterindeki grafiklerin (bolum 13, 21, 22, 41-45) PyQt
arayuzune gomulebilen Figure nesneleri olarak yeniden uretimi.

Grafik icerigi orijinal ile aynidir (ayni veriler, ayni bin sayisi,
ayni baslik ve eksen etiketleri, ayni esik cizgileri). Yalnizca
pyplot.show() yerine Figure dondurulur.
"""

import numpy as np
import matplotlib
from matplotlib.figure import Figure
import seaborn as sns

# Orijinal kodun grafik ayarlari (bolum 1)
try:
    matplotlib.style.use("seaborn-v0_8-whitegrid")
except Exception:
    pass
sns.set_palette("Set2")


def _new_fig(w=7.6, h=4.6):
    fig = Figure(figsize=(w, h), dpi=100)
    fig.patch.set_facecolor("white")
    return fig


# ---------------- Skor dagilim grafikleri (bolum 13) ----------------

def fig_score_distribution(scores, threshold, title, xlabel):
    fig = _new_fig()
    ax = fig.add_subplot(111)
    sns.histplot(scores, bins=80, kde=True, ax=ax)
    ax.axvline(threshold, color="red", linestyle="--", linewidth=2,
               label="Yuzde 1 Esik Degeri")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Kayit Sayisi")
    ax.legend()
    fig.tight_layout()
    return fig


def fig_log_score_distribution(scores, threshold, title, xlabel):
    log_scores = np.log1p(scores)
    log_threshold = np.log1p(threshold)
    fig = _new_fig()
    ax = fig.add_subplot(111)
    sns.histplot(log_scores, bins=80, kde=True, ax=ax)
    ax.axvline(log_threshold, color="red", linestyle="--", linewidth=2,
               label="Yuzde 1 Esik Degeri")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Kayit Sayisi")
    ax.legend()
    fig.tight_layout()
    return fig


def fig_trimmed_score_distribution(scores, threshold, title, xlabel,
                                   upper_quantile=0.995):
    upper_limit = np.quantile(scores, upper_quantile)
    trimmed_scores = scores[scores <= upper_limit]
    fig = _new_fig()
    ax = fig.add_subplot(111)
    sns.histplot(trimmed_scores, bins=80, kde=True, ax=ax)
    if threshold <= upper_limit:
        ax.axvline(threshold, color="red", linestyle="--", linewidth=2,
                   label="Yuzde 1 Esik Degeri")
    else:
        ax.axvline(upper_limit, color="orange", linestyle="--", linewidth=2,
                   label="Grafik Ust Siniri")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Kayit Sayisi")
    ax.legend()
    fig.tight_layout()
    return fig


def fig_ae_training_loss(history):
    """Autoencoder egitim/dogrulama kaybi (bolum 13.4)."""
    fig = _new_fig()
    ax = fig.add_subplot(111)
    ax.plot(history.history["loss"], label="Egitim Kaybi")
    ax.plot(history.history["val_loss"], label="Dogrulama Kaybi")
    ax.set_title("Autoencoder Egitim ve Dogrulama Kaybi")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.legend()
    fig.tight_layout()
    return fig


# ---------------- Matris isi haritalari (bolum 21, 22) ----------------

def fig_overlap_heatmap(overlap_matrix):
    fig = _new_fig(6.6, 4.8)
    ax = fig.add_subplot(111)
    sns.heatmap(overlap_matrix.astype(int), annot=True, fmt="d",
                cmap="YlGnBu", cbar=True, ax=ax)
    ax.set_title("Modeller Arasi Anomali Cakisma Matrisi")
    ax.set_xlabel("Model")
    ax.set_ylabel("Model")
    fig.tight_layout()
    return fig


def fig_jaccard_heatmap(jaccard_matrix):
    fig = _new_fig(6.6, 4.8)
    ax = fig.add_subplot(111)
    sns.heatmap(jaccard_matrix.astype(float), annot=True, fmt=".3f",
                cmap="YlOrRd", vmin=0, vmax=1, cbar=True, ax=ax)
    ax.set_title("Modeller Arasi Jaccard Benzerligi")
    ax.set_xlabel("Model")
    ax.set_ylabel("Model")
    fig.tight_layout()
    return fig


# ---------------- Sentetik fraud grafikleri (bolum 41-45) ----------------

def fig_model_detection_rate(synthetic_detection_results):
    fig = _new_fig(8.0, 4.8)
    ax = fig.add_subplot(111)
    sns.barplot(data=synthetic_detection_results, x="MODEL",
                y="SENTETIK_FRAUD_TESPIT_ORANI_YUZDE", ax=ax)
    ax.set_title("Model Bazinda Sentetik Fraud Tespit Orani")
    ax.set_xlabel("Model")
    ax.set_ylabel("Tespit Orani (%)")
    ax.set_ylim(0, 100)
    for i, row in synthetic_detection_results.reset_index(drop=True).iterrows():
        ax.text(i, row["SENTETIK_FRAUD_TESPIT_ORANI_YUZDE"] + 1,
                "%.2f%%" % row["SENTETIK_FRAUD_TESPIT_ORANI_YUZDE"], ha="center")
    fig.tight_layout()
    return fig


def fig_scenario_detection_rate(scenario_detection_results):
    fig = _new_fig(9.0, 5.0)
    ax = fig.add_subplot(111)
    sns.barplot(data=scenario_detection_results, x="SENTETIK_SENARYO",
                y="SENTETIK_FRAUD_TESPIT_ORANI_YUZDE", hue="MODEL", ax=ax)
    ax.set_title("Senaryo Bazinda Sentetik Fraud Tespit Orani")
    ax.set_xlabel("Sentetik Fraud Senaryosu")
    ax.set_ylabel("Tespit Orani (%)")
    ax.set_ylim(0, 100)
    for label in ax.get_xticklabels():
        label.set_rotation(20)
        label.set_ha("right")
    ax.legend(title="Model")
    fig.tight_layout()
    return fig


def fig_synthetic_overlap(synthetic_overlap_summary):
    fig = _new_fig(8.4, 5.0)
    ax = fig.add_subplot(111)
    sns.barplot(data=synthetic_overlap_summary, x="MODEL_KOMBINASYONU",
                y="ORAN_YUZDE", ax=ax)
    ax.set_title("Sentetik Fraud Kayitlarinda Model Yakalama Kombinasyonlari")
    ax.set_xlabel("Model Kombinasyonu")
    ax.set_ylabel("Oran (%)")
    for label in ax.get_xticklabels():
        label.set_rotation(30)
        label.set_ha("right")
    for i, row in synthetic_overlap_summary.reset_index(drop=True).iterrows():
        ax.text(i, row["ORAN_YUZDE"] + 0.5, "%.2f%%" % row["ORAN_YUZDE"], ha="center")
    fig.tight_layout()
    return fig


def fig_synth_vs_original_kde(df_synth_prep, score_col, title):
    """Orijinal ve sentetik kayitlarin skor yogunlugu (bolum 44/45)."""
    fig = _new_fig(8.0, 4.6)
    ax = fig.add_subplot(111)
    sns.kdeplot(data=df_synth_prep[df_synth_prep["SENTETIK_FRAUD"] == 0],
                x=score_col, label="Orijinal Kayitlar", fill=True, alpha=0.4, ax=ax)
    sns.kdeplot(data=df_synth_prep[df_synth_prep["SENTETIK_FRAUD"] == 1],
                x=score_col, label="Sentetik Fraud Kayitlari", fill=True, alpha=0.4, ax=ax)
    ax.set_title(title + " - Orijinal ve Sentetik Karsilastirmasi")
    ax.set_xlabel(title)
    ax.set_ylabel("Yogunluk")
    ax.legend()
    fig.tight_layout()
    return fig
