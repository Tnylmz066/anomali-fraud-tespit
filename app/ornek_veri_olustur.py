# -*- coding: utf-8 -*-
"""
ornek_veri_olustur.py
=====================
Gercek veri seti henuz olmadigindan, orijinal not defterindeki 37 sutunlu
semaya uygun ORNEK (sahte) bir Excel dosyasi uretir. Bu dosya yalnizca
uygulamayi uctan uca test etmek icindir; gercek veri yerine gecmez.

Sutun adlari, analysis_core icindeki beklenen normallestirilmis adlarla
birebir ayni verilir; boylece sutun uyum kontrolu sorunsuz gecer.
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analysis_core as ac  # noqa: E402


def generate_sample(n_rows=12000, seed=42):
    rng = np.random.default_rng(seed)

    data = {}

    # Kimlik sutunlari
    data["POLICE_NO"] = rng.integers(100000, 180000, n_rows)      # tekrarli
    data["PROVIZYON_NO"] = rng.integers(1, 5 * n_rows, n_rows)

    # Surekli / sayisal degiskenler
    data["YENILEME_NO"] = rng.integers(0, 11, n_rows)
    data["POL_BAS_TAR"] = rng.integers(40000, 45500, n_rows)      # tarih serisi
    data["POL_BAS_TAN_FRK"] = rng.integers(0, 366, n_rows)
    data["SIGORTALI_YASI"] = np.clip(rng.normal(42, 16, n_rows), 0, 95).round(0)
    data["POL_BAS_HASAR_SURE"] = np.clip(rng.normal(120, 70, n_rows), 0, 366).round(1)
    data["SKYT_HASAR_SURE"] = np.clip(rng.normal(90, 60, n_rows), 0, 366).round(1)
    data["HASAR_IHBAR_SURE"] = np.clip(rng.exponential(8, n_rows), 0, 90).round(1)
    data["POL_BAS_ODEME_SURE"] = np.clip(rng.normal(150, 80, n_rows), 0, 450).round(1)

    # Tutarlar (saga carpik dagilim)
    talep = np.exp(rng.normal(6.2, 1.0, n_rows)) + 50.0
    data["TALEP_TUTAR"] = talep.round(2)
    odeme_orani = np.clip(rng.normal(0.82, 0.18, n_rows), 0.0, 1.0)
    data["ODENEN_TUTAR"] = (talep * odeme_orani).round(2)

    # Kategorik (sayisal kodlanmis) degiskenler - cesitli kardinalitelerde
    cat_card = {
        "POLICE_TIPI": 3,
        "ACENTE_NO": 200,
        "ACENTE_BOLGE": 12,
        "CINSIYET": 2,
        "BRY_TIP": 5,
        "PAKET_ID": 30,
        "HASAR_STATU": 6,
        "HASAR_KAYNAGI": 4,
        "ICMAL_STATUSU": 5,
        "ANA_TEMINAT_ADI": 15,
        "ALT_TEMINAT": 60,
        "ISLEM_TIPI": 4,
        "DOKTOR_BRANS": 40,
        "DOKTOR_ID": 2000,
        "ICD_KODU": 1500,
        "KURUM_TIPI": 5,
        "KURUM_KODU": 800,
        "KURUM_IL": 81,
        "KURUM_ILCE": 300,
        "BANKA_ADI": 25,
        "SUBE_ADI": 500,
        "IBAN_NO": 6000,
        "ANA_TEMINAT_KODU": 15,
        "TEMINAT_KODU": 200,
        "VAKA_TIPI": 6,
    }
    for col, card in cat_card.items():
        data[col] = rng.integers(1, card + 1, n_rows)

    # Sutun sirasini orijinal semaya gore diz
    ordered_cols = ac.id_columns + ac.numeric_features + ac.categorical_features
    df = pd.DataFrame(data)[ordered_cols]

    # Gercekci olmasi icin bir miktar eksik deger ekle
    for col in ["SKYT_HASAR_SURE", "POL_BAS_ODEME_SURE", "ICD_KODU", "DOKTOR_ID"]:
        mask = rng.random(n_rows) < 0.03
        df.loc[mask, col] = np.nan

    return df


def main():
    out_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_path = os.path.join(out_dir, "ornek_veri.xlsx")
    n_rows = 12000
    if len(sys.argv) > 1:
        try:
            n_rows = int(sys.argv[1])
        except ValueError:
            pass

    print("Ornek veri uretiliyor: %d satir..." % n_rows)
    df = generate_sample(n_rows=n_rows)
    df.to_excel(out_path, index=False)
    print("Olusturuldu:", out_path)
    print("Boyut:", df.shape)


if __name__ == "__main__":
    main()
