# -*- coding: utf-8 -*-
"""
main.py
=======
Anomali ve Fraud Tespit Sistemi - masaustu arayuzu (PyQt5).

Bu arayuz, analysis_core icindeki (orijinal Colab kodu ile birebir ayni)
analiz hattini calistirir ve sonuclari tablolar + grafikler halinde gosterir.
Colab'a ozgu files.upload / files.download / display cagrilarinin yerini
sirasiyla dosya secici, kaydetme penceresi ve gomulu tablolar/grafikler alir.
"""

import os
import sys
import traceback

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import pandas as pd

from PyQt5.QtCore import Qt, QThread, pyqtSignal, QAbstractTableModel, QModelIndex
from PyQt5.QtGui import QFont, QColor, QIcon
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QFileDialog, QProgressBar, QTabWidget, QTableView,
    QHeaderView, QFrame, QCheckBox, QScrollArea, QMessageBox, QSizePolicy,
    QStackedWidget
)

# Yerel moduller
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analysis_core as ac
import figures as fg

from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)

APP_TITLE = "Anomali ve Fraud Tespit Sistemi"
APP_SUBTITLE = "Isolation Forest  -  Local Outlier Factor  -  Autoencoder"

# Renk paleti
COL_BG = "#eef1f7"
COL_PRIMARY = "#4f46e5"
COL_PRIMARY_DARK = "#4338ca"
COL_IF = "#2563eb"
COL_LOF = "#0891b2"
COL_AE = "#7c3aed"
COL_TOTAL = "#475569"


# ============================================================
# Pandas DataFrame -> Qt tablo modeli
# ============================================================

class DataFrameModel(QAbstractTableModel):
    def __init__(self, df=pd.DataFrame(), parent=None):
        super().__init__(parent)
        self._df = df.copy()

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self._df)

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else self._df.shape[1]

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        value = self._df.iat[index.row(), index.column()]
        if role == Qt.DisplayRole:
            return self._format(value)
        if role == Qt.TextAlignmentRole:
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return int(Qt.AlignRight | Qt.AlignVCenter)
            return int(Qt.AlignLeft | Qt.AlignVCenter)
        return None

    @staticmethod
    def _format(value):
        if value is None:
            return ""
        try:
            if pd.isna(value):
                return ""
        except (ValueError, TypeError):
            pass
        if isinstance(value, float):
            if value == int(value) and abs(value) < 1e15:
                return str(int(value))
            return "{:,.4f}".format(value)
        return str(value)

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role != Qt.DisplayRole:
            return None
        if orientation == Qt.Horizontal:
            return str(self._df.columns[section])
        return str(self._df.index[section])


# ============================================================
# Arka plan isi thread'i
# ============================================================

class AnalysisWorker(QThread):
    progress = pyqtSignal(float, str)
    finished_ok = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, df_raw, run_synthetic=True, parent=None):
        super().__init__(parent)
        self.df_raw = df_raw
        self.run_synthetic = run_synthetic

    def run(self):
        try:
            def cb(pct, msg):
                self.progress.emit(float(pct), str(msg))

            real = ac.run_full_analysis(self.df_raw, progress_callback=cb)

            synth = None
            if self.run_synthetic:
                synth = ac.run_synthetic_test(real["df_numeric"], progress_callback=cb)
            else:
                self.progress.emit(1.0, "Analiz tamamlandi.")

            self.finished_ok.emit({"real": real, "synth": synth})
        except Exception:
            self.failed.emit(traceback.format_exc())


# ============================================================
# Ana pencere
# ============================================================

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.resize(1240, 840)

        self.selected_file = None
        self.results = None
        self._figs = []  # figure referanslarini canli tut

        self._build_ui()
        self._apply_style()

    # -------------------- UI kurulumu --------------------

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Header
        header = QFrame()
        header.setObjectName("Header")
        header.setFixedHeight(86)
        hl = QVBoxLayout(header)
        hl.setContentsMargins(28, 14, 28, 14)
        hl.setSpacing(2)
        title = QLabel(APP_TITLE)
        title.setObjectName("HeaderTitle")
        subtitle = QLabel(APP_SUBTITLE)
        subtitle.setObjectName("HeaderSubtitle")
        hl.addWidget(title)
        hl.addWidget(subtitle)
        root.addWidget(header)

        # Govde
        body = QWidget()
        body.setObjectName("Body")
        bl = QVBoxLayout(body)
        bl.setContentsMargins(22, 18, 22, 18)
        bl.setSpacing(14)
        root.addWidget(body, 1)

        # Kontrol karti
        control = QFrame()
        control.setObjectName("Card")
        cg = QGridLayout(control)
        cg.setContentsMargins(18, 16, 18, 16)
        cg.setHorizontalSpacing(14)
        cg.setVerticalSpacing(10)

        self.btn_select = QPushButton("  Excel Dosyasi Sec")
        self.btn_select.setObjectName("Secondary")
        self.btn_select.setCursor(Qt.PointingHandCursor)
        self.btn_select.clicked.connect(self.on_select_file)

        self.lbl_file = QLabel("Henuz dosya secilmedi.")
        self.lbl_file.setObjectName("FileLabel")
        self.lbl_file.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        self.chk_synth = QCheckBox("Sentetik fraud testini de calistir")
        self.chk_synth.setChecked(True)

        self.btn_run = QPushButton("Analizi Baslat")
        self.btn_run.setObjectName("Primary")
        self.btn_run.setCursor(Qt.PointingHandCursor)
        self.btn_run.setEnabled(False)
        self.btn_run.clicked.connect(self.on_run)

        cg.addWidget(self.btn_select, 0, 0)
        cg.addWidget(self.lbl_file, 0, 1)
        cg.addWidget(self.chk_synth, 0, 2)
        cg.addWidget(self.btn_run, 0, 3)
        cg.setColumnStretch(1, 1)
        bl.addWidget(control)

        # Ilerleme
        prog_row = QHBoxLayout()
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(True)
        self.progress.setFormat("%p%")
        self.lbl_status = QLabel("Hazir.")
        self.lbl_status.setObjectName("StatusLabel")
        prog_row.addWidget(self.progress, 3)
        prog_row.addWidget(self.lbl_status, 2)
        bl.addLayout(prog_row)

        # Icerik: baslangic ekrani + sonuc sekmeleri (stacked)
        self.stack = QStackedWidget()
        bl.addWidget(self.stack, 1)

        # Bos durum
        self.empty = QFrame()
        self.empty.setObjectName("Card")
        el = QVBoxLayout(self.empty)
        el.setAlignment(Qt.AlignCenter)
        empty_icon = QLabel("📊")
        empty_icon.setAlignment(Qt.AlignCenter)
        empty_icon.setStyleSheet("font-size: 54px;")
        empty_text = QLabel(
            "Baslamak icin bir Excel dosyasi secip\n\"Analizi Baslat\" dugmesine basin.")
        empty_text.setAlignment(Qt.AlignCenter)
        empty_text.setObjectName("EmptyText")
        el.addWidget(empty_icon)
        el.addWidget(empty_text)
        self.stack.addWidget(self.empty)

        # Sekmeler
        self.tabs = QTabWidget()
        self.tabs.setObjectName("MainTabs")
        self.stack.addWidget(self.tabs)
        self.stack.setCurrentWidget(self.empty)

        # Alt bar: disa aktarma
        export_row = QHBoxLayout()
        export_row.addStretch(1)
        self.btn_export_real = QPushButton("Gercek Veri Sonuclari (Excel)")
        self.btn_export_real.setObjectName("Export")
        self.btn_export_real.setCursor(Qt.PointingHandCursor)
        self.btn_export_real.setEnabled(False)
        self.btn_export_real.clicked.connect(self.on_export_real)
        self.btn_export_synth = QPushButton("Sentetik Test Sonuclari (Excel)")
        self.btn_export_synth.setObjectName("Export")
        self.btn_export_synth.setCursor(Qt.PointingHandCursor)
        self.btn_export_synth.setEnabled(False)
        self.btn_export_synth.clicked.connect(self.on_export_synth)
        export_row.addWidget(self.btn_export_real)
        export_row.addWidget(self.btn_export_synth)
        bl.addLayout(export_row)

    def _apply_style(self):
        self.setStyleSheet("""
        QWidget { font-family: 'Segoe UI', Arial; font-size: 13px; color: #1f2433; }
        #Body { background: #eef1f7; }
        #Header {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 #4338ca, stop:1 #6d28d9);
        }
        #HeaderTitle { color: white; font-size: 21px; font-weight: 700; }
        #HeaderSubtitle { color: #d7d4f7; font-size: 12px; letter-spacing: 1px; }
        #Card {
            background: white; border: 1px solid #e2e6ef; border-radius: 12px;
        }
        #FileLabel { color: #5a6172; padding: 4px 8px; }
        #EmptyText { color: #7a8295; font-size: 15px; }
        #StatusLabel { color: #5a6172; }
        QPushButton#Primary {
            background: #4f46e5; color: white; border: none; border-radius: 9px;
            padding: 10px 22px; font-weight: 700; font-size: 14px;
        }
        QPushButton#Primary:hover { background: #4338ca; }
        QPushButton#Primary:disabled { background: #b9b7e8; }
        QPushButton#Secondary {
            background: #eef0fb; color: #4338ca; border: 1px solid #d7d9f5;
            border-radius: 9px; padding: 10px 18px; font-weight: 600;
        }
        QPushButton#Secondary:hover { background: #e3e6fa; }
        QPushButton#Export {
            background: #ffffff; color: #15803d; border: 1px solid #86efac;
            border-radius: 9px; padding: 9px 16px; font-weight: 600;
        }
        QPushButton#Export:hover { background: #f0fdf4; }
        QPushButton#Export:disabled { color: #b6c2bb; border-color: #e2e8e6; }
        QProgressBar {
            border: none; border-radius: 8px; background: #e2e6ef; height: 18px;
            text-align: center; color: #3a3f51; font-weight: 600;
        }
        QProgressBar::chunk {
            border-radius: 8px;
            background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #6366f1, stop:1 #8b5cf6);
        }
        QCheckBox { color: #3a3f51; }
        QTabWidget::pane { border: 1px solid #e2e6ef; border-radius: 10px; background: white; top: -1px; }
        QTabBar::tab {
            background: #e9edf6; color: #5a6172; padding: 9px 16px; margin-right: 3px;
            border-top-left-radius: 8px; border-top-right-radius: 8px; font-weight: 600;
        }
        QTabBar::tab:selected { background: white; color: #4338ca; }
        QTabBar::tab:hover { color: #4338ca; }
        QTableView {
            background: white; gridline-color: #eef1f7; border: 1px solid #e6e9f1;
            border-radius: 8px; selection-background-color: #e0e7ff; selection-color: #1f2433;
        }
        QTableView::item { padding: 4px 6px; }
        QHeaderView::section {
            background: #f3f5fb; color: #3a3f51; padding: 7px 8px; border: none;
            border-right: 1px solid #e6e9f1; border-bottom: 2px solid #d7dcec; font-weight: 700;
        }
        QScrollArea { border: none; background: transparent; }
        """)

    # -------------------- Olaylar --------------------

    def on_select_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Excel dosyasi sec", os.path.expanduser("~"),
            "Excel Dosyalari (*.xlsx *.xls);;Tum Dosyalar (*.*)")
        if path:
            self.selected_file = path
            self.lbl_file.setText(path)
            self.btn_run.setEnabled(True)
            self.lbl_status.setText("Dosya secildi. Analizi baslatabilirsiniz.")

    def on_run(self):
        if not self.selected_file:
            return
        try:
            self.lbl_status.setText("Excel okunuyor...")
            QApplication.processEvents()
            df_raw = pd.read_excel(self.selected_file)
        except Exception as exc:
            QMessageBox.critical(self, "Hata", "Excel okunamadi:\n%s" % exc)
            self.lbl_status.setText("Hazir.")
            return

        self._set_running(True)
        self.worker = AnalysisWorker(df_raw, run_synthetic=self.chk_synth.isChecked())
        self.worker.progress.connect(self.on_progress)
        self.worker.finished_ok.connect(self.on_finished)
        self.worker.failed.connect(self.on_failed)
        self.worker.start()

    def on_progress(self, pct, msg):
        self.progress.setValue(int(pct * 100))
        self.lbl_status.setText(msg)

    def on_failed(self, tb):
        self._set_running(False)
        self.lbl_status.setText("Hata olustu.")
        QMessageBox.critical(self, "Analiz Hatasi",
                             "Analiz sirasinda bir hata olustu:\n\n" + tb[-1500:])

    def on_finished(self, payload):
        self.results = payload
        try:
            self._populate(payload)
        except Exception:
            self.on_failed(traceback.format_exc())
            return
        self._set_running(False)
        self.progress.setValue(100)
        self.lbl_status.setText("Tamamlandi. Sonuclar hazir.")
        self.stack.setCurrentWidget(self.tabs)
        self.btn_export_real.setEnabled(True)
        self.btn_export_synth.setEnabled(payload["synth"] is not None)

    def _set_running(self, running):
        self.btn_run.setEnabled(not running and self.selected_file is not None)
        self.btn_select.setEnabled(not running)
        self.chk_synth.setEnabled(not running)
        if running:
            self.btn_run.setText("Calisiyor...")
            self.progress.setValue(0)
        else:
            self.btn_run.setText("Analizi Baslat")

    # -------------------- Sonuclari yerlestir --------------------

    def _populate(self, payload):
        real = payload["real"]
        synth = payload["synth"]

        self.tabs.clear()
        self._figs = []

        self.tabs.addTab(self._tab_overview(real), "Genel Bakis")
        self.tabs.addTab(self._tab_scores(real), "Skor Dagilimlari")
        self.tabs.addTab(self._tab_overlap(real), "Model Cakismasi")
        self.tabs.addTab(self._tab_top_records(real), "En Anomalik Kayitlar")
        self.tabs.addTab(self._tab_strong(real), "Guclu Anomaliler")
        if synth is not None:
            self.tabs.addTab(self._tab_synth(synth), "Sentetik Fraud Testi")

    # ---- yardimci widget ureticileri ----

    def _make_table(self, df, max_height=None):
        view = QTableView()
        view.setModel(DataFrameModel(df))
        view.setAlternatingRowColors(False)
        view.verticalHeader().setVisible(False)
        view.setEditTriggers(QTableView.NoEditTriggers)
        view.setSelectionBehavior(QTableView.SelectRows)
        view.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        view.horizontalHeader().setStretchLastSection(True)
        view.resizeColumnsToContents()
        view.setWordWrap(False)
        if max_height:
            view.setMaximumHeight(max_height)
        return view

    def _scroll(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        vbox = QVBoxLayout(container)
        vbox.setContentsMargins(14, 14, 14, 14)
        vbox.setSpacing(14)
        scroll.setWidget(container)
        return scroll, vbox

    def _section_label(self, text):
        lbl = QLabel(text)
        lbl.setStyleSheet("font-size: 15px; font-weight: 700; color: #2d3344; margin-top: 4px;")
        return lbl

    def _add_chart(self, layout, fig):
        self._figs.append(fig)
        card = QFrame()
        card.setObjectName("Card")
        cl = QVBoxLayout(card)
        cl.setContentsMargins(8, 8, 8, 8)
        canvas = FigureCanvas(fig)
        canvas.setMinimumHeight(360)
        toolbar = NavigationToolbar(canvas, card)
        cl.addWidget(toolbar)
        cl.addWidget(canvas)
        layout.addWidget(card)

    def _metric_card(self, title, value, color):
        card = QFrame()
        card.setObjectName("Card")
        card.setMinimumHeight(96)
        v = QVBoxLayout(card)
        v.setContentsMargins(16, 12, 16, 12)
        val = QLabel(str(value))
        val.setStyleSheet("font-size: 28px; font-weight: 800; color: %s;" % color)
        cap = QLabel(title)
        cap.setStyleSheet("color: #7a8295; font-size: 12px; font-weight: 600;")
        bar = QFrame()
        bar.setFixedHeight(4)
        bar.setStyleSheet("background: %s; border-radius: 2px;" % color)
        v.addWidget(val)
        v.addWidget(cap)
        v.addStretch(1)
        v.addWidget(bar)
        return card

    # ---- sekmeler ----

    def _tab_overview(self, real):
        scroll, vbox = self._scroll()
        adf = real["analysis_df"]

        cards = QHBoxLayout()
        cards.setSpacing(14)
        cards.addWidget(self._metric_card("Toplam Kayit", "{:,}".format(len(adf)), COL_TOTAL))
        cards.addWidget(self._metric_card(
            "Isolation Forest Anomali", int(adf["IF_ANOMALY"].sum()), COL_IF))
        cards.addWidget(self._metric_card(
            "Local Outlier Factor Anomali", int(adf["LOF_ANOMALY"].sum()), COL_LOF))
        cards.addWidget(self._metric_card(
            "Autoencoder Anomali", int(adf["AE_ANOMALY"].sum()), COL_AE))
        vbox.addLayout(cards)

        # Veri / sutun bilgisi
        shape = real["shape"]
        exp = real["expected_shape"]
        info = "Veri boyutu: %d satir x %d sutun  (beklenen: %d x %d)" % (
            shape[0], shape[1], exp[0], exp[1])
        miss_def = real["missing_defined_columns"]
        extra = real["extra_columns"]
        if not miss_def and not extra:
            info += "   |   Tum sutunlar beklenen semayla uyumlu."
        else:
            info += "   |   Eksik: %s   Fazla: %s" % (miss_def, extra)
        info_lbl = QLabel(info)
        info_lbl.setStyleSheet("color: #5a6172; padding: 4px 2px;")
        vbox.addWidget(info_lbl)

        vbox.addWidget(self._section_label("Model Bazinda Anomali Ozeti"))
        vbox.addWidget(self._make_table(real["model_summary"], max_height=160))

        vbox.addWidget(self._section_label("Anomali Skoru Ozet Istatistikleri"))
        vbox.addWidget(self._make_table(real["score_summary"], max_height=170))

        vbox.addWidget(self._section_label("Model Calisma Sureleri (saniye)"))
        vbox.addWidget(self._make_table(real["runtime_summary"], max_height=160))

        vbox.addStretch(1)
        return scroll

    def _tab_scores(self, real):
        scroll, vbox = self._scroll()
        if_r, lof_r, ae_r = real["if_result"], real["lof_result"], real["ae_result"]
        adf = real["analysis_df"]

        vbox.addWidget(self._section_label("Isolation Forest"))
        self._add_chart(vbox, fg.fig_score_distribution(
            adf["IF_SCORE"].values, if_r["threshold"],
            "Isolation Forest Anomali Skoru Dagilimi", "Isolation Forest Anomali Skoru"))

        vbox.addWidget(self._section_label("Local Outlier Factor"))
        self._add_chart(vbox, fg.fig_score_distribution(
            adf["LOF_SCORE"].values, lof_r["threshold"],
            "Local Outlier Factor Anomali Skoru Dagilimi", "Local Outlier Factor Anomali Skoru"))
        self._add_chart(vbox, fg.fig_log_score_distribution(
            adf["LOF_SCORE"].values, lof_r["threshold"],
            "LOF Log Donusumlu Skor Dagilimi", "log(1 + LOF Anomali Skoru)"))
        self._add_chart(vbox, fg.fig_trimmed_score_distribution(
            adf["LOF_SCORE"].values, lof_r["threshold"],
            "LOF Skor Dagilimi - Ust %0.5 Kirpilmis", "Local Outlier Factor Anomali Skoru"))

        vbox.addWidget(self._section_label("Autoencoder"))
        self._add_chart(vbox, fg.fig_score_distribution(
            adf["AE_SCORE"].values, ae_r["threshold"],
            "Autoencoder Rekonstruksiyon Hatasi Dagilimi", "Autoencoder Rekonstruksiyon Hatasi"))
        self._add_chart(vbox, fg.fig_log_score_distribution(
            adf["AE_SCORE"].values, ae_r["threshold"],
            "Autoencoder Log Donusumlu Hata Dagilimi", "log(1 + Rekonstruksiyon Hatasi)"))
        self._add_chart(vbox, fg.fig_trimmed_score_distribution(
            adf["AE_SCORE"].values, ae_r["threshold"],
            "Autoencoder Hata Dagilimi - Ust %0.5 Kirpilmis", "Autoencoder Rekonstruksiyon Hatasi"))
        self._add_chart(vbox, fg.fig_ae_training_loss(ae_r["history"]))

        return scroll

    def _tab_overlap(self, real):
        scroll, vbox = self._scroll()

        row = QHBoxLayout()
        row.setSpacing(14)
        c1 = QFrame(); c1.setObjectName("Card"); l1 = QVBoxLayout(c1); l1.setContentsMargins(8, 8, 8, 8)
        f1 = fg.fig_overlap_heatmap(real["overlap_matrix"]); self._figs.append(f1)
        cv1 = FigureCanvas(f1); cv1.setMinimumHeight(320); l1.addWidget(cv1)
        c2 = QFrame(); c2.setObjectName("Card"); l2 = QVBoxLayout(c2); l2.setContentsMargins(8, 8, 8, 8)
        f2 = fg.fig_jaccard_heatmap(real["jaccard_matrix"]); self._figs.append(f2)
        cv2 = FigureCanvas(f2); cv2.setMinimumHeight(320); l2.addWidget(cv2)
        row.addWidget(c1); row.addWidget(c2)
        vbox.addLayout(row)

        vbox.addWidget(self._section_label("Genel Cakisma Ozeti"))
        vbox.addWidget(self._make_table(real["overlap_summary"], max_height=320))

        vbox.addWidget(self._section_label("Anomali Kombinasyonlari"))
        vbox.addWidget(self._make_table(real["combination_summary"], max_height=240))

        vbox.addWidget(self._section_label("Jaccard Benzerligi (Model Ciftleri)"))
        vbox.addWidget(self._make_table(real["jaccard_summary"], max_height=140))

        vbox.addStretch(1)
        return scroll

    def _tab_top_records(self, real):
        sub = QTabWidget()
        sub.setObjectName("MainTabs")
        for name, key in [("Isolation Forest", "top_if"),
                          ("Local Outlier Factor", "top_lof"),
                          ("Autoencoder", "top_ae")]:
            w = QWidget()
            lo = QVBoxLayout(w)
            lo.setContentsMargins(12, 12, 12, 12)
            lo.addWidget(self._section_label("%s - En Anomalik Ilk 20 Kayit" % name))
            lo.addWidget(self._make_table(real[key]))
            sub.addTab(w, name)
        return sub

    def _tab_strong(self, real):
        scroll, vbox = self._scroll()
        strong = real["strong_anomalies"]
        cols = [c for c in [
            "POLICE_NO", "PROVIZYON_NO", "TALEP_TUTAR", "ODENEN_TUTAR",
            "POL_BAS_HASAR_SURE", "SKYT_HASAR_SURE", "HASAR_IHBAR_SURE",
            "POL_BAS_ODEME_SURE", "DOKTOR_ID", "ICD_KODU", "KURUM_KODU",
            "IF_SCORE", "LOF_SCORE", "AE_SCORE",
            "IF_ANOMALY", "LOF_ANOMALY", "AE_ANOMALY",
            "ANOMALY_MODEL_COUNT", "ANOMALY_COMBINATION"
        ] if c in strong.columns]
        vbox.addWidget(self._section_label(
            "En az iki model tarafindan yakalanan kayit sayisi: %d" % len(strong)))
        show = strong[cols].head(50) if len(strong) else strong[cols]
        vbox.addWidget(self._make_table(show))
        vbox.addStretch(1)
        return scroll

    def _tab_synth(self, synth):
        scroll, vbox = self._scroll()

        vbox.addWidget(self._section_label("Model Bazinda Sentetik Fraud Tespiti"))
        vbox.addWidget(self._make_table(synth["synthetic_detection_results"], max_height=160))
        self._add_chart(vbox, fg.fig_model_detection_rate(synth["synthetic_detection_results"]))

        vbox.addWidget(self._section_label("Senaryo Bazinda Tespit"))
        vbox.addWidget(self._make_table(synth["scenario_detection_results"], max_height=320))
        self._add_chart(vbox, fg.fig_scenario_detection_rate(synth["scenario_detection_results"]))

        vbox.addWidget(self._section_label("Model Yakalama Kombinasyonlari"))
        vbox.addWidget(self._make_table(synth["synthetic_overlap_general"], max_height=200))
        vbox.addWidget(self._make_table(synth["synthetic_overlap_summary"], max_height=200))
        self._add_chart(vbox, fg.fig_synthetic_overlap(synth["synthetic_overlap_summary"]))

        vbox.addWidget(self._section_label("Orijinal ve Sentetik Skor Karsilastirmasi"))
        self._add_chart(vbox, fg.fig_synth_vs_original_kde(
            synth["df_synth_prep"], "IF_SCORE_SYNTH", "Isolation Forest Sentetik Test Skoru"))
        self._add_chart(vbox, fg.fig_synth_vs_original_kde(
            synth["df_synth_prep"], "LOG_LOF_SCORE_SYNTH", "log(1 + LOF Skoru)"))
        self._add_chart(vbox, fg.fig_synth_vs_original_kde(
            synth["df_synth_prep"], "LOG_AE_SCORE_SYNTH",
            "log(1 + Autoencoder Rekonstruksiyon Hatasi)"))

        return scroll

    # -------------------- Disa aktarma --------------------

    def on_export_real(self):
        if not self.results:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Gercek veri sonuclarini kaydet",
            os.path.join(os.path.expanduser("~"), "anomali_model_sonuclari_gercek_veri.xlsx"),
            "Excel Dosyasi (*.xlsx)")
        if not path:
            return
        try:
            ac.export_real_results_to_excel(self.results["real"], path)
            QMessageBox.information(self, "Kaydedildi", "Sonuclar kaydedildi:\n%s" % path)
        except Exception as exc:
            QMessageBox.critical(self, "Hata", "Kaydedilemedi:\n%s" % exc)

    def on_export_synth(self):
        if not self.results or self.results["synth"] is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Sentetik test sonuclarini kaydet",
            os.path.join(os.path.expanduser("~"), "sentetik_fraud_test_sonuclari.xlsx"),
            "Excel Dosyasi (*.xlsx)")
        if not path:
            return
        try:
            ac.export_synthetic_results_to_excel(self.results["synth"], path)
            QMessageBox.information(self, "Kaydedildi", "Sonuclar kaydedildi:\n%s" % path)
        except Exception as exc:
            QMessageBox.critical(self, "Hata", "Kaydedilemedi:\n%s" % exc)


def main():
    try:
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    except Exception:
        pass

    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 10))

    icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app_icon.ico")
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))

    win = MainWindow()
    if os.path.exists(icon_path):
        win.setWindowIcon(QIcon(icon_path))
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
