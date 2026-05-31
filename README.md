# Anomali ve Fraud Tespit Sistemi

Sigorta hasar verilerinde **anomali / fraud (sahtekarlik) tespiti** yapan masaüstü uygulaması.
Üç bağımsız makine öğrenmesi modelini birlikte çalıştırır ve sonuçları tablolar + grafiklerle sunar:

- **Isolation Forest**
- **Local Outlier Factor (LOF)**
- **Autoencoder** (TensorFlow / Keras)

Ayrıca veriye yapay (sentetik) fraud kayıtları enjekte ederek modellerin **tespit oranını** ölçen bir sentetik test modülü içerir.

---

## 🚀 Kurulum (Windows) — Tek Tıkla

> Bilgisayarınıza elle Python kurmanıza veya başka bir ayar yapmanıza **gerek yoktur.**

1. Bu depoyu indirin: yeşil **`Code`** düğmesi → **`Download ZIP`** → indirilen ZIP'i bir klasöre çıkarın.
   (Veya `git clone` ile klonlayın.)
2. Çıkardığınız klasördeki **`KUR.bat`** dosyasına **çift tıklayın.**
   Kurulum otomatik olarak şunları yapar:
   - Uygun Python (3.9–3.11) yoksa **winget ile otomatik kurar**,
   - İzole bir ortam (`.venv`) oluşturur ve tüm paketleri (**TensorFlow dâhil**) yükler,
   - Masaüstüne **"Anomali ve Fraud Tespit"** kısayolunu **otomatik oluşturur.**
   - İlk kurulum, TensorFlow indirileceği için **birkaç dakika** sürebilir.
3. Kurulum bitince masaüstündeki **kısayoldan** uygulamayı başlatın.

> Not: İlk açılışta Windows Defender SmartScreen uyarısı çıkarsa "Daha fazla bilgi → Yine de çalıştır" diyebilirsiniz (kod açık kaynaktır).

---

## 🖥️ Kullanım

1. **Excel Dosyası Seç** → analiz edilecek `.xlsx` dosyasını seçin.
2. **Analizi Başlat** → üç model sırayla çalışır; ilerleme çubuğu ve durum metni canlı güncellenir (pencere donmaz).
3. Sekmeler:
   - **Genel Bakış** — anomali sayıları, model özeti, skor istatistikleri, çalışma süreleri
   - **Skor Dağılımları** — her modelin skor histogramları + %1 eşik çizgisi + Autoencoder eğitim kaybı
   - **Model Çakışması** — çakışma ve Jaccard ısı haritaları, kombinasyon tabloları
   - **En Anomalik Kayıtlar** — her model için ilk 20 kayıt
   - **Güçlü Anomaliler** — en az iki model tarafından yakalanan kayıtlar
   - **Sentetik Fraud Testi** — model/senaryo bazında tespit oranları ve grafikleri
4. Alttaki yeşil düğmelerle sonuçları **Excel'e** aktarın (orijinal not defteriyle aynı sayfa yapısında).

İsterseniz **"Sentetik fraud testini de çalıştır"** kutusunun işaretini kaldırarak yalnızca gerçek veri analizini (daha hızlı) çalıştırabilirsiniz.

---

## 📋 Beklenen Veri

37 sütunlu sigorta hasar tablosu (`.xlsx`). Sütun adları otomatik **normalleştirilir** (Türkçe karakterler, boşluklar vb. düzeltilir), bu yüzden başlıkların birebir aynı yazılması gerekmez. Beklenen sütun listesi `app/analysis_core.py` içindedir.

Gerçek veriniz yokken test için örnek veri üretebilirsiniz:

```
.venv\Scripts\python.exe app\ornek_veri_olustur.py
```

---

## 🧩 Proje Yapısı

```
.
├─ KUR.bat                  # Tek tıkla kurulum (bunu çalıştırın)
├─ kurulum.ps1             # Kurulum mantığı (Python/venv/paket/kısayol)
├─ requirements.txt        # Python bağımlılıkları
├─ Baslat_konsol.bat       # Uygulamayı konsolla başlatır (sorun gidermek için)
└─ app/
   ├─ main.py              # PyQt5 masaüstü arayüzü
   ├─ analysis_core.py     # Analiz mantığı (orijinal not defteriyle birebir)
   ├─ figures.py           # Grafik üretimi
   ├─ ornek_veri_olustur.py# Örnek/test verisi üretici
   ├─ generate_icon.py     # Uygulama ikonu üretici
   └─ app_icon.ico         # Uygulama ikonu
```

---

## 🛠️ Geliştirici Notları

- Uygulamayı elle çalıştırma: `.venv\Scripts\pythonw.exe app\main.py`
- Bağımlılıkları yeniden yükleme: `.venv\Scripts\python.exe -m pip install -r requirements.txt`
- Analiz mantığı (modeller, parametreler `RANDOM_STATE=42`, `CONTAMINATION=0.01`, sentetik senaryolar) orijinal Google Colab not defterinden **değiştirilmeden** taşınmıştır. Yalnızca Colab'a özgü `files.upload` / `files.download` / `display` çağrıları masaüstü karşılıklarıyla değiştirilmiştir.

---

## ⚙️ Gereksinimler

- Windows 10 / 11
- ~2 GB boş disk (TensorFlow dâhil)
- Python 3.9–3.11 (yoksa `KUR.bat` winget ile otomatik kurar)
