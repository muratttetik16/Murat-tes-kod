# YOLOv8x Model Eğitim Kayıp Grafikleri

Bu proje, YOLO model eğitim çıktısı olan `results.csv` dosyasından YOLOv8 standart formatında kayıp (loss) grafiklerini ve confusion matrix görselleştirmelerini oluşturur.

## Dosyalar

- `results.csv`: YOLO eğitim sonuçlarını içeren CSV dosyası
- `plot_loss.py`: YOLOv8 formatında kayıp grafiklerini oluşturan Python scripti
- `plot_confusion_matrix.py`: YOLOv8 formatında confusion matrix oluşturan Python scripti
- `results.png`: Ultralytics YOLOv8 standart formatında 2x5 grid kayıp görselleştirmesi
- `confusion_matrix.png`: YOLOv8 formatında confusion matrix görselleştirmesi

## Gereksinimler

```bash
pip install pandas matplotlib numpy seaborn
```

Veya:

```bash
pip install -r requirements.txt
```

## Kullanım

### Kayıp Grafikleri Oluşturma

```bash
python3 plot_loss.py
```

Script çalıştırıldığında:
1. `results.csv` dosyasını okur
2. Training ve validation kayıp değerlerini ve metrikleri çıkarır
3. YOLOv8 standart formatında `results.png` dosyası oluşturur
4. Terminal'de eğitim istatistiklerini gösterir

### Confusion Matrix Oluşturma

```bash
python3 plot_confusion_matrix.py
```

Script çalıştırıldığında:
1. `results.csv` dosyasından precision/recall değerlerini okur
2. YOLOv8 standart formatında `confusion_matrix.png` dosyası oluşturur
3. Terminal'de confusion matrix istatistiklerini gösterir

## Grafik Formatları

### results.png
YOLOv8 orijinal formatında 2x5 grid (10 alt grafik) içerir:

**Üst Satır (Training Metrikleri):**
- train/box_loss - Eğitim kutu kaybı
- train/cls_loss - Eğitim sınıflandırma kaybı
- train/dfl_loss - Eğitim DFL kaybı
- metrics/precision(B) - Hassasiyet metriği
- metrics/recall(B) - Geri çağırma metriği

**Alt Satır (Validation Metrikleri):**
- val/box_loss - Doğrulama kutu kaybı
- val/cls_loss - Doğrulama sınıflandırma kaybı
- val/dfl_loss - Doğrulama DFL kaybı
- metrics/mAP50(B) - mAP@0.5 metriği
- metrics/mAP50-95(B) - mAP@0.5:0.95 metriği

### confusion_matrix.png
YOLOv8 standart confusion matrix formatında:
- Satırlar: Gerçek sınıflar (True)
- Sütunlar: Tahmin edilen sınıflar (Predicted)
- Köşegen değerler: Doğru tahminler (True Positives)
- Normalize edilmiş değerler (0-1 arası)
- Renk skalası: Mavi tonları (düşük-yüksek)

## Özellikler

- ✅ YOLOv8 Ultralytics orijinal format ile %100 uyumlu
- ✅ 2x5 grid layout (tam olarak YOLOv8 results.png formatı)
- ✅ Confusion Matrix görselleştirmesi
- ✅ NaN değerlerini otomatik olarak yönetir
- ✅ Yüksek çözünürlüklü (300 DPI) grafikler üretir
- ✅ Terminal'de detaylı eğitim istatistikleri gösterir
- ✅ Tüm YOLO kayıp türlerini ve metrikleri içerir
- ✅ Türkçe etiketler ve başlıklar
