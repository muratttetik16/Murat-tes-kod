# YOLOv8x Model Eğitim Kayıp Grafikleri

Bu proje, YOLO model eğitim çıktısı olan `results.csv` dosyasından YOLOv8 standart formatında kayıp (loss) grafiklerini oluşturur.

## Dosyalar

- `results.csv`: YOLO eğitim sonuçlarını içeren CSV dosyası
- `plot_loss.py`: YOLOv8 formatında kayıp grafiklerini oluşturan Python scripti
- `results.png`: Ultralytics YOLOv8 standart formatında 2x5 grid kayıp görselleştirmesi

## Gereksinimler

```bash
pip install pandas matplotlib numpy
```

Veya:

```bash
pip install -r requirements.txt
```

## Kullanım

Grafikleri oluşturmak için:

```bash
python3 plot_loss.py
```

Script çalıştırıldığında:
1. `results.csv` dosyasını okur
2. Training ve validation kayıp değerlerini ve metrikleri çıkarır
3. YOLOv8 standart formatında `results.png` dosyası oluşturur
4. Terminal'de eğitim istatistiklerini gösterir

## Grafik Formatı

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

## Özellikler

- ✅ YOLOv8 Ultralytics orijinal format ile %100 uyumlu
- ✅ 2x5 grid layout (tam olarak YOLOv8 results.png formatı)
- ✅ NaN değerlerini otomatik olarak yönetir
- ✅ Yüksek çözünürlüklü (300 DPI) grafikler üretir
- ✅ Terminal'de detaylı eğitim istatistikleri gösterir
- ✅ Tüm YOLO kayıp türlerini ve metrikleri içerir
