# YOLO Loss Graph Generator

Bu proje, YOLO model eğitim çıktısı olan `results.csv` dosyasından kayıp (loss) grafiklerini oluşturur.

## Dosyalar

- `results.csv`: YOLO eğitim sonuçlarını içeren CSV dosyası
- `plot_loss.py`: Kayıp grafiklerini oluşturan Python scripti
- `loss_graph.png`: 6 ayrı grafik içeren kayıp görselleştirmesi (train/val box, cls, dfl losses)
- `loss_graph_combined.png`: Tüm kayıp eğrilerini tek grafikte gösteren görselleştirme

## Gereksinimler

```bash
pip install pandas matplotlib numpy
```

## Kullanım

Grafikleri oluşturmak için:

```bash
python3 plot_loss.py
```

Script çalıştırıldığında:
1. `results.csv` dosyasını okur
2. Training ve validation kayıp değerlerini çıkarır
3. İki farklı grafik oluşturur:
   - `loss_graph.png`: 6 ayrı subplot içeren detaylı grafik
   - `loss_graph_combined.png`: Tüm kayıp eğrilerini tek grafikte gösteren özet grafik
4. Terminal'de son epoch kayıp değerlerini gösterir

## Grafikler

### loss_graph.png
YOLO eğitim sürecindeki 6 farklı kayıp türünü ayrı grafiklerde gösterir:
- Train Box Loss (Eğitim Kutu Kaybı)
- Train Classification Loss (Eğitim Sınıflandırma Kaybı)
- Train DFL Loss (Eğitim DFL Kaybı)
- Validation Box Loss (Doğrulama Kutu Kaybı)
- Validation Classification Loss (Doğrulama Sınıflandırma Kaybı)
- Validation DFL Loss (Doğrulama DFL Kaybı)

### loss_graph_combined.png
Tüm eğitim ve doğrulama kayıplarını tek bir grafikte birleştirir, karşılaştırma yapmayı kolaylaştırır.

## Özellikler

- NaN değerlerini otomatik olarak yönetir
- Yüksek çözünürlüklü (300 DPI) grafikler üretir
- YOLO'nun orijinal çıktısına benzer stil ve format
- Terminal'de kayıp istatistikleri gösterir
