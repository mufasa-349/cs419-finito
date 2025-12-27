import cv2
import numpy as np
from pathlib import Path

# Görüntüyü yükle
input_path = "hela_dataset-train-01/t009_processed.tif"
output_path = "hela_dataset-train-01/t009_processed.tif"

# TIF dosyasını oku (binary görüntü)
binary = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

if binary is None:
    print(f"Hata: {input_path} dosyası okunamadı!")
    exit(1)

print(f"Binary görüntü boyutu: {binary.shape}")
print(f"Binary görüntü tipi: {binary.dtype}")

# Morphological işlemler için kernel oluştur
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
# Alternatif kernel seçenekleri:
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
# kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))

# 1. Closing işlemi: Dilation + Erosion
# Küçük delikleri kapatır ve nesneleri birleştirir
closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
print(f"Closing işlemi tamamlandı")
print(f"Closing öncesi - Beyaz: {np.sum(binary == 255)}, Siyah: {np.sum(binary == 0)}")
print(f"Closing sonrası - Beyaz: {np.sum(closed == 255)}, Siyah: {np.sum(closed == 0)}")

# 2. Opening işlemi: Erosion + Dilation
# Küçük gürültüleri ve ince çıkıntıları temizler
opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
print(f"Opening işlemi tamamlandı")
print(f"Opening sonrası - Beyaz: {np.sum(opened == 255)}, Siyah: {np.sum(opened == 0)}")

# Sonucu kaydet
cv2.imwrite(output_path, opened)
print(f"Closing + Opening uygulanmış görüntü kaydedildi: {output_path}")

