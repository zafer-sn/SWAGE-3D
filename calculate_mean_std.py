import os
import numpy as np
from PIL import Image
import argparse

def calculate_depth_stats(input_dir):
    """
    Verilen dizindeki tüm '_depth.png' dosyalarını tarayarak
    derinlik kanalı için ortalama (mean) ve standart sapma (std) hesaplar.
    """
    depth_values = []
    
    print(f"Veri seti taranıyor: {input_dir}")
    
    # Tüm alt klasörleri gez
    files_found = 0
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith("_depth.png"):
                file_path = os.path.join(root, file)
                
                try:
                    # Resmi aç ve grayscale'e çevir
                    img = Image.open(file_path).convert('L')
                    
                    # Numpy array'e çevir ve [0, 1] aralığına normalize et
                    img_np = np.array(img) / 255.0
                    
                    # Değerleri listeye ekle (düzleştirerek)
                    depth_values.append(img_np.flatten())
                    files_found += 1
                except Exception as e:
                    print(f"Dosya okunurken hata: {file_path} - {e}")

    if files_found == 0:
        print("Hata: Hiçbir derinlik haritası (_depth.png) bulunamadı!")
        return

    print(f"Toplam {files_found} derinlik haritası işleniyor...")
    
    # Bellek yönetimi için parça parça hesaplama yapılabilir, 
    # ancak ShapeNet alt kümeleri için concatenation genellikle sorun olmaz.
    try:
        all_pixels = np.concatenate(depth_values)
        
        # İstatistikleri hesapla
        mean = np.mean(all_pixels)
        std = np.std(all_pixels)
        
        print("\n" + "="*40)
        print("HESAPLANAN SONUÇLAR")
        print("="*40)
        print(f"Dataset Yolu: {input_dir}")
        print(f"İşlenen Dosya Sayısı: {files_found}")
        print("-" * 40)
        print(f"Mean (Ortalama): {mean:.4f}")
        print(f"Std (Standart Sapma): {std:.4f}")
        print("="*40)
        print("NOT: Bu değerleri utils.py dosyasındaki 'ShapeNetPlusImageDataset' sınıfında")
        print("self.depth_mean ve self.depth_std değişkenlerine atamalısınız.")
        
    except MemoryError:
        print("Bellek hatası: Veri seti çok büyük. Batch tabanlı hesaplama gerekebilir.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='input', 
                        help='Veri setinin bulunduğu ana klasör (örn: input/car/train)')
    args = parser.parse_args()
    
    if os.path.exists(args.input_dir):
        calculate_depth_stats(args.input_dir)
    else:
        print(f"Klasör bulunamadı: {args.input_dir}")
        print("Lütfen geçerli bir veri yolu belirtin.")
