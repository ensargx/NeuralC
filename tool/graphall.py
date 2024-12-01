import os
import pandas as pd
import matplotlib.pyplot as plt

# 1. Ortak Parametreler
algorithms = ["gd", "sgd", "adam"]  # 3 farklı algoritma
nums = ["10", "98", "65", "54", "120"]  # 5 farklı num
directory = "iter500"  # Dosyaların bulunduğu dizin
output_dir = "output_graphs"  # Çıktı grafikleri için dizin
N = 500

# 2. Çıktı Klasörü Kontrolü
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 3. Verilerin İşlenmesi ve Grafiklerin Çizilmesi
for num in nums:
    # Grafik için figür ve eksenleri hazırlıyoruz
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    
    # Algoritmaların karşılaştırılması
    for algorithm in algorithms:
        file_path = f"{directory}/{algorithm}_{num}.csv"  # Dosya yolu
        if not os.path.exists(file_path):
            print(f"Dosya bulunamadı: {file_path}")
            continue
        
        # Veriyi yükle
        df = pd.read_csv(file_path, header=None)

        # İlk N satırı seç (N, veri satırlarından büyükse tümü alınır)
        df = df.head(N)
        
        # Epoch, cost, süre, accuracy sütunlarını seç
        epochs = df[0]
        costs = df[1]
        durations = df[2]
        accuracies = df[3]
        
        # Her algoritmayı farklı bir renkle çizelim
        label = f"{algorithm.upper()}"
        
        # Süre vs Cost
        axs[0, 0].plot(durations, costs, label=label, alpha=0.7)
        axs[0, 0].set_title("Süre vs Cost")
        axs[0, 0].set_xlabel("Süre(ms)")
        axs[0, 0].set_ylabel("Cost")
        
        # Epoch vs Cost
        axs[0, 1].plot(epochs, costs, label=label, alpha=0.7)
        axs[0, 1].set_title("Epoch vs Cost")
        axs[0, 1].set_xlabel("Epoch")
        axs[0, 1].set_ylabel("Cost")
        
        # Süre vs Accuracy
        axs[1, 0].plot(durations, accuracies, label=label, alpha=0.7)
        axs[1, 0].set_title("Süre vs Accuracy")
        axs[1, 0].set_xlabel("Süre(ms)")
        axs[1, 0].set_ylabel("Accuracy")
        
        # Epoch vs Accuracy
        axs[1, 1].plot(epochs, accuracies, label=label, alpha=0.7)
        axs[1, 1].set_title("Epoch vs Accuracy")
        axs[1, 1].set_xlabel("Epoch")
        axs[1, 1].set_ylabel("Accuracy")
    
    # Tüm grafikler için birleştirilmiş etiketler ve başlık
    for ax in axs.flat:
        ax.legend()  # Algoritma etiketleri
        ax.grid(alpha=0.3)
    
    fig.suptitle(f"{num} No.lu Karşılaştırma", fontsize=16)
    
    # Grafikleri kaydet
    output_path = f"{output_dir}/comparison_{num}.png"
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Başlık için boşluk bırak
    plt.savefig(output_path)
    plt.close(fig)  # Bellek kullanımı için grafiği kapat
    
    print(f"Grafikler kaydedildi: {output_path}")

