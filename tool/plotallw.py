import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# 1. Ortak Parametreler
algorithms = ["gd", "sgd", "adam"]  # 3 farklı algoritma
nums = ["10", "98", "65", "54", "120"]  # 5 farklı num
directory = "iter500"
N = 500

# 2. Tüm Dosyaların İşlenmesi
data_list = []  # Tüm veriler
labels = []  # Algoritma ve num bilgisi

for algorithm in algorithms:
    for num in nums:
        # Dosyayı yükleme (header=None eklenmiştir)
        file = f"{directory}/{algorithm}_{num}.csv"
        df = pd.read_csv(file, header=None)
        
        # İlk N satırı seç (N, veri satırlarından büyükse tümü alınır)
        df = df.head(N)
        
        # w değerlerini al (4. sütundan itibaren)
        data = df.iloc[:, 4:].values
        data_list.append(data)  # Veriyi kaydet
        
        # Etiketleme
        labels.append(f"{algorithm.upper()}_{num}")

# Verileri birleştirme
data_combined = np.vstack(data_list)

# Veriyi standardize etme
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_combined)

# T-SNE ile 2 boyuta indirme
tsne = TSNE(n_components=2, random_state=42, init='pca')  # PCA ile başlatma
data_tsne = tsne.fit_transform(data_scaled)

# 3. Yörüngelerin Çizimi
plt.figure(figsize=(12, 12))

start_idx = 0
for label, data in zip(labels, data_list):
    end_idx = start_idx + len(data)
    tsne_segment = data_tsne[start_idx:end_idx]
    
    # Çizgiler: Kırmızıdan yeşile geçiş
    for i in range(1, len(tsne_segment)):
        color_ratio = i / len(tsne_segment)  # Oran hesaplama
        gradient_color = (1 - color_ratio, color_ratio, 0)  # Kırmızıdan yeşile
        plt.plot(tsne_segment[i-1:i+1, 0], tsne_segment[i-1:i+1, 1], 
                 color=gradient_color, alpha=0.8)
    
    # Algoritma adı ve numarasını başlangıç renginde yaz
    algorithm_name = label.split("_")[0].lower()
    
    # Başlangıç noktası rengi (algoritmalara göre)
    if algorithm_name == 'adam':
        start_color = 'red'
    elif algorithm_name == 'gd':
        start_color = 'blue'
    elif algorithm_name == 'sgd':
        start_color = 'green'
    
    if algorithm_name == 'gd':
        # numara yaz
        plt.text(tsne_segment[0, 0], tsne_segment[0, 1], label[3:], fontsize=8, ha="right", va="top")

    # Noktalar
    plt.scatter(tsne_segment[:, 0], tsne_segment[:, 1], label=None, alpha=0.6, color=start_color)
    
    start_idx = end_idx

# Algoritmaların renklerini belirtmek için legend ekliyoruz
plt.scatter([], [], color="red", label="Adam")
plt.scatter([], [], color="blue", label="GD")
plt.scatter([], [], color="green", label="SGD")

# Genel başlık ve açıklamalar
plt.title("T-SNE Trajectories for All Algorithms and Initializations", fontsize=16)
plt.xlabel("TSNE Component 1")
plt.ylabel("TSNE Component 2")
plt.figtext(0.5, 0.01, 
            "Trajectories start in red and transition to green as they evolve.\n"
            "Black dots indicate the starting point with algorithm and initialization number labeled.",
            wrap=True, horizontalalignment='center', fontsize=10)

# Kaydetme
plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # Alt açıklama ve başlık için yer bırak
plt.legend(loc="upper left")  # Legend'i sol üst köşeye yerleştir
plt.savefig("tsne.png", bbox_inches='tight')
plt.close()

