import pandas as pd
from sklearn.model_selection import train_test_split

# Dosyaları yükle
mnist_train = pd.read_csv("mnist_train.csv")
mnist_test = pd.read_csv("mnist_test.csv")

# Dosyaları birleştir
mnist_combined = pd.concat([mnist_train, mnist_test], ignore_index=True)

# Veriyi ve etiketi ayır (label sütunu 1. index)
labels = mnist_combined.iloc[:, 0]  # İlk sütun (y)
features = mnist_combined.iloc[:, 1:]  # Geri kalan sütunlar (x)

# Sadece label'ı 0 veya 1 olanları seç
valid_labels = labels[labels.isin([0, 1])]
valid_features = features.loc[valid_labels.index]

# Label 0 olanları -1'e dönüştür
valid_labels = valid_labels.replace(0, -1)

# x değerlerini 0-1 aralığına getir
valid_features = valid_features / 255.0

# Veriyi %80 eğitim, %20 test olarak böl
x_train, x_test, y_train, y_test = train_test_split(valid_features, valid_labels, test_size=0.2, random_state=42)

# Eğitim verilerini kaydet
x_train.to_csv("mnist_train_x.csv", index=False)
y_train.to_csv("mnist_train_y.csv", index=False)

# Test verilerini kaydetmek istersen:
x_test.to_csv("mnist_test_x.csv", index=False)
y_test.to_csv("mnist_test_y.csv", index=False)

print("Dosyalar başarıyla oluşturuldu!")

