import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# ALGORİTMA 7.6: HİDDEN MARKOV SİSTEMİNİ SİMÜLE ETMEK
# =============================================================================

# 1. Gerçek Olasılık Matrisleri (Ajanın bilmediği gizli dünya)
# Durumlar: [Güneşli, Yağmurlu]
# A matrisi: Sütunların toplamı 1 olmalıdır (Markov özelliği)
A_hmm = np.array([[0.8, 0.4],  # Güneşliden->Güneşliye (%80), Yağmurludan->Güneşliye (%40)
                  [0.2, 0.6]]) # Güneşliden->Yağmurluya (%20), Yağmurludan->Yağmurluya (%60)

# Gözlemler: [Yürüyüş, Alışveriş, Temizlik]
# B matrisi: Durum (sütun) verildiğinde gözlem (satır) olasılıkları
B_hmm = np.array([[0.6, 0.1],  # Güneşliyse Yürüyüş (%60), Yağmurluysa Yürüyüş (%10)
                  [0.3, 0.4],  # Güneşliyse Alışveriş (%30), Yağmurluysa Alışveriş (%40)
                  [0.1, 0.5]]) # Güneşliyse Temizlik (%10), Yağmurluysa Temizlik (%50)

# 2. Simülasyon Değişkenleri
m = 1000 # 1000 günlük veri toplayacağız
hidden_states = np.zeros(m, dtype=int)
observations = np.zeros(m, dtype=int)

# 3. Rastgele Başlangıç Durumu
current_state = 0 # 0: Güneşli ile başlayalım
hidden_states[0] = current_state

# Gözlem seçimi (Mevcut duruma göre B matrisinden)
observations[0] = np.random.choice([0, 1, 2], p=B_hmm[:, current_state])

# 4. Zaman İçinde Simülasyon Döngüsü
np.random.seed(42)
for k in range(1, m):
    # Bir sonraki GİZLİ durumu seç (A matrisine göre)
    next_state = np.random.choice([0, 1], p=A_hmm[:, current_state])
    hidden_states[k] = next_state
    
    # O durumun GÖZLEMİNİ seç (B matrisine göre)
    observations[k] = np.random.choice([0, 1, 2], p=B_hmm[:, next_state])
    
    current_state = next_state

print("İlk 10 Günlük Gizli Hava Durumu :", hidden_states[:10])
print("İlk 10 Günlük Davranış Gözlemi :", observations[:10])