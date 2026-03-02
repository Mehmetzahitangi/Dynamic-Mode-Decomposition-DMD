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

# =============================================================================
# ALGORİTMA 7.7: HMM İLE DMD'Yİ KARŞILAŞTIRMAK
# =============================================================================

# 1. Gözlemleri One-Hot Vektörlere Çevirme
# Gözlem uzayımız 3 boyutlu olduğu için veriyi 3xM boyutlu bir matrise çeviriyoruz.
Y_data = np.zeros((3, m))
for k in range(m):
    Y_data[observations[k], k] = 1.0

# 2. Gecikme Koordinatları (Hankel Matrisi) Oluşturma
# DMD'nin rastgeleliği çözebilmesi için geçmişi yığması (Frame Stacking) şarttır!
delay_shifts = 5 # Son 5 günü üst üste yığalım
H_data = np.zeros((3 * delay_shifts, m - delay_shifts + 1))

for i in range(delay_shifts):
    H_data[i*3:(i+1)*3, :] = Y_data[:, i:m - delay_shifts + i + 1]

# 3. Hankel Matrislerini Geçmiş ve Gelecek Olarak Ayırma
H1 = H_data[:, :-1]
H2 = H_data[:, 1:]

# 4. DMD Algoritmasını Çalıştırma
U, S_vals, Vh = np.linalg.svd(H1, full_matrices=False)
r = 2 # Hava durumunun 2 gizli durumu olduğunu tahmin ediyoruz (Latent Space = 2)
Ur = U[:, :r]
Sr = np.diag(S_vals[:r])
Vr = Vh[:r, :].conj().T

# A_tilde = Ur^* * H2 * Vr * Sr^-1
A_tilde_hmm = Ur.conj().T @ H2 @ Vr @ np.linalg.inv(Sr)

# 5. Özdeğer Karşılaştırması (Spektral Analiz)
lambda_dmd, W_dmd = np.linalg.eig(A_tilde_hmm)
lambda_true, W_true = np.linalg.eig(A_hmm)

print("\n--- ÖZDEĞER (SPEKTRAL) KARŞILAŞTIRMA ---")
print("Gerçek HMM (A) Özdeğerleri :", np.sort(np.round(lambda_true, 4)))
print("Hankel DMD Özdeğerleri     :", np.sort(np.round(lambda_dmd, 4)))

# Not: Olasılıksal Markov matrislerinin en büyük (dominant) özdeğeri HER ZAMAN 1'dir.
# Bu 1 değeri, sistemin "Kararlı Olasılık Dağılımını" (Steady-State Probability) temsil eder.