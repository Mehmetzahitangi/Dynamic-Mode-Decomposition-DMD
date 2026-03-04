"""
Ajanımız artık sadece sensörlerin ham verisine $x_1 ve x_2) bakmayacak. Bir Gözlem Sözlüğü (Observable Dictionary) oluşturacağız. 
Ajan, ortamı daha yüksek boyutlu bir uzaya (Y) taşıyacak (Lifting).
Bu sistemde sorunun kaynağı: x_1^2 kısmıydı. Sözlüğümüzü :y = [ x_1  x_2  x_1^2 ] şeklinde yapalım
"""



import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# 1. DOĞRUSAL OLMAYAN (NONLINEAR) SİSTEMİ YARATMA
# Sistem: 
# x1(k+1) = 0.9 * x1(k)
# x2(k+1) = 0.5 * x2(k) + x1(k)^2   <-- İşleri bozan kaotik terim!
# =============================================================================
np.random.seed(42)

m = 200  # 200 zaman adımı (Snapshot)
X_true = np.zeros((2, m))

# Başlangıç durumu
X_true[:, 0] = [2.0, -1.0]

# Sistemi simüle etme / Veri toplama
for k in range(m - 1):
    X_true[0, k+1] = 0.9 * X_true[0, k]
    X_true[1, k+1] = 0.5 * X_true[1, k] + (X_true[0, k]**2)


#X1 = X_true[:, :-1]  # Geçmiş (Y)
#X2 = X_true[:, 1:]   # Gelecek (Y')


# =============================================================================
# ADIM 2: EXTENDED DMD (EDMD) - KOOPMAN GÖZLEMLERİ
# =============================================================================
# 1. Gözlem Sözlüğünü/Matrislerini (Observable Dictionary) Oluşturma
# Yeni uzayımız 3 boyutlu olacak: [x1, x2, x1^2]
Y_true = np.zeros((3, m))
Y_true[0, :] = X_true[0, :]       # x1
Y_true[1, :] = X_true[1, :]       # x2
Y_true[2, :] = X_true[0, :]**2    # x1'in karesi

Y1 = Y_true[:, :-1] # Geçmiş Gözlemler
Y2 = Y_true[:, 1:]  # Gelecek Gözlemler

# 2. EDMD (Koopman Matrisi - K) Hesaplama
# Artık X yerine, bu Y matrisi üzerinde çalışıyoruz
K_edmd = Y2 @ np.linalg.pinv(Y1) # Koopman Matrisi K = Y2@Y1(pseudo inverse hali)

# 3. Geleceği Tahmin Etme (EDMD ile)
Y_edmd_pred = np.zeros((3, m))
Y_edmd_pred[:, 0] = Y_true[:, 0]

for k in range(m - 1):
    Y_edmd_pred[:, k+1] = K_edmd @ Y_edmd_pred[:, k]

# 4. EDMD GÖRSELLEŞTİRME 
plt.figure(figsize=(10, 5))
plt.plot(X_true[1, :], 'b-', linewidth=4, alpha=0.5, label='Gerçek x2 (Doğrusal Olmayan)')
plt.plot(Y_edmd_pred[1, :], 'r--', linewidth=2, label='EDMD x2 Tahmini')
plt.title("Extended DMD: Koopman Gözlemleri ile Çözüm", fontsize=14, fontweight='bold')
plt.xlabel("Zaman Adımı")
plt.legend()
plt.grid(True)
plt.show()

print("--- EDMD KOOPMAN MATRİSİ (K) ---")
print(np.round(K_edmd, 3))
print("\nMatris : 2. satır, 3. sütundaki değer tam olarak '1.0' çıkmalı.")
print("Yani ajan 'x2'nin geleceği, x1^2'ye bağlıdır' kuralını kendisi bulmuş olur.")