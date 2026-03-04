"""
x_1'in geleceği sadece kendisine bağlıdır (Doğrusal: x_1 => lambda*x_1)
x_2'nin geleceği ise x_1'in karesine bağlıdır (Doğrusal Olmayan: x_1^2)
Önce bu veriyi üreteceğiz, sonra ajan (Standart DMD) bu veriye bakıp fiziği çözmeye çalışacak.
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


X1 = X_true[:, :-1]  # Geçmiş (Y)
X2 = X_true[:, 1:]   # Gelecek (Y')


# =============================================================================
# 2. STANDART DMD UYGULAMASI
# =============================================================================
# Standart DMD sadece x2 = A * x1 denklemini çözmeye çalışır.
A_std = X2 @ np.linalg.pinv(X1) # pseudo inverse

# Geleceği yanlış A matrisiyle tahmin edelim
X_std_pred = np.zeros((2, m))
X_std_pred[:, 0] = X_true[:, 0]

for k in range(m - 1):
    X_std_pred[:, k+1] = A_std @ X_std_pred[:, k]

# =============================================================================
# 3. GÖRSELLEŞTİRME
# =============================================================================
plt.figure(figsize=(10, 5))

# x1 dinamiği (Doğrusal olduğu için ikisi de başarır)
plt.plot(X_true[0, :], 'k-', linewidth=3, label='Gerçek x1')
plt.plot(X_std_pred[0, :], 'r--', linewidth=2, label='Standart DMD x1')

# x2 dinamiği (Doğrusal olmayan karesel terim burada)
plt.plot(X_true[1, :], 'b-', linewidth=3, label='Gerçek x2')
plt.plot(X_std_pred[1, :], 'g--', linewidth=2, label='Standart DMD x2 (ÇÖKÜŞ)')

plt.title("Standart DMD Doğrusal Olan ve Olmayan Sistemde Başarısı", fontsize=14, fontweight='bold')
plt.xlabel("Zaman Adımı")
plt.legend()
plt.grid(True)
plt.show()

# Bulunan Fiziksel Özellikler
print("--- STANDART DMD FİZİK MOTORU (A) ---")
print(np.round(A_std, 3))
print("Gerçekte olması gereken: x1^2 terimini bulmalıydı ama bulamadı")