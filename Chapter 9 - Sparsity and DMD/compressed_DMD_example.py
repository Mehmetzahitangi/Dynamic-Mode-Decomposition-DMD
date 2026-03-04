import numpy as np
import matplotlib.pyplot as plt
import time


# 1. SİSTEMİ OLUŞTURMA (Yüksek Çözünürlüklü Simülasyon)

n = 10000  # Sensör sayısı (Örn: 10.000 piksellik bir kamera)
m = 200    # Zaman adımı (Snapshot sayısı)

# Uçağın kanadındaki 2 temel titreşim frekansı ve büyüme/küçülme oranları
# Dinamik 1: omega = 10 (titreşim), sönümlenme = -0.1
# Dinamik 2: omega = 35 (titreşim), büyüme = 0.2
t = np.linspace(0, 5, m)
x = np.linspace(0, 10, n)
Xgrid, Tgrid = np.meshgrid(x, t)

# veri matrisi (n x m)
X_true = (np.sin(2 * Xgrid) * np.exp((-0.1 + 10j) * Tgrid) + 
          np.cos(5 * Xgrid) * np.exp((0.2 + 35j) * Tgrid)).T

X = X_true[:, :-1]
X_prime = X_true[:, 1:]
dt = t[1] - t[0]


# 2. STANDART DMD
print("--- STANDART DMD BAŞLIYOR ---")
start_time = time.time()

# 10.000 satırlık matrisin SVD'si
U, S_vals, Vh = np.linalg.svd(X, full_matrices=False)
r = 2 # Sistemin 2 temel dinamiği olduğunu biliyoruz
Ur = U[:, :r]
Sr = np.diag(S_vals[:r])
Vr = Vh[:r, :].conj().T

A_tilde = Ur.conj().T @ X_prime @ Vr @ np.linalg.inv(Sr)
Lambda_exact, W_exact = np.linalg.eig(A_tilde)
Omega_exact = np.log(Lambda_exact) / dt

exact_time = time.time() - start_time
print(f"Hesaplama Süresi: {exact_time:.4f} saniye")
print(f"Bulunan Frekanslar (Omega): {np.round(Omega_exact, 2)}")


# 3. COMPRESSED DMD (cDMD) - Sıkıştırılmış Zekâ
print("\n--- COMPRESSED DMD BAŞLIYOR ---")
start_time = time.time()

p = 100  # 10.000 sensör yerine sadece 100 rastgele ölçüm alacağız (%1'i)

# Ölçüm Matrisi C (Gaussian Random Projection)
np.random.seed(42)
C = np.random.randn(p, n)

# SIKIŞTIRMA (Compressing the Data)
Y = C @ X
Y_prime = C @ X_prime

# SADECE 100 satırlık minik Y matrisi üzerinde SVD!
U_y, S_y_vals, Vh_y = np.linalg.svd(Y, full_matrices=False)
Ur_y = U_y[:, :r]
Sr_y = np.diag(S_y_vals[:r])
Vr_y = Vh_y[:r, :].conj().T

# Minik uzaydaki A matrisi
A_tilde_y = Ur_y.conj().T @ Y_prime @ Vr_y @ np.linalg.inv(Sr_y)
Lambda_compressed, W_y = np.linalg.eig(A_tilde_y)
Omega_compressed = np.log(Lambda_compressed) / dt

compressed_time = time.time() - start_time
print(f"Hesaplama Süresi: {compressed_time:.4f} saniye")
print(f"Bulunan Frekanslar (Omega): {np.round(Omega_compressed, 2)}")


# 4. KARŞILAŞTIRMA
print("\n--- 🏆 LABORATUVAR SONUCU ---")
print(f"Hızlanma Oranı: cDMD, Standart DMD'den yaklaşık {exact_time/compressed_time:.1f} kat daha hızlı!")
error = np.linalg.norm(np.sort(Omega_exact) - np.sort(Omega_compressed))
print(f"Gerçek Fizik İle Sıkıştırılmış Fizik Arasındaki Hata: {error:.6f}")