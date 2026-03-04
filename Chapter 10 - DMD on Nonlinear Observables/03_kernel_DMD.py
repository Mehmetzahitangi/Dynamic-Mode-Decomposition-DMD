

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
# ADIM 2: KERNEL DMD (ÇEKİRDEK HİLESİ)
# =============================================================================
# Özellik matrisini oluşturmamıza gerek yok (sadece X1 ve X2'yi kullan)
# Polinom kernel ile iç çarpım matrislerini oluşturuyoruz
# Polinom Kernel: k(x, y) = (1 + x^T * y)^2

# Gram Matrisleri (Sadece m x m boyutunda, yani 199 x 199) (Replay Bufferdaki her bir anın, diğer anlarla ne kadar benzediğini(iç çarpım) gösterir)
G = (1 + X1.T @ X1)**2      # Geçmişin kendisiyle benzerliği, G => gram matrisi geçmişin kendisiyle çarpımı. (BU SAYEDE SİMETRİK OLUR)
A_hat = (1 + X1.T @ X2)**2  # Geçmişin gelecekle benzerliği, G matrisine benzer, t anındaki verilerin t+1 anındaki verilerle benzerliğini/korelasyonunu ölçer. Zaman içindeki değişim bu matrisin içinde gizlidir.

# G matrisinin SVD'si (Yüksek boyutlu bir uzaya çıkmadan hesaplama yapıyoruz)
eigvals, V = np.linalg.eigh(G) # simetrik/hermitian eigenvalue decomposition. Daha hızlı, hafıza kullanımı daha iyi

# Çok küçük özdeğerleri (gürültüyü) at
valid = eigvals > 1e-10
Sigma2 = eigvals[valid] # Koopman modlarının "Enerjileri"/varyansını tutar
V = V[:, valid] # V ise temel özellikleri tutar

Sigma = np.diag(np.sqrt(Sigma2))
Sigma_inv = np.diag(1.0 / np.sqrt(Sigma2))

# Kernel uzayındaki Koopman Matrisi (r x r boyutunda)
K_kernel = (Sigma @ V.T) @ A_hat @ (V @ Sigma_inv) # kernel uzayındaki Koopman Matrisi

print("--- KERNEL DMD ÖZDEĞERLERİ ---")
# K_kernel'in özdeğerleri, EDMD'nin (K_edmd) özdeğerleriyle eşleşecek
lambda_kernel = np.linalg.eigvals(K_kernel)
print(np.sort(np.round(lambda_kernel, 3)))
print("\nKernel DMD, büyük boyutlu özellikleri hesaplamadan sadece iç çarpımlarla sistemin fiziksel durumu çözülür.")