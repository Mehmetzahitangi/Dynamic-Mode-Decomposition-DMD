import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA, PCA

# 1. PARAMETRELER VE VERİ OLUŞTURMA
xi = np.linspace(-10, 10, 400)
t = np.linspace(0, 4*np.pi, 200)
Xgrid, Tgrid = np.meshgrid(xi, t)

# İki farklı frekansta titreyen iki sinyal
f1 = (1 / np.cosh(Xgrid + 3)) * np.exp(2.3j * Tgrid)
f2 = (1 / np.cosh(Xgrid)) * np.tanh(Xgrid) * np.exp(2.8j * Tgrid)
X_total = (f1 + f2).T  

# Scikit-learn reel sayılarla çalışır
X_real = np.real(X_total)

# Gerçek (Ground Truth) Uzaysal Modlar
true_mode1 = 1 / np.cosh(xi + 3)
true_mode2 = (1 / np.cosh(xi)) * np.tanh(xi)

# 2. MAKİNE ÖĞRENMESİ MODELLERİ
# PCA
pca = PCA(n_components=2)
spatial_modes_pca = pca.fit_transform(X_real) 

# ICA
ica = FastICA(n_components=2, random_state=42)
spatial_modes_ica = ica.fit_transform(X_real)

# 3. GÖRSELLEŞTİRME (MATPLOTLIB)
fig, axs = plt.subplots(3, 1, figsize=(10, 10))

# A. Gerçek Modlar
axs[0].plot(xi, true_mode1, label='Gerçek Mod 1 (Gövde)', color='blue', linewidth=2)
axs[0].plot(xi, true_mode2, label='Gerçek Mod 2 (Rüzgar)', color='red', linewidth=2)
axs[0].set_title('1. Orijinal Uzaysal Sinyaller (Ground Truth)', fontsize=14, fontweight='bold')
axs[0].legend()
axs[0].grid(True)

# B. PCA Modları (Normalize edilmiş)
pca_m1 = spatial_modes_pca[:, 0] / np.max(np.abs(spatial_modes_pca[:, 0]))
pca_m2 = spatial_modes_pca[:, 1] / np.max(np.abs(spatial_modes_pca[:, 1]))
# Yönleri (işaretleri) gerçek sinyalle eşitleme
if pca_m1[100] < 0: pca_m1 = -pca_m1
if pca_m2[200] < 0: pca_m2 = -pca_m2

axs[1].plot(xi, pca_m1, label='PCA Mod 1', linestyle='--', color='purple', linewidth=2)
axs[1].plot(xi, pca_m2, label='PCA Mod 2', linestyle='--', color='orange', linewidth=2)
axs[1].set_title('2. PCA Uzaysal Modları (Varyans Tabanlı - Karışmış)', fontsize=14, fontweight='bold')
axs[1].legend()
axs[1].grid(True)

# C. ICA Modları (Normalize edilmiş)
ica_m1 = spatial_modes_ica[:, 0] / np.max(np.abs(spatial_modes_ica[:, 0]))
ica_m2 = spatial_modes_ica[:, 1] / np.max(np.abs(spatial_modes_ica[:, 1]))
# Yönleri (işaretleri) gerçek sinyalle eşitleme
if ica_m1[100] < 0: ica_m1 = -ica_m1
if ica_m2[200] > 0: ica_m2 = -ica_m2

axs[2].plot(xi, ica_m1, label='ICA Mod 1', linestyle='-.', color='teal', linewidth=2)
axs[2].plot(xi, ica_m2, label='ICA Mod 2', linestyle='-.', color='brown', linewidth=2)
axs[2].set_title('3. Fast ICA Uzaysal Modları (Bağımsızlık Tabanlı - Şekiller Bozuk)', fontsize=14, fontweight='bold')
axs[2].legend()
axs[2].grid(True)

plt.tight_layout()
plt.show()