import numpy as np
import matplotlib.pyplot as plt


# DURAN DALGA VERİSİNİ ÜRETME (f(x,t) = cos(x)*cos(t))

# Uzay ekseni (x) ve Zaman ekseni (t)
x = np.linspace(-np.pi, np.pi, 200)
t = np.linspace(0, 4*np.pi, 100)
dt = t[1] - t[0]

Xgrid, Tgrid = np.meshgrid(x, t)
# Dalga verisi: f(x,t) = cos(x) * cos(t)
# Boyut: (Uzay, Zaman) -> (200, 100)
data = (np.cos(Xgrid) * np.cos(Tgrid)).T

print("Boyut: (Uzay, Zaman) -> ", data.shape)
print(data)

# 1. ALGORİTMA STANDART DMD (Çöküş yaşanacak senaryo)

X1_std = data[:, :-1]
X2_std = data[:, 1:]

# SVD ve Standart DMD
U_std, S_std_vals, Vh_std = np.linalg.svd(X1_std, full_matrices=False)

r_std = 2 # 2 mod arıyoruz
U_r_std = U_std[:, :r_std]
S_r_std = np.diag(S_std_vals[:r_std])
V_r_std = Vh_std[:r_std, :].conj().T

A_tilde_std = U_r_std.conj().T @ X2_std @ V_r_std @ np.linalg.inv(S_r_std)
eigenvalues_std, W_std = np.linalg.eig(A_tilde_std) # lambda, W

# Sürekli zaman (Continuous-time) frekanslarını bulma: omega = ln(lambda) / dt
omega_std = np.log(eigenvalues_std) / dt # lambda / dt

# DMD Modları (Phi)
phi = X2_std @ V_r_std @ np.linalg.inv(S_r_std) @ W_std

# Başlangıç genlikleri (b) - Sözde ters (pinv) ile bulunur
x0_std = X1_std[:, 0]
b_std = np.linalg.pinv(phi) @ x0_std

# Zaman Dinamikleri ve Sistemin Yeniden İnşası (X_dmd)
time_dynamics_std = np.zeros((r_std, len(t)), dtype=complex)
for i in range(len(t)):
    time_dynamics_std[:, i] = b_std * np.exp(omega_std * t[i])
    
X_dmd_std = phi @ time_dynamics_std

print("--- STANDART DMD SONUÇLARI ---")
print("Bulunan Frekanslar (Omega):", np.round(omega_std, 4))
print("Gerçekte olması gereken: Saf sanal titreşim (+i ve -i)")
print("Sonuç: STANDART DMD ÇÖKTÜ! (Sanal eksende titreşimi bulamadı, reel değerler verdi)\n")

# ALGORİTMA 2: GECİKME KOORDİNATLI DMD (Hankel DMD - Başarılı Senaryo)

# Veriyi 1 adım gecikmeli olarak alt alta yığıyoruz (Shift-Stacking / Frame Stacking)
# H matrisi boyutu iki katına çıkacak: (400, 99)
H_data = np.vstack((data[:, :-1], data[:, 1:]))
# Artık zaman vektörümüz 1 adım kısaldı
t_aug = t[:-1]
print("H matris boyutu -> ", H_data.shape)

# Hankel verisini geçmiş ve gelecek olarak ayırma
H1 = H_data[:, :-1]
H2 = H_data[:, 1:]

# SVD ve Hankel DMD
U_H, S_H_vals, Vh_H = np.linalg.svd(H1, full_matrices=False)

r_H = 2 # Yine 2 mod arıyoruz
U_r_H = U_H[:, :r_H]
S_r_H = np.diag(S_H_vals[:r_H])
V_r_H = Vh_H[:r_H, :].conj().T

A_tilde_H = U_r_H.conj().T @ H2 @ V_r_H @ np.linalg.inv(S_r_H)
eigenvalues_H, W_H = np.linalg.eig(A_tilde_H)
print("eigenvalues: ",eigenvalues_H)

omega_H = np.log(eigenvalues_H) / dt

# Augmented (Genişletilmiş) DMD Modları
Phi_aug = H2 @ V_r_H @ np.linalg.inv(S_r_H) @ W_H

# Başlangıç genlikleri
x0_aug = H1[:, 0]
b_aug = np.linalg.pinv(Phi_aug) @ x0_aug

# Zaman Dinamikleri ve Yeniden İnşa
time_dynamics_aug = np.zeros((r_H, len(t_aug)), dtype=complex)
for i in range(len(t_aug)):
    time_dynamics_aug[:, i] = b_aug * np.exp(omega_H * t_aug[i])

# Genişletilmiş Uzaydaki DMD Çıktısı (Alt alta 2 sistem var)
H_dmd_aug = Phi_aug @ time_dynamics_aug

# Sadece gerçek fiziksel sisteme ait olan İLK YARIYI (İlk 200 satırı) alıyoruz
X_dmd_aug = H_dmd_aug[:200, :]

print("--- HANKEL DMD (GECİKME KOORDİNATLARI) SONUÇLARI ---")
print("Bulunan Frekanslar (Omega):", np.round(omega_H, 4))
print("Sonuç: HANKEL DMD SONUCU BAŞARILI! (Tam olarak +1j ve -1j sanal titreşim frekanslarını yakalar)")


# GÖRSELLEŞTİRME (KİTAPTAKİ GRAFİKLER)

# Kompleks sayıların sadece reel kısımlarını çizdirelim
fig, axs = plt.subplots(1, 3, figsize=(18, 5))

# 1. Gerçek Sistem
c1 = axs[0].pcolor(t, x, data, cmap='RdBu', shading='auto', vmin=-1, vmax=1)
axs[0].set_title('Gerçek Duran Dalga\n$f(x,t) = \cos(x)\cos(t)$', fontsize=14, fontweight='bold')
axs[0].set_xlabel('Zaman (t)')
axs[0].set_ylabel('Uzay (x)')
fig.colorbar(c1, ax=axs[0])

# 2. Standart DMD
c2 = axs[1].pcolor(t, x, np.real(X_dmd_std), cmap='RdBu', shading='auto', vmin=-1, vmax=1)
axs[1].set_title('Standart DMD\n(ÇÖKÜŞ: Titreşimi Kaçırdı)', fontsize=14, fontweight='bold', color='darkred')
axs[1].set_xlabel('Zaman (t)')
fig.colorbar(c2, ax=axs[1])

# 3. Augmented (Hankel) DMD
# Zaman matrisini Hankel'e göre 1 adım kırptığımız için Tgrid ve Xgrid'i de fixleyelim
Tgrid_aug, Xgrid_aug = np.meshgrid(t_aug, x)
c3 = axs[2].pcolor(t_aug, x, np.real(X_dmd_aug), cmap='RdBu', shading='auto', vmin=-1, vmax=1)
axs[2].set_title('Data Augmented DMD\n(ZAFER: Kusursuz Yeniden İnşa)', fontsize=14, fontweight='bold', color='darkgreen')
axs[2].set_xlabel('Zaman (t)')
fig.colorbar(c3, ax=axs[2])

plt.tight_layout()
plt.show()