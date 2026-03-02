import numpy as np
import matplotlib.pyplot as plt

# 1. PARAMETRELER VE VERİ OLUŞTURMA
xi = np.linspace(-10, 10, 400)
t = np.linspace(0, 4*np.pi, 200)
dt = t[1] - t[0]
Xgrid, Tgrid = np.meshgrid(xi, t)

f1 = (1 / np.cosh(Xgrid + 3)) * np.exp(2.3j * Tgrid)
f2 = (1 / np.cosh(Xgrid)) * np.tanh(Xgrid) * np.exp(2.8j * Tgrid)
X_total = (f1 + f2).T

# 2. DMD ALGORİTMASI
X1 = X_total[:, :-1]
X2 = X_total[:, 1:]

U, S, Vh = np.linalg.svd(X1, full_matrices=False)
r = 2
Ur = U[:, :r]
Sr = np.diag(S[:r])
Vr = Vh[:r, :].conj().T

# Atilde ve Özdeğerler
Atilde = Ur.conj().T @ X2 @ Vr @ np.linalg.inv(Sr)
lambda_vals, W = np.linalg.eig(Atilde)
omega = np.log(lambda_vals) / dt

# Uzaysal Modlar (Phi) ve Başlangıç Genlikleri (b)
Phi = X2 @ Vr @ np.linalg.inv(Sr) @ W
b = np.linalg.pinv(Phi) @ X1[:, 0]

# 3. VERİ EŞLEŞTİRME VE DÜZENLEME (Çizim Öncesi)
# Phi matrisi kompleks olduğu için gerçek kısımlarını alıyoruz
dmd_mode_1 = np.real(Phi[:, 0])
dmd_mode_2 = np.real(Phi[:, 1])

# Gerçek modlar (Ground truth)
true_mode1 = 1 / np.cosh(xi + 3)
true_mode2 = (1 / np.cosh(xi)) * np.tanh(xi)

# Algoritma modların sırasını rastgele verebilir. Frekanslara (omega) bakarak 
# hangi DMD modunun hangi gerçek moda ait olduğunu tespit ediyoruz.
if np.abs(np.imag(omega[0]) - 2.3) < np.abs(np.imag(omega[1]) - 2.3):
    mode_a = dmd_mode_1 / np.max(np.abs(dmd_mode_1))
    mode_b = dmd_mode_2 / np.max(np.abs(dmd_mode_2))
else:
    mode_a = dmd_mode_2 / np.max(np.abs(dmd_mode_2))
    mode_b = dmd_mode_1 / np.max(np.abs(dmd_mode_1))

# Yön (işaret) düzeltmeleri (Eksiler artılara dönsün ki grafikler üst üste otursun)
if mode_a[np.argmax(np.abs(true_mode1))] * true_mode1[np.argmax(np.abs(true_mode1))] < 0:
    mode_a = -mode_a
if mode_b[np.argmax(np.abs(true_mode2))] * true_mode2[np.argmax(np.abs(true_mode2))] < 0:
    mode_b = -mode_b

# 4. GÖRSELLEŞTİRME (MATPLOTLIB)
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# Mod 1 Karşılaştırması (Gövde)
axs[0].plot(xi, true_mode1, 'k-', linewidth=4, label='Gerçek Sinyal 1')
axs[0].plot(xi, mode_a, 'r--', linewidth=2, label='DMD Modu ($\Phi_1$)')
axs[0].set_title(f"1. Sinyal Eşleşmesi (Frekans $\omega approx 2.3j$)")
axs[0].legend()
axs[0].grid(True)

# Mod 2 Karşılaştırması (Rüzgar)
axs[1].plot(xi, true_mode2, 'k-', linewidth=4, label='Gerçek Sinyal 2')
axs[1].plot(xi, mode_b, 'b--', linewidth=2, label='DMD Modu ($\Phi_2$)')
axs[1].set_title(f"2. Sinyal Eşleşmesi (Frekans $\omega approx 2.8j$)")
axs[1].legend()
axs[1].grid(True)

plt.suptitle("DMD'nin Çıkardığı Uzaysal Modlar ($\Phi$) vs Gerçek Sinyaller", fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

# İsteğe bağlı konsol çıktıları
print(f"Tespit Edilen Frekanslar (Omega): {np.round(np.imag(omega), 2)}j")
print(f"Başlangıç Genlikleri (b vektörü): {np.round(np.abs(b), 2)}")