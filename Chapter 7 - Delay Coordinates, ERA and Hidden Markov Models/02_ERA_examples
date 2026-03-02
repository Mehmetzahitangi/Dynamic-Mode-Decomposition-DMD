import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# ALGORİTMA 7.1: DÜRTÜ YANITI (MARKOV PARAMETRELERİ) ÜRETME
# =============================================================================
# Amaç: Bilinmeyen bir sistem yaratmak ve ona bir darbe (impulse) vurup 
# sadece tek bir sensörden okunan verileri (y_k) toplamaktır.

# 1. Gerçek Sistem (Bizim bilmediğimiz Kara Kutu)
# Uçağın 3 boyutlu, karmaşık ve salınımlı bir fiziği olsun
A_true = np.array([[0.8, -0.5, 0.0],
                   [0.5,  0.8, 0.0],
                   [0.0,  0.0, 0.9]])
B_true = np.array([[1.0], 
                   [0.0], 
                   [0.0]]) # Sadece 1. duruma darbe vurabiliyoruz
C_true = np.array([[0.0, 1.0, 1.0]]) # Sadece 2. ve 3. durumun toplamını ölçebiliyoruz

m = 100 # 100 zaman adımı veri toplayacağız
y_data = np.zeros(m) # Sadece 1 boyutlu sensör okumamız!

# 2. Impulse (Darbe) Uygulama ve Veri Toplama
# t=0 anında u=1 verilir, geri kalan tüm zamanlarda u=0'dır.
x_current = B_true # x_1 = B * u_0 (u_0 = 1 olduğu için x_1 = B)
y_data[0] = (C_true @ x_current)[0, 0]

for k in range(1, m):
    x_current = A_true @ x_current # Serbest salınım (A^k)
    y_data[k] = (C_true @ x_current)[0, 0] # Sensör okuması (C * x)

# Elimizde sadece y_data adında tek boyutlu bir dizi var. Fiziği bilmiyoruz!


# =============================================================================
# ALGORİTMA 7.2: ERA (EIGENSYSTEM REALIZATION ALGORITHM)
# =============================================================================
# Amaç: O tek boyutlu y_data dizisinden, A, B ve C matrislerini bulmak.

# 1. Hankel Matrislerini Oluşturma
n_rows = 40 # Geçmişi ne kadar yığacağımız (Pencere boyutu)
n_cols = 40
H = np.zeros((n_rows, n_cols))
H_prime = np.zeros((n_rows, n_cols))

for i in range(n_rows):
    for j in range(n_cols):
        H[i, j] = y_data[i + j + 1]       # Y_1, Y_2, Y_3...
        H_prime[i, j] = y_data[i + j + 2] # Y_2, Y_3, Y_4... (Bir adım kaydırılmış)

# 2. Hankel Matrisinin SVD'sini Alma
U, S_vals, Vh = np.linalg.svd(H, full_matrices=False)

# 3. Model İndirgeme (Rank Seçimi)
r = 3 # Sistemimizin 3 boyutlu olduğunu biliyoruz (veya S_vals grafiğine bakarak anlarız)
Ur = U[:, :r]
Sr = np.diag(S_vals[:r])
Vr = Vh[:r, :].conj().T

# 4. Gözlemlenebilirlik ve Kontrol Edilebilirlik Matrislerinin Kökleri
Sr_sqrt = np.sqrt(Sr)
Sr_inv_sqrt = np.linalg.inv(Sr_sqrt)

# 5. Gizli Uzaydaki (Latent Space) A, B, C Matrislerini Çıkarma
A_era = Sr_inv_sqrt @ Ur.conj().T @ H_prime @ Vr @ Sr_inv_sqrt

# B matrisi, V matrisinin ilk sütunlarından elde edilir
E_u = np.zeros((n_cols, 1)); E_u[0, 0] = 1
B_era = Sr_sqrt @ Vr.conj().T @ E_u

# C matrisi, U matrisinin ilk satırlarından elde edilir
E_y = np.zeros((n_rows, 1)); E_y[0, 0] = 1
C_era = E_y.conj().T @ Ur @ Sr_sqrt


# =============================================================================
# ALGORİTMA 7.3: ERA MODELİNİN YENİDEN İNŞASI VE KARŞILAŞTIRMA
# =============================================================================
# Amaç: Öğrendiğimiz A_era, B_era, C_era matrisleriyle kendi simülasyonumuzu 
# yapıp, orijinal uçağın sensör verisiyle birebir eşleşip eşleşmediğine bakmak.

y_era_pred = np.zeros(m)

# Öğrenilen modelle Impulse Response (Darbe Yanıtı) simülasyonu
x_era_current = B_era
y_era_pred[0] = (C_era @ x_era_current)[0, 0]

for k in range(1, m):
    x_era_current = A_era @ x_era_current
    y_era_pred[k] = (C_era @ x_era_current)[0, 0]

# --- GÖRSELLEŞTİRME ---
plt.figure(figsize=(10, 5))
plt.plot(y_data, 'k-', linewidth=4, label='Gerçek Sensör Okuması (Algoritma 7.1)')
plt.plot(y_era_pred, 'r--', linewidth=2, label='ERA Algoritması Tahmini (Algoritma 7.3)')
plt.title("ERA: Tek Bir Sensörden Gizli Fiziğin Yeniden İnşası", fontweight='bold')
plt.xlabel("Zaman Adımı")
plt.ylabel("Sensör (İrtifa) Ölçümü")
plt.legend()
plt.grid(True)
plt.show()

print("--- ERA MATRİS BOYUTLARI ---")
print(f"Öğrenilen Latent A Boyutu: {A_era.shape}")
print(f"Öğrenilen Latent B Boyutu: {B_era.shape}")
print(f"Öğrenilen Latent C Boyutu: {C_era.shape}")

# Özdeğerleri (Fiziğin Ruhunu) karşılaştıralım:
lambda_true = np.sort(np.linalg.eigvals(A_true))
lambda_era = np.sort(np.linalg.eigvals(A_era))

print("\nGerçek Sistemin Özdeğerleri:\n", np.round(lambda_true, 4))
print("ERA'nın Bulduğu Özdeğerler:\n", np.round(lambda_era, 4))