import numpy as np
import matplotlib.pyplot as plt


# 1. DMDc FONKSİYONU (Ajanın "Dünya Modeli" Öğrenme Motoru)

def compute_dmdc(X, X_prime, Upsilon, p, r):
    """
    Simülasyonumuz çok küçük ise (sadece 2 boyutlu) olduğu için, ajanın boyut indirgeme (Latent Space) yapmasına gerek yoktur. 
    Bu durumda fonksiyondan dönen A_exact ve B_exact (orijinal uzaydaki) matrislerini kullanabiliriz.
    """
    n = X.shape[0]     # Durum boyutu
    q = Upsilon.shape[0] # Aksiyon boyutu

    # 1. Genişletilmiş Matris
    omega = np.vstack((X, Upsilon))
    U_tilde, S_tilde_vals, Vh_tilde = np.linalg.svd(omega, full_matrices=False)

    U_tilde_p = U_tilde[:, :p]
    S_tilde_p = np.diag(S_tilde_vals[:p])
    V_tilde_p = Vh_tilde[:p, :].conj().T

    # 2. Splitting (Koparma)
    U_tilde_1 = U_tilde_p[:n, :]        
    U_tilde_2 = U_tilde_p[n:n+q, :]     

    # 3. İkinci İzdüşüm (Latent Space)
    U_hat, S_hat_vals, Vh_hat = np.linalg.svd(X_prime, full_matrices=False)
    U_hat_r = U_hat[:, :r]

    # 4. G Matrisi ve İndirgenmiş (Latent) A, B Matrisleri
    G_approx = X_prime @ V_tilde_p @ np.linalg.inv(S_tilde_p)
    A_tilde = U_hat_r.conj().T @ G_approx @ U_tilde_1.conj().T @ U_hat_r
    B_tilde = U_hat_r.conj().T @ G_approx @ U_tilde_2.conj().T
    
    # Simülasyonumuz için gereken Orijinal Uzay (Exact) Matrisleri
    A_exact = G_approx @ U_tilde_1.conj().T
    B_exact = G_approx @ U_tilde_2.conj().T

    # 5. Spektral Analiz ve DMD Modları (Kitaptaki eksik kısım tamamlandı)
    lambda_vals, W = np.linalg.eig(A_tilde)
    Phi = G_approx @ U_tilde_1.conj().T @ U_hat_r @ W
    
    return A_tilde, B_tilde, U_hat_r, lambda_vals, Phi, A_exact, B_exact


# 2. YÜKSEK BOYUTLU SİMÜLASYON OLUŞTURMA (50 Sensörlü Uçak)
m = 200 # 200 zaman adımı
n_true = 2  # Gerçek fizik 2 boyutlu
q = 1       # Aksiyon 1 boyutlu
n_sensors = 50 # Uçağın üzerindeki sensör sayısı (Yüksek Boyut!)

# Gerçek (Gizli) Fizik Matrisleri
A_true = np.array([[0.9, -0.2],
                   [0.1,  0.8]])
B_true = np.array([[1.0],
                   [0.5]])

# Gizli 2 boyutu, 50 boyutlu sensör verisine dağıtan ölçüm matrisi (Rastgele)
np.random.seed(42)
C_high_dim = np.random.randn(n_sensors, n_true) 

# Veri Üretimi
X_hidden = np.zeros((n_true, m))
X_hidden[:, 0] = [2.0, -1.0]
U_data = np.sin(np.linspace(0, 10, m-1)).reshape(1, m-1)

for k in range(m-1):
    X_hidden[:, k+1] = A_true @ X_hidden[:, k] + B_true @ U_data[:, k]

# SENSÖR VERİSİ (Ajanın gördüğü 50 boyutlu karmaşık veri)
X_data_high = C_high_dim @ X_hidden 


# 3. LATENT DMDc ALGORİTMASINI ÇALIŞTIRMA
X = X_data_high[:, :-1]
X_prime = X_data_high[:, 1:]
Upsilon = U_data

# BOYUT KESMELERİ (ASIL OLAY)
# 50 sensör verisine bakıyoruz ama "Bu işin arkasında 2 temel dinamik var" diyerek r=2 seçiyoruz.
r = 2 
p = r + q # p = 2 + 1 = 3 (Birleşik uzay için kesme)

A_tilde, B_tilde, U_hat_r, lambda_vals, Phi, A_exact, B_exact = compute_dmdc(X, X_prime, Upsilon, p, r)

print("--- BOYUT (Sıfırdan Latent Uzaya) ---")
print(f"Orijinal A Matrisi Boyutu Olması Gereken: {n_sensors}x{n_sensors}")
print(f"Latent (Öğrenilen) A_tilde Matrisi Boyutu: {A_tilde.shape[0]}x{A_tilde.shape[1]}")
print(f"Latent (Öğrenilen) B_tilde Matrisi Boyutu: {B_tilde.shape[0]}x{B_tilde.shape[1]}")

# 4. LATENT SPACE'DE (GİZLİ UZAYDA) GELECEĞİ TAHMİN ETME
# Ajanın tahmin edeceği 50 sensörlük gelecek verisi
X_pred_high = np.zeros((n_sensors, m))
X_pred_high[:, 0] = X_data_high[:, 0]

# ADIM A: İlk durumu Encoder ile sıkıştır (50D -> 2D)
x_latent_current = U_hat_r.conj().T @ X_data_high[:, 0]

for k in range(m-1):
    # ADIM B: Latent Dynamics (Küçücük matrislerle geleceği çok hızlı hesapla)
    x_latent_next = A_tilde @ x_latent_current + B_tilde @ U_data[:, k]
    
    # ADIM C: Decoder ile sonucu geri büyüt ve kaydet (2D -> 50D)
    X_pred_high[:, k+1] = U_hat_r @ x_latent_next
    
    # Döngüyü ilerlet
    x_latent_current = x_latent_next


# 5. GÖRSELLEŞTİRME (Herhangi bir sensörü kontrol edelim)
sensor_id = 10 # 50 sensörden 10. sensörün tahminini çizdirelim

plt.figure(figsize=(10, 4))
plt.plot(X_data_high[sensor_id, :], 'k-', linewidth=4, label=f'Gerçek Sensör {sensor_id} Çıktısı')
plt.plot(X_pred_high[sensor_id, :], 'r--', linewidth=2, label=f'Latent DMDc Tahmini (Decoder Çıktısı)')
plt.title(f"Yüksek Boyutlu Sensör Verisinin Latent Space'den Geri Çatılması (Reconstruction)", fontweight='bold')
plt.legend()
plt.grid(True)
plt.show()