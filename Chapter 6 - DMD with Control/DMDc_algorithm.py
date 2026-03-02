import numpy as np

def compute_dmdc(X, X_prime, Upsilon, p, r):
    """
    DMDc (Dynamic Mode Decomposition with Control) Algoritması
    
    Parametreler:
    X       : Geçmiş durumlar matrisi (n x m) -> RL'deki 'State'
    X_prime : Gelecek durumlar matrisi (n x m) -> RL'deki 'Next State'
    Upsilon : Kontrol/Aksiyon matrisi (q x m) -> RL'deki 'Action'
    p       : Omega matrisi (Birleşik uzay) için kesme rank'ı
    r       : X_prime matrisi (İndirgenmiş Latent Space) için kesme rank'ı
    """

    n = X.shape[0]     # Durum uzayının boyutu (örneğin sensör sayısı)
    q = Upsilon.shape[0] # Aksiyon uzayının boyutu (örneğin motor sayısı)

    # 1. Durum ve Kontrolü Birleştirme (Augmented Matrix)
    # omega = [X; Upsilon] alt alta birleştiriliyor
    omega = np.vstack((X, Upsilon))

    # Omega'nın SVD'si
    U_tilde, S_tilde_vals, Vh_tilde = np.linalg.svd(omega, full_matrices=False)

    U_tilde_p = U_tilde[:, :p]
    S_tilde_p = np.diag(S_tilde_vals[:p])
    V_tilde_p = Vh_tilde[:p, :].conj().T


    # 2. U_tilde matrisini parçalama (Splitting)

    # U_tilde matrisini durumlar (A) ve aksiyonlar (B) için ikiye bölüyoruz
    U_tilde_1 = U_tilde_p[:n, :]        # İlk n satır, uçağın aerodinamiği olarak düşün
    U_tilde_2 = U_tilde_p[n:n+q, :]     # Sonraki q satır, uçağa verilen gaz/fren tepkisi vb düşün

    # ADIM 3: Latent Space (Gizli Uzay) İçin İkinci İzdüşüm, gelecek uzayını indirgeme 
    # Çıktı matrisi X_prime'ın (X') SVD'si ve r boyutuna sıkıştırılması
    U_hat, S_hat_vals, Vh_hat = np.linalg.svd(X_prime, full_matrices=False)
    
    U_hat_r = U_hat[:, :r]
    # S_hat_r ve V_hat_r bu algoritmada matrisleri bulmak için direkt gerekmiyor,
    # U_hat_r izdüşüm yapmak için yeterli.

    # ADIM 4: İndirgenmiş A ve B Matrislerinin Hesaplanması
    # Ortak bir çarpan matrisi tanımlayalım (Kod tekrarını önlemek için)
    # G_approx = X_prime * V_tilde * S_tilde^{-1}
    G_approx = X_prime @ V_tilde_p @ np.linalg.inv(S_tilde_p)
    
    # A_tilde = U_hat^* * X_prime * V_tilde * S_tilde^{-1} * U_tilde_1^* * U_hat
    A_tilde = U_hat_r.conj().T @ G_approx @ U_tilde_1.conj().T @ U_hat_r
    # B_tilde = U_hat^* * X_prime * V_tilde * S_tilde^{-1} * U_tilde_2^*
    B_tilde = U_hat_r.conj().T @ G_approx @ U_tilde_2.conj().T
    
    # (Opsiyonel) Eğer tam boyutlu (Latent space'de olmayan) orjinal A ve B isteniyorsa:
    A_exact = G_approx @ U_tilde_1.conj().T
    B_exact = G_approx @ U_tilde_2.conj().T

    # 5. Spektral Analiz  Bulduğumuz A tilde matrisinin özdeğerleri (lambda), uçağın kararlı (stable) mı yoksa düşmeye meyilli (unstable) mi olduğunu söyler.
    lambda_vals, W = np.linalg.eig(A_tilde)

    # Phi = X_prime * V_tilde_p * inv(S_tilde_p) * U_tilde_1^* * U_hat_r * W
    Phi = G_approx @ U_tilde_1.conj().T @ U_hat_r @ W # dinamiklerin gerçek dünyadaki şekli (sensırlerdeki veya piksellerdeki)
    
    
    return A_tilde, B_tilde, U_hat_r, lambda_vals, Phi
