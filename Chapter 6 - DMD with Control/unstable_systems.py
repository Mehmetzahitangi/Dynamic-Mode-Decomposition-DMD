import numpy as np

def compute_dmdc_known_B(X, X_prime, Upsilon, B_known, r):
    """
    Bilinen B Matrisi(aracın kontrol matrisi) ile DMDc Algoritması (Kararsız Sistemler İçin İdeal)
    
    Parametreler:
    X         : Geçmiş durumlar (n x m)
    X_prime   : Gelecek durumlar (n x m)
    Upsilon   : Kontrol/Aksiyon komutları (q x m)
    B_known   : Sistem mühendislerinden alınan kesin kontrol matrisi (n x q)
    r         : SVD kesme rank'ı (İndirgenmiş uzay boyutu)
    """
    
    # 1. ADIM: "Zorlanmamış" (Unforced) Geleceği Bulmak
    # X_prime = A*X + B*Upsilon denkleminden B*Upsilon'u çıkartıyoruz.
    # Bu, "Ajan hiç gaza basmasaydı uçak nereye savrulurdu?" sorusunun cevabıdır.
    X_prime_unforced = X_prime - (B_known @ Upsilon)
    
    # 2. ADIM: Arındırılmış Veriyle Standart DMD Çalıştırmak
    # Artık elimizde sadece doğal fiziği (A) yansıtan saf bir veri var.
    U, S_vals, Vh = np.linalg.svd(X, full_matrices=False)
    
    # Boyut İndirgeme (Rank r)
    Ur = U[:, :r]
    Sr = np.diag(S_vals[:r])
    Vr = Vh[:r, :].conj().T
    
    # 3. ADIM: A Matrisini Çıkarma
    # A_tilde = Ur^* * X_prime_unforced * Vr * Sr^{-1}
    A_tilde = Ur.conj().T @ X_prime_unforced @ Vr @ np.linalg.inv(Sr)
    
    # (İsteğe bağlı) Özdeğer Analizi: Bu değerler sistemin ne kadar 
    # kararsız (unstable) olduğunu matematiksel olarak kanıtlar!
    lambda_vals, W = np.linalg.eig(A_tilde)
    
    return A_tilde, lambda_vals