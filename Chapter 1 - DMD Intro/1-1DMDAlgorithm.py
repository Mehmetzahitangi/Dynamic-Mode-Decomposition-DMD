import numpy as np

def DMD_compute(self, X1, X2, r, dt):
    """
    Dynamic Mode Decomposition (Exact DMD)
    
    INPUTS:
    X1: X, data matrix / past states matrix (n*m)
    X2: X', shifted data matrix / Future Data Matrix (n*m)
    r: target rank of SVD (low dim. space)
    dt: time step, X1 to X2 (X to X')

    OUTPUTS:
    Phi: DMD modes
    Omega:  the continuous-time DMD eigenvalues
    lambda_vals:  the discrete-time DMD eigenvalues
    b:  a vector of magnitudes of modes Phi
    """

    # 1. SVD and rank reduction / boyut sıkıştırma
    U, S, Vh = np.linalg.svd(X1, full_matrices=False) # İşleminde dönen Vh, aslında V matrisinin kompleks eşlenik transpozesidir (V* veya V^H).
    
    # Truncate to rank-r / r boyutuna kesme
    U_r = U[:, :r]
    S_r = np.diag(S[:r])
    V_r = Vh[:r, :].conj().T # Vh'nin kompleks eşlenik transpozesi aldık 

    # Use Matrix Product 

    # 2. Atilde(Low Dim. Dynamic Matrix) Calculation
    # Formula: Atilde = U_r^* * X2 * V_r * S_r^{-1}
    # U_r.conj() ) ==  U_r'
    atilde = U_r.conj().T @ X2 @ V_r @ np.linalg.inv(S_r) #S^-1 için linalg.inv kullandık
    
    # 3. Eigendecomposition / Eigenvalue dec.
    lambda_vals, W = np.linalg.eig(atilde)  # lambda_vals'ı doğrudan bir 1D vektör olarak verir.
    # lambda_vals/D ==  discrete-time eigenvalues => [W_r, D] = eig(Atilde); lambda = diag(D); 
    # W_r eigenvalue matrix, D eigenvalue barındıran kçşegen bir matristir, daha sonra lambda = diag(D);  ile o matrisin içindeki vektörü çekersin.

    # 4. Phi calculation of Exact DMD modes
    Phi = X2 @ V_r / S_r @ W # DMD modes

    # continuous-time eigenvalues (Omega)
    omega = np.log(lambda_vals)/dt 

    # compute DMD mode amplitudes(genlik) (b)
    x1 = X1[:, 0]
    b = np.linalg.pinv(Phi) @ x1

    return Phi, omega, lambda_vals, b

# ----- GELECEK DURUMU TAHMİN ETME (Reconstruction/Prediction Future State) -----
def predict_future(Phi, omega, b, t):
    """ İstenilen t anındaki sistemi (örneğin drone'un konumunu) tahmin eder 
    x_dmd: the data matrix reconstructed by Phi, omega, b"""
    # Formül: x(t) = Phi * exp(omega * t) * b
    time_dynamics = np.exp(omega * t) * b
    x_dmd = Phi @ time_dynamics
    return x_dmd

#5. Sistem Çözücüsü (\ Operatörü vs pinv)MATLAB: 
# Başlangıç genliklerini ($b$) bulmak için kullanılan b = Phi \ x1 ifadesindeki \ (backslash) operatörü, 
# MATLAB'ın en güçlü silahıdır. 
# Phi matrisinin yapısına bakar, eğer kare matris değilse arka planda otomatik olarak en küçük kareler
#  (least squares) veya pseudoinverse hesabı yapar.
# Python: Python'da böyle sihirli bir operatör yoktur.
# Nasıl Düzelttik: Formülsel karşılığı olan Moore-Penrose sözde tersini (pseudoinverse) açıkça kodlamak zorundaydım. 
# Bu yüzden b = np.linalg.pinv(Phi) @ x1 kullandım. np.linalg.pinv, 
# SVD tabanlı bir ters alma işlemi yaparak MATLAB'ın \ operatörünün yaptığına en yakın ve sayısal olarak en stabil (numerically stable) sonucu verir.

"""
Kod mimarisindeki bu çeviri detayları,
DMD algoritmalarını bir yapay zeka projesinin (örneğin PyTorch tabanlı bir RL ajanının)
veri ön işleme (preprocessing) veya kayıp fonksiyonu (loss function)
bloğuna entegre ederken boyut (shape) hataları almanı engelleyecektir.
"""