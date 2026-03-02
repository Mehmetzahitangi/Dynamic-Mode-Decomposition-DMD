import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


# 1. Parametreler ve Koopman Lineer Dinamikleri (A Matrisi)
mu = -0.05
lam = -1.0

# 3x3 Koopman Matrisi (Tamamen lineer)
A = np.array([
    [mu, 0, 0],
    [0, lam, -lam],
    [0, 0, 2 * mu]
])

# Özdeğer ve Özvektör analizi
D, T = np.linalg.eig(A)

# Kararlı alt uzayın (yeşil düzlem) eğimini hesaplama
slope_stab_man = T[2, 2] / T[1, 2] 

# 2. Koopman Yörüngelerini Entegre Etme (Simülasyon)
# Üç farklı başlangıç koşulu (y0A, y0B, y0C)
y0A = np.array([1.5, 1.0, 2.25])
y0B = np.array([1.0, -1.0, 1.0])
y0C = np.array([2.0, -1.0, 4.0])

# Zaman aralığı (0'dan 100'e kadar)
t_span = (0, 100)
t_eval = np.arange(0, 100.01, 0.01)

# Türev fonksiyonu: dy/dt = A * y
def koopman_ode(t, y):
    return A @ y

# SciPy ile diferansiyel denklemleri çözme (MATLAB'in ode45 karşılığı)
solA = solve_ivp(koopman_ode, t_span, y0A, t_eval=t_eval)
solB = solve_ivp(koopman_ode, t_span, y0B, t_eval=t_eval)
solC = solve_ivp(koopman_ode, t_span, y0C, t_eval=t_eval)

yA = solA.y.T
yB = solB.y.T
yC = solC.y.T




# --- ALGORITHM 3.2: Koopman Lineerleştirmesini Çizdirme ---

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 1. Çekici Manifold y_2 = y_1^2 (KIRMIZI MANİFOLD)
X, Z = np.meshgrid(np.arange(-2, 2.01, 0.01), np.arange(1, 4.01, 0.01))
Y = X**2
ax.plot_surface(X, Y, Z, color='r', alpha=0.1, edgecolor='none')

# 2. Değişmez Set y_3 = y_1^2 (MAVİ MANİFOLD)
X1, Y1 = np.meshgrid(np.arange(-2, 2.01, 0.01), np.arange(1, 4.01, 0.01))
Z1 = X1**2
ax.plot_surface(X1, Y1, Z1, color='b', alpha=0.1, edgecolor='none')

# 3. Koopman Lineer Sisteminin Kararlı Değişmez Alt Uzayı (YEŞİL DÜZLEM)
X2, Y2 = np.meshgrid(np.arange(-2, 2.01, 0.01), np.arange(0, 4.01, 0.01))
Z2 = slope_stab_man * Y2
ax.plot_surface(X2, Y2, Z2, color=[0.3, 0.7, 0.3], alpha=0.7, edgecolor='none')

# 4. Kesişim Çizgileri
x_line = np.arange(-2, 2.01, 0.01)
# Yeşil ve Mavi yüzeylerin kesişimi
ax.plot(x_line, (1/slope_stab_man)*x_line**2, x_line**2, color='g', linewidth=2)
# Kırmızı ve Mavi yüzeylerin kesişimi
ax.plot(x_line, x_line**2, x_line**2, color='r', linewidth=2)
# Tabandaki gölge (kırmızı manifoldun izdüşümü)
ax.plot(x_line, x_line**2, -1 + 0*x_line, color='r', linewidth=2)

# 5. Koopman Yörüngelerini (Trajectories) Çizdirme
# Tabandaki 2 Boyutlu Gölgeler (Orijinal Nonlinear Sistem - Siyah Çizgiler)
ax.plot(yA[:, 0], yA[:, 1], -1 + 0*yA[:, 0], 'k', linewidth=1)
ax.plot(yB[:, 0], yB[:, 1], -1 + 0*yB[:, 0], 'k--', linewidth=1)
ax.plot(yC[:, 0], yC[:, 1], -1 + 0*yC[:, 0], 'k', linewidth=1)

# 3 Boyutlu Uzaydaki Gerçek Koopman Yörüngeleri (Siyah Kalın Çizgiler)
ax.plot(yA[:, 0], yA[:, 1], yA[:, 3-1], 'k', linewidth=1.5)
ax.plot(yB[:, 0], yB[:, 1], yB[:, 3-1], 'k', linewidth=1.5)
ax.plot(yC[:, 0], yC[:, 1], yC[:, 3-1], 'k', linewidth=1.5)

# Orijin (Denge noktası)
ax.plot([0], [0], [0], marker='o', markersize=8, color='k')

# Görsel ayarlar
ax.set_zticks([0, 1, 2, 3, 4, 5])
ax.set_xlim([-4, 4])
ax.set_ylim([-1, 4])
ax.set_zlim([-1, 4])
ax.set_xlabel('y_1 (Hız/Açı)')
ax.set_ylabel('y_2 (İrtifa)')
ax.set_zlabel('y_3 (Gizli Koopman Değişkeni)')
ax.view_init(elev=8, azim=-15)

plt.tight_layout()
plt.show()