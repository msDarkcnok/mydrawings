
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score


# 1. Generarea datelor sintetice
rng = np.random.default_rng(7)

N = 3000

S = rng.uniform(200, 1000, N)  # Radiatie
T = rng.uniform(0, 50, N)      # Temperatura

Smax, Topt, alpha = 1000, 25.0, 0.01

E = (S / Smax) * np.exp(-alpha * (T - Topt)**2)

y = (E > 0.5).astype(int)

X = np.c_[S, T]


# 2. Impartirea datelor si scalarea
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

X_tr, X_val, y_tr, y_val = train_test_split(
    X_tr, y_tr,
    test_size=0.2,
    random_state=42,
    stratify=y_tr
)

scaler = StandardScaler().fit(X_tr)

X_trs = scaler.transform(X_tr)
X_vals = scaler.transform(X_val)
X_tes = scaler.transform(X_te)


# 3. Construirea retelei neuronale MLP 2-3-2-1
clf = MLPClassifier(
    hidden_layer_sizes=(3, 2),
    activation='tanh',
    solver='lbfgs',
    max_iter=2000,
    random_state=42
)

clf.fit(X_trs, y_tr)


# 4. Evaluarea modelului
for split_name, Xs, ys in [
    ('Train', X_trs, y_tr),
    ('Val', X_vals, y_val),
    ('Test', X_tes, y_te)
]:
    yhat = clf.predict(Xs)

    print(
        f'{split_name} acc: '
        f'{accuracy_score(ys, yhat)*100:.2f}%'
    )


# 5. Frontiera de decizie in spatiul (S, T)
smin, smax = 150, 1050
tmin, tmax = -5, 55

xx, yy = np.meshgrid(
    np.linspace(smin, smax, 400),
    np.linspace(tmin, tmax, 400)
)

grid = np.c_[xx.ravel(), yy.ravel()]

grid_s = scaler.transform(grid)

proba = clf.predict_proba(grid_s)[:, 1]

zz = proba.reshape(xx.shape)


# 6. Reprezentarea grafica
plt.figure()

plt.contourf(
    xx, yy, zz,
    levels=[0, 0.5, 1],
    alpha=0.35
)

cs = plt.contour(
    xx, yy, zz,
    levels=[0.5],
    colors='k',
    linewidths=2
)

plt.clabel(cs, fmt={'0.5': 'p=0.5'})


# Puncte de test colorate dupa clasele reale
mask0 = (y_te == 0)
mask1 = (y_te == 1)

plt.scatter(
    X_te[mask0, 0],
    X_te[mask0, 1],
    s=10,
    label='Ineficient',
    alpha=0.8
)

plt.scatter(
    X_te[mask1, 0],
    X_te[mask1, 1],
    s=10,
    label='Eficient',
    alpha=0.8
)

plt.title('MLP 2-3-2-1 - frontiera de decizie')

plt.xlabel('Radiatie S [W/m²]')
plt.ylabel('Temperatura T [°C]')

plt.legend(loc='best')
plt.grid(True)
plt.show()
