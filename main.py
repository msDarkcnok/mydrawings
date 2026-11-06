import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Vectorul de intrare
x = np.array([[205.0, 17.0]])

# Conditii de minim si maxim pentru fiecare variabila
xmin = np.array([175.0, 12.0])
xmax = np.array([260.0, 18.0])

# Verificare conditii
if np.all((x >= xmin) & (x <= xmax)):
    print("Valorile de intrare sunt valide.")
else:
    raise ValueError(f"Valorile trebuie să fie în intervalele: x1∈[{xmin[0]}, {xmax[0]}], x2∈[{xmin[1]}, {xmax[1]}]")

# Modelul secvențial
model = Sequential([
    Dense(units=3, activation='sigmoid', input_shape=(2,)),  # Strat 1
    Dense(units=1, activation='sigmoid')                     # Strat 2
])

# Propagare înainte
a2 = model(x)

# Rezultat
print("Outputul rețelei:", a2.numpy())

# === Decizie binară ===
yh = (a2.numpy() >= 0.5).astype(int)
print("Decizia finală (ŷ):", yh)