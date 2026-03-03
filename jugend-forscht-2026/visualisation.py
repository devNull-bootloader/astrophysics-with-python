import numpy as np
import matplotlib.pyplot as plt

# ========== Constants ==========
mu = 3.986e14         # gravitational constant × Earth's mass
R_earth = 6371e3      # Earth's radius (m)

# ========== Orbit altitudes (from 200 km to 36,000 km) ==========
h = np.linspace(200e3, 36000e3, 200)
r2 = R_earth + h
r1 = R_earth + 200e3  # reference LEO at 200 km

# ========== Hohmann transfer Δv from LEO (200 km) to each target orbit ==========
a_transfer = (r1 + r2) / 2
v_circ1 = np.sqrt(mu / r1)
v_circ2 = np.sqrt(mu / r2)
v_perigee = np.sqrt(mu * (2 / r1 - 1 / a_transfer))
v_apogee = np.sqrt(mu * (2 / r2 - 1 / a_transfer))

delta_v1 = np.abs(v_perigee - v_circ1)
delta_v2 = np.abs(v_circ2 - v_apogee)
delta_v_total = delta_v1 + delta_v2

# ========== Plot ==========
plt.figure(figsize=(8, 5))
plt.plot(h / 1000, delta_v_total / 1000, color='royalblue', linewidth=2, label='Total Δv (Hohmann)')
plt.plot(h / 1000, delta_v1 / 1000, color='green', linewidth=1.5, linestyle='--', label='Δv₁ (departure burn)')
plt.plot(h / 1000, delta_v2 / 1000, color='orange', linewidth=1.5, linestyle='--', label='Δv₂ (insertion burn)')
plt.title("Hohmann Transfer Δv from LEO (200 km) vs Target Orbit Altitude", fontsize=13)
plt.xlabel("Target orbit altitude (km)")
plt.ylabel("Δv (km/s)")
plt.legend()
plt.grid(True)
plt.show()
