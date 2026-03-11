import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sys

# ================== CONSTANTS ==================
mu = 3.986e14
r_earth = 6.371e6
g0 = 9.80665

Isp = 320
m0 = 1000

rp1 = 7.0e6
ra1 = 1.2e7

rp2 = 3.0e7
ra2 = 1.2e8

rp_boost = 6.95e6

# ================== ORBIT PARAMETERS ==================
a1 = (rp1 + ra1) / 2
e1 = (ra1 - rp1) / (ra1 + rp1)

a2 = (rp2 + ra2) / 2
e2 = (ra2 - rp2) / (ra2 + rp2)

# ================== REFERENCE VALUES ==================
a_direct = (ra1 + ra2) / 2
v1_at_ra1 = np.sqrt(mu * (2 / ra1 - 1 / a1))
v2_at_ra2 = np.sqrt(mu * (2 / ra2 - 1 / a2))
v_direct_at_ra1 = np.sqrt(mu * (2 / ra1 - 1 / a_direct))
v_direct_at_ra2 = np.sqrt(mu * (2 / ra2 - 1 / a_direct))

dv1_direct = v_direct_at_ra1 - v1_at_ra1
dv2_direct = v2_at_ra2 - v_direct_at_ra2
dv_direct = abs(dv1_direct) + abs(dv2_direct)

# ================== OBERTH-ASSISTED TRANSFER ==================
a_t1 = (ra1 + rp_boost) / 2
e_t1 = (ra1 - rp_boost) / (ra1 + rp_boost)

a_t2 = (ra2 + rp_boost) / 2
e_t2 = (ra2 - rp_boost) / (ra2 + rp_boost)

# ================== VELOCITIES ==================
v_t1_at_ra1 = np.sqrt(mu * (2 / ra1 - 1 / a_t1))
v_t1_at_rp = np.sqrt(mu * (2 / rp_boost - 1 / a_t1))
v_t2_at_rp = np.sqrt(mu * (2 / rp_boost - 1 / a_t2))
v_t2_at_ra2 = np.sqrt(mu * (2 / ra2 - 1 / a_t2))

dv1 = v_t1_at_ra1 - v1_at_ra1
dv2 = v_t2_at_rp - v_t1_at_rp
dv3 = v2_at_ra2 - v_t2_at_ra2

dv_oberth = abs(dv1) + abs(dv2) + abs(dv3)

# ================== ROCKET EQUATION ==================
dm_direct = m0 * (1 - np.exp(-dv_direct / (Isp * g0)))
dm_oberth = m0 * (1 - np.exp(-dv_oberth / (Isp * g0)))

print(f"Direkt:   Δv = {dv_direct:.1f} m/s,  Treibstoff = {dm_direct:.1f} kg")
print(f"Oberth:   Δv = {dv_oberth:.1f} m/s,  Treibstoff = {dm_oberth:.1f} kg")
print(f"Ersparnis: Δv = {dv_direct - dv_oberth:.1f} m/s,  Treibstoff = {dm_direct - dm_oberth:.1f} kg")

# ================== ORBIT PERIODS AND SAMPLING ==================
T1 = 2 * np.pi * np.sqrt(a1**3 / mu)
Tt1 = np.pi * np.sqrt(a_t1**3 / mu)
Tt2 = np.pi * np.sqrt(a_t2**3 / mu)
T2 = 2 * np.pi * np.sqrt(a2**3 / mu)

N1 = 200
Nt1 = 200
Nt2 = 300
N2 = 200

# ================== KEPLER SOLVER ==================
def kepler(M, e):
    E = M.copy()
    for _ in range(15):
        E -= (E - e * np.sin(E) - M) / (1 - e * np.cos(E))
    return E

# ================== ORBIT 1 ==================
t1 = np.linspace(0, T1, N1)
M1 = np.pi + 2 * np.pi * t1 / T1
E1 = kepler(M1, e1)
theta1 = 2 * np.arctan2(
    np.sqrt(1 + e1) * np.sin(E1 / 2),
    np.sqrt(1 - e1) * np.cos(E1 / 2)
)
r1_vals = a1 * (1 - e1 * np.cos(E1))
x1 = r1_vals * np.cos(theta1)
y1 = r1_vals * np.sin(theta1)

# ================== TRANSFER ORBIT 1 ==================
t_t1 = np.linspace(0, Tt1, Nt1)
M_t1 = np.sqrt(mu / a_t1**3) * t_t1
M_t1_shifted = np.pi + M_t1
E_t1 = kepler(M_t1_shifted, e_t1)

theta_t1 = 2 * np.arctan2(
    np.sqrt(1 + e_t1) * np.sin(E_t1 / 2),
    np.sqrt(1 - e_t1) * np.cos(E_t1 / 2)
)

r_t1_vals = a_t1 * (1 - e_t1 * np.cos(E_t1))
x_t1 = r_t1_vals * np.cos(theta_t1)
y_t1 = r_t1_vals * np.sin(theta_t1)

# ================== TRANSFER ORBIT 2 ==================
t_t2 = np.linspace(0, Tt2, Nt2)
M_t2 = np.sqrt(mu / a_t2**3) * t_t2
E_t2 = kepler(M_t2, e_t2)

theta_t2 = 2 * np.arctan2(
    np.sqrt(1 + e_t2) * np.sin(E_t2 / 2),
    np.sqrt(1 - e_t2) * np.cos(E_t2 / 2)
)

r_t2_vals = a_t2 * (1 - e_t2 * np.cos(E_t2))
x_t2 = r_t2_vals * np.cos(theta_t2)
y_t2 = r_t2_vals * np.sin(theta_t2)

# ================== ORBIT 2 ==================
t2 = np.linspace(0, T2, N2)
M2 = np.pi + 2 * np.pi * t2 / T2
E2 = kepler(M2, e2)
theta2 = 2 * np.arctan2(
    np.sqrt(1 + e2) * np.sin(E2 / 2),
    np.sqrt(1 - e2) * np.cos(E2 / 2)
)
r2_vals = a2 * (1 - e2 * np.cos(E2))
x2 = r2_vals * np.cos(theta2)
y2 = r2_vals * np.sin(theta2)

# ================== COMBINE ==================
x = np.concatenate([x1, x_t1, x_t2, x2])
y = np.concatenate([y1, y_t1, y_t2, y2])
r = np.sqrt(x**2 + y**2)

# ================== SPEED & ENERGY ==================
N_total = N1 + Nt1 + Nt2 + N2
v = np.zeros(N_total)

v[:N1] = np.sqrt(mu * (2 / r[:N1] - 1 / a1))
v[N1:N1+Nt1] = np.sqrt(mu * (2/r[N1:N1+Nt1] - 1/a_t1))
v[N1+Nt1:N1+Nt1+Nt2] = np.sqrt(mu * (2/r[N1+Nt1:N1+Nt1+Nt2] - 1/a_t2))
v[N1+Nt1+Nt2:] = np.sqrt(mu * (2 / r[N1+Nt1+Nt2:] - 1 / a2))

energy = 0.5 * v**2 - mu / r

# ================== FIGURE ==================
fig = plt.figure(figsize=(14, 8))
gs = fig.add_gridspec(2, 2)

ax_orbit = fig.add_subplot(gs[:, 0])
ax_speed = fig.add_subplot(gs[0, 1])
ax_energy = fig.add_subplot(gs[1, 1])

# ================== ORBIT VIEW ==================
ax_orbit.set_aspect('equal')
lim = ra2 * 1.1
ax_orbit.set_xlim(-lim, lim)
ax_orbit.set_ylim(-lim, lim)
ax_orbit.set_title("Oberth-Transfer (elliptisch)")

ax_orbit.add_patch(plt.Circle((0, 0), r_earth, color='skyblue', zorder=0))
ax_orbit.plot(x1, y1, '--', color='gray')
ax_orbit.plot(x_t1, y_t1, '--', color='orange')
ax_orbit.plot(x_t2, y_t2, '--', color='orange')
ax_orbit.plot(x2, y2, '--', color='gray')

corner_text = (
    f"Δv Direkt: {dv_direct:.0f} m/s\n"
    f"Δv Oberth: {dv_oberth:.0f} m/s\n"
    f"Vorteil: {dv_direct - dv_oberth:.0f} m/s\n"
    f"Treibstoff: {dm_direct - dm_oberth:.1f} kg"
)
ax_orbit.text(
    0.03,
    0.97,
    corner_text,
    transform=ax_orbit.transAxes,
    fontsize=8,
    va='top',
    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
)

sat, = ax_orbit.plot([], [], 'ro', markersize=6)

# ================== SPEED GRAPH ==================
ax_speed.set_title("Geschwindigkeit vs Zeit")
ax_speed.set_ylabel("m/s")
ax_speed.set_xlim(0, N_total)
ax_speed.set_ylim(v.min() * 0.95, v.max() * 1.05)

line_v, = ax_speed.plot([], [], 'r')

# ================== ENERGY GRAPH ==================
ax_energy.set_title("Spezifische Energie")
ax_energy.set_xlabel("Zeitschritt")
ax_energy.set_ylabel("J/kg")
ax_energy.set_xlim(0, N_total)
ax_energy.set_ylim(energy.min() * 1.05, energy.max() * 0.95)

line_e, = ax_energy.plot([], [], 'b')

# ================== ANIMATION ==================
def update(i):
    sat.set_data([x[i]], [y[i]])
    line_v.set_data(np.arange(i), v[:i])
    line_e.set_data(np.arange(i), energy[:i])
    return sat, line_v, line_e

plt.tight_layout()

is_streamlit = "streamlit" in sys.modules

if is_streamlit:
    import streamlit as st

    update(N_total - 1)
    st.pyplot(fig, width='stretch')
    plt.close(fig)
elif plt.get_backend().lower().endswith('agg'):
    update(N_total - 1)
    plt.savefig("oberth-transfer.png", dpi=150)
    plt.close(fig)
else:
    manager = plt.get_current_fig_manager()
    try:
        manager.full_screen_toggle()
    except Exception:
        pass

    anim = FuncAnimation(
        fig,
        update,
        frames=N_total,
        interval=20,
        blit=False
    )
    plt.show()