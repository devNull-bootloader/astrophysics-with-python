# Der Code ist in Englisch, da ich eher Englisch spreche.

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

r1 = 7e6
r2 = 4.2e7

# ================== HOHMANN TRANSFER ==================
a_t = (r1 + r2) / 2
e_t = (r2 - r1) / (r1 + r2)

T1 = 2*np.pi*np.sqrt(r1**3 / mu)
Tt = np.pi*np.sqrt(a_t**3 / mu)
T2 = 2*np.pi*np.sqrt(r2**3 / mu)

N1, Nt, N2 = 200, 300, 200

# ================== ORBIT 1 ==================
t1 = np.linspace(0, T1, N1)
theta1 = 2*np.pi*t1/T1
x1 = r1*np.cos(theta1)
y1 = r1*np.sin(theta1)

# ================== TRANSFER ORBIT ==================
def kepler(M, e):
    E = M.copy()
    for _ in range(10):
        E -= (E - e*np.sin(E) - M) / (1 - e*np.cos(E))
    return E

t_t = np.linspace(0, Tt, Nt)
M = np.sqrt(mu/a_t**3) * t_t
E = kepler(M, e_t)

theta_t = 2*np.arctan2(
    np.sqrt(1+e_t)*np.sin(E/2),
    np.sqrt(1-e_t)*np.cos(E/2)
)

r_t = a_t * (1 - e_t*np.cos(E))
x_t = r_t*np.cos(theta_t)
y_t = r_t*np.sin(theta_t)

# ================== ORBIT 2 ==================
T2 = 2*T2
t2 = np.linspace(0, T2, N2)
theta2 = theta_t[-1] + 2*np.pi*t2/T2
x2 = r2*np.cos(theta2)
y2 = r2*np.sin(theta2)

# ================== COMBINE ==================
x = np.concatenate([x1, x_t, x2])
y = np.concatenate([y1, y_t, y2])
r = np.sqrt(x**2 + y**2)

# ================== SPEED & ENERGY ==================
v = np.zeros_like(r)

v[:N1] = np.sqrt(mu / r1)
v[N1:N1+Nt] = np.sqrt(mu * (2/r[N1:N1+Nt] - 1/a_t))
v[N1+Nt:] = np.sqrt(mu / r2)

energy = 0.5*v**2 - mu/r

# ================== FIGURE ==================
fig = plt.figure(figsize=(11, 6))
gs = fig.add_gridspec(2, 2)

ax_orbit = fig.add_subplot(gs[:, 0])
ax_speed = fig.add_subplot(gs[0, 1])
ax_energy = fig.add_subplot(gs[1, 1])

# ================== ORBIT VIEW ==================
ax_orbit.set_aspect('equal')
lim = r2 * 1.15
ax_orbit.set_xlim(-lim, lim)
ax_orbit.set_ylim(-lim, lim)
ax_orbit.set_title("Hohmann-Transfer")

ax_orbit.add_patch(plt.Circle((0,0), r_earth, color='skyblue', zorder=0))
ax_orbit.plot(x1, y1, '--', color='gray')
ax_orbit.plot(x_t, y_t, '--', color='orange')
ax_orbit.plot(x2, y2, '--', color='gray')

sat, = ax_orbit.plot([], [], 'ro', markersize=6)

# ================== SPEED GRAPH ==================
ax_speed.set_title("Geschwindigkeit vs Zeit")
ax_speed.set_ylabel("m/s")
ax_speed.set_xlim(0, len(x))
ax_speed.set_ylim(v.min()*0.95, v.max()*1.05)
line_v, = ax_speed.plot([], [], 'r')

# ================== ENERGY GRAPH ==================
ax_energy.set_title("Spezifische Energie")
ax_energy.set_xlabel("Zeitschritt")
ax_energy.set_ylabel("J/kg")
ax_energy.set_xlim(0, len(x))
ax_energy.set_ylim(energy.min()*1.05, energy.max()*0.95)
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

    update(len(x) - 1)
    st.pyplot(fig)
    plt.close(fig)
elif plt.get_backend().lower().endswith('agg'):
    update(len(x) - 1)
    plt.savefig("hohmann-transfer.png", dpi=150)
    plt.close(fig)
else:
    anim = FuncAnimation(
        fig,
        update,
        frames=len(x),
        interval=20,
        blit=False
    )
    plt.show()
