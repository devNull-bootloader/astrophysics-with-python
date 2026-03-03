import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
from matplotlib.widgets import Button

# --- Physics constants ---
G = 6.67430e-11  # gravitational constant
dt = 60 * 60 * 24  # 1 day per frame
frame_count = 0
sun_gravity_on = True  # Sun gravity toggle
paused = False  # Pause toggle
elapsed_days = 0  # Simulation time tracker

# --- Planet data ---
planets = [
    {"name": "Mercury", "pos": [0.39 * 1.496e11, 0], "vel": [0, 47360],
     "mass": 3.285e23, "mass_on": True, "x_path": [], "y_path": [], "color": "gray"},
    {"name": "Venus", "pos": [0.72 * 1.496e11, 0], "vel": [0, 35020],
     "mass": 4.867e24, "mass_on": True, "x_path": [], "y_path": [], "color": "orange"},
    {"name": "Earth", "pos": [1.496e11, 0], "vel": [0, 29780],
     "mass": 5.972e24, "mass_on": True, "x_path": [], "y_path": [], "color": "blue"},
    {"name": "Mars", "pos": [1.52 * 1.496e11, 0], "vel": [0, 24077],
     "mass": 6.417e23, "mass_on": True, "x_path": [], "y_path": [], "color": "red"},
    {"name": "Jupiter", "pos": [5.2 * 1.496e11, 0], "vel": [0, 13070],
     "mass": 1.898e27, "mass_on": True, "x_path": [], "y_path": [], "color": "brown"},
    {"name": "Saturn", "pos": [9.537 * 1.496e11, 0], "vel": [0, 9690],
     "mass": 5.683e26, "mass_on": True, "x_path": [], "y_path": [], "color": "goldenrod"}
]

# --- Store initial positions/velocities for reset ---
initial_states = []
for p in planets:
    initial_states.append({"pos": p["pos"][:], "vel": p["vel"][:]})

# --- Setup figure ---
fig, ax = plt.subplots()
fig.patch.set_facecolor('black')
ax.set_facecolor("black")
ax.set_aspect('equal', adjustable='box')

# --- Physics update ---
def update_positions():
    for i, p1 in enumerate(planets):
        fx = fy = 0
        for j, p2 in enumerate(planets):
            if i != j and p2["mass_on"]:
                dx = p2["pos"][0] - p1["pos"][0]
                dy = p2["pos"][1] - p1["pos"][1]
                dist = math.sqrt(dx**2 + dy**2)
                F = G * p1["mass"] * p2["mass"] / dist**2
                fx += F * dx / dist
                fy += F * dy / dist

        # Sun gravity (toggleable)
        if sun_gravity_on:
            dx = 0 - p1["pos"][0]
            dy = 0 - p1["pos"][1]
            dist = math.sqrt(dx**2 + dy**2)
            F = G * p1["mass"] * 1.989e30 / dist**2
            fx += F * dx / dist
            fy += F * dy / dist

        # Update velocity & position
        p1["vel"][0] += fx / p1["mass"] * dt
        p1["vel"][1] += fy / p1["mass"] * dt
        p1["pos"][0] += p1["vel"][0] * dt
        p1["pos"][1] += p1["vel"][1] * dt

        # Save path
        p1["x_path"].append(p1["pos"][0])
        p1["y_path"].append(p1["pos"][1])

# --- Animation ---
def animate(frame):
    global frame_count, elapsed_days
    if paused:
        return
    frame_count += 1
    elapsed_days += dt / (60 * 60 * 24)
    ax.clear()
    ax.set_facecolor("black")
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')

    update_positions()

    # Automatic scaling
    max_distance = max(math.sqrt(p["pos"][0]**2 + p["pos"][1]**2) for p in planets)
    margin = 0.3 * max_distance
    ax.set_xlim(-max_distance - margin, max_distance + margin)
    ax.set_ylim(-max_distance - margin, max_distance + margin)

    # Sun core & subtle halo
    glow_alpha = 0.2 + 0.1 * (math.sin(frame_count * 0.05) + 1) / 2
    ax.scatter(0, 0, color='yellow', s=200, zorder=3)      # Sun core
    ax.scatter(0, 0, color='yellow', s=100, alpha=glow_alpha, zorder=1)  # subtle halo

    # Planets
    for p in planets:
        # Orbit path
        ax.plot(p["x_path"], p["y_path"], color=p["color"], lw=1)
        # Planet core
        ax.scatter(p["pos"][0], p["pos"][1], color=p["color"], s=30, zorder=4)
        # Planet glow halo
        planet_alpha = 0.1 + 0.05 * (math.sin(frame_count * 0.1) + 1) / 2
        ax.scatter(p["pos"][0], p["pos"][1], color=p["color"], s=150, alpha=planet_alpha, zorder=2)
        # Planet name label
        ax.text(p["pos"][0], p["pos"][1], f" {p['name']}", color=p["color"],
                fontsize=6, va='center', zorder=5)

    # Elapsed time display
    years = elapsed_days / 365.25
    ax.text(0.02, 0.97, f"Time: {years:.2f} years", transform=ax.transAxes,
            color='white', fontsize=9, va='top')

# --- Button callbacks ---
def reset(event):
    global elapsed_days
    for i, p in enumerate(planets):
        p["pos"] = initial_states[i]["pos"][:]
        p["vel"] = initial_states[i]["vel"][:]
        p["x_path"].clear()
        p["y_path"].clear()
    elapsed_days = 0

def toggle_mass(planet_index):
    def inner(event):
        planets[planet_index]["mass_on"] = not planets[planet_index]["mass_on"]
        status = "ON" if planets[planet_index]["mass_on"] else "OFF"
        print(f"{planets[planet_index]['name']} mass toggled {status}")
    return inner

def toggle_sun_gravity(event):
    global sun_gravity_on
    sun_gravity_on = not sun_gravity_on
    status = "ON" if sun_gravity_on else "OFF"
    print(f"Sun gravity toggled {status}")

def toggle_pause(event):
    global paused
    paused = not paused
    status = "PAUSED" if paused else "RUNNING"
    print(f"Simulation {status}")

# --- Add buttons ---
ax_reset = plt.axes([0.81, 0.05, 0.1, 0.05])
btn_reset = Button(ax_reset, 'Reset')
btn_reset.on_clicked(reset)

ax_pause = plt.axes([0.81, 0.12, 0.1, 0.05])
btn_pause = Button(ax_pause, 'Pause/Play')
btn_pause.on_clicked(toggle_pause)

# Buttons for each planet
for i, p in enumerate(planets):
    ax_btn = plt.axes([0.01, 0.05 + i*0.06, 0.1, 0.05])
    btn = Button(ax_btn, p["name"])
    btn.on_clicked(toggle_mass(i))

# Sun gravity toggle button
ax_sun = plt.axes([0.81, 0.19, 0.1, 0.05])
btn_sun = Button(ax_sun, "Sun Gravity")
btn_sun.on_clicked(toggle_sun_gravity)

# --- Run animation ---
ani = animation.FuncAnimation(fig, animate, frames=360, interval=20)
plt.show()
