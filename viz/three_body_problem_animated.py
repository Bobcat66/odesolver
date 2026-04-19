import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Line3DCollection

# =========================
# Load data
# =========================
df = pd.read_csv("three_body_problem.csv")

t  = df["t"].to_numpy()

x1 = df["x1"].to_numpy()
y1 = df["y1"].to_numpy()
z1 = df["z1"].to_numpy()

x2 = df["x2"].to_numpy()
y2 = df["y2"].to_numpy()
z2 = df["z2"].to_numpy()

x3 = df["x3"].to_numpy()
y3 = df["y3"].to_numpy()
z3 = df["z3"].to_numpy()

# =========================
# Figure
# =========================
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection="3d")

ax.set_xlim(min(x1.min(), x2.min(), x3.min()),
            max(x1.max(), x2.max(), x3.max()))

ax.set_ylim(min(y1.min(), y2.min(), y3.min()),
            max(y1.max(), y2.max(), y3.max()))

ax.set_zlim(min(z1.min(), z2.min(), z3.min()),
            max(z1.max(), z2.max(), z3.max()))

# =========================
# Helper for fading trail
# =========================
trail = 150   # number of previous points visible

def make_fading_line(color):
    dummy = np.array([[[0,0,0],[0,0,0]]], dtype=float)
    lc = Line3DCollection(dummy, linewidths=2)
    lc.set_color(color)
    ax.add_collection3d(lc)
    return lc

line1 = make_fading_line("red")
line2 = make_fading_line("green")
line3 = make_fading_line("blue")

point1, = ax.plot([], [], [], "o", color="red",   markersize=6)
point2, = ax.plot([], [], [], "o", color="green", markersize=6)
point3, = ax.plot([], [], [], "o", color="blue",  markersize=6)

# =========================
# Build fading segments
# =========================
def update_fade(line, x, y, z, i, color):
    start = max(0, i - trail)
    
    xs = x[start:i+1]
    ys = y[start:i+1]
    zs = z[start:i+1]

    if len(xs) < 2:
        return

    pts = np.array([xs, ys, zs]).T
    segs = np.stack([pts[:-1], pts[1:]], axis=1)

    n = len(segs)

    # alpha fades old -> new
    colors = np.zeros((n,4))
    
    if color == "red":
        colors[:,0] = 1
    elif color == "green":
        colors[:,1] = 1
    elif color == "blue":
        colors[:,2] = 1

    colors[:,3] = np.linspace(0.02, 1.0, n)

    line.set_segments(segs)
    line.set_color(colors)

# =========================
# Animation
# =========================
def update(i):

    update_fade(line1, x1, y1, z1, i, "red")
    update_fade(line2, x2, y2, z2, i, "green")
    update_fade(line3, x3, y3, z3, i, "blue")

    point1.set_data([x1[i]], [y1[i]])
    point1.set_3d_properties([z1[i]])

    point2.set_data([x2[i]], [y2[i]])
    point2.set_3d_properties([z2[i]])

    point3.set_data([x3[i]], [y3[i]])
    point3.set_3d_properties([z3[i]])

    return line1, line2, line3, point1, point2, point3

ani = FuncAnimation(
    fig,
    update,
    frames=range(0, len(t)),
    interval=20,
    blit=False
)

plt.show()

ani.save("three_body_problem.mp4", writer="ffmpeg", fps=30, dpi=200)


