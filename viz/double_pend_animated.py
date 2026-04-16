import pandas as pd

df = pd.read_csv("double_pendulum.csv")

t = df["t"].to_numpy()
x1 = df["x1"].to_numpy()
y1 = df["y1"].to_numpy()
x2 = df["x2"].to_numpy()
y2 = df["y2"].to_numpy()
speed = df["speed"].to_numpy()


import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import LineCollection
import numpy as np
fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.set_xlim(x2.min() - 0.5, x2.max() + 0.5)
ax.set_ylim(y2.min() - 0.5, y2.max() + 0.5)

line, = ax.plot([], [], 'o-', lw=2)
trail = LineCollection([], linewidths=2)
ax.add_collection(trail)

x2_hist, y2_hist = [], []

max_trail = 200  # length of trail



def update(i):
    # pendulum rods
    line.set_data([0, x1[i], x2[i]], [0, y1[i], y2[i]])

    # build trail window
    start = max(0, i - max_trail)
    
    points = np.array([x2[start:i], y2[start:i]]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # color by velocity
    seg_speed = speed[start:i-1]

    # normalize colors
    norm = plt.Normalize(speed.min(), speed.max())
    colors = plt.cm.plasma(norm(seg_speed))

    # fade older segments
    t = np.linspace(0, 1, len(colors))
    alphas = np.exp(-5 * (1 - t))  # 5 = fade strength
    colors[:, -1] = alphas

    trail.set_segments(segments)
    trail.set_color(colors)

    return line, trail

ani = FuncAnimation(
    fig,
    update,
    frames=len(x1),
    interval=20,
    blit=True
)

plt.show()

ani.save("double_pendulum.mp4", writer="ffmpeg", fps=30, dpi=200)