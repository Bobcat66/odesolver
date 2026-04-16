import pandas as pd

df = pd.read_csv("lorenz3.csv")

t = df["t"].to_numpy()
x = df["x"].to_numpy()
y = df["y"].to_numpy()
z = df["z"].to_numpy()

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

line, = ax.plot([], [], [], lw=1)

ax.set_xlim(x.min(), x.max())
ax.set_ylim(y.min(), y.max())
ax.set_zlim(z.min(), z.max())


point, = ax.plot([], [], [], 'ro')
def update(i):
    line.set_data(x[:i], y[:i])
    line.set_3d_properties(z[:i])

    point.set_data([x[i]], [y[i]])
    point.set_3d_properties([z[i]])

    return line, point

step = 10
ani = FuncAnimation(fig, update, frames=range(0, len(t), step), interval=1, blit=False)

plt.show()

ani.save("lorenz.mp4", writer="ffmpeg", fps=30, dpi=200)