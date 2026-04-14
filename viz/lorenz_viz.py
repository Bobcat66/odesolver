# Jank
import matplotlib
matplotlib.use("QtAgg")
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("../lorenz.csv")

ax = plt.figure().add_subplot(projection='3d')
ax.plot(df['x'], df['y'], df['z'])
plt.show()