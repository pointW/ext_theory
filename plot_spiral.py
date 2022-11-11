import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

fig, ax = plt.subplots(figsize=(5, 5))
r = np.arange(0, .09, 0.00001)
nverts = len(r)
theta = np.array(range(nverts)) * (2*np.pi)/(nverts-1)
theta = 90*np.pi*r
xoffset, yoffset = 0, 0
yy = 10*r * np.sin(theta) + yoffset
xx = 10*r * np.cos(theta) + xoffset
spiral = zip(xx,yy)
collection = LineCollection([list(spiral)], colors='k')
ax.add_collection(collection)

r = np.arange(0., .09, 0.00001)
nverts = len(r)
theta = np.array(range(nverts)) * (2*np.pi)/(nverts-1)
theta = 90*np.pi*r + 3
xoffset, yoffset = 0, 0
yy = 10*r * np.sin(theta) + yoffset
xx = 10*r * np.cos(theta) + xoffset
spiral = zip(xx,yy)
collection = LineCollection([list(spiral)], colors='r')
ax.add_collection(collection)

ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
# In polar coordinates
r = np.arange(0, 3.4, 0.01)
theta = 2*np.pi*r
# ax.plot(theta, r, linewidth=1, color='k')
plt.show()