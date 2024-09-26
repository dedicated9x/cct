import numpy as np
import matplotlib.pyplot as plt
import matplotlib; matplotlib.use('TkAgg')

fig, ax = plt.subplots(1, 1)

arr =  np.zeros((100, 100))

arr[20, 70] = 1
ax.text(20, 70, ".", bbox={"facecolor": "y", "alpha": 0.9})

ax.imshow(arr)

plt.show()