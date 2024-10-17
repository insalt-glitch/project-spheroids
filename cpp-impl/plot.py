import numpy as np
import matplotlib.pyplot as plt

d = np.loadtxt("out").T
print(d.shape)
plt.plot(d[1], d[2])
plt.show()