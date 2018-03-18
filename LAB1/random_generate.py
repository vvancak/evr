import numpy as np
import matplotlib.pyplot as plt


mean = [2.5, 2.5]
cov = [[2, 0], [0, 2]]
x, y = np.random.uniform(0, 5, 2000).T.reshape(2, 1000)

plt.scatter(x, y, s=1)
plt.show()
