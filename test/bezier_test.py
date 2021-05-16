import bezier
import numpy as np
import matplotlib.pyplot as plt

nodes1 = np.asfortranarray([
    [0.0, 1.0, 2.0],
    [0.0, 2.0, 0.5]
])

curve1 = bezier.Curve(nodes1, degree=2)

RES = 64.
RES_I = int(RES)

xs = np.zeros((RES_I, 1))
ys = np.zeros((RES_I, 1))

for i in range(RES_I):
    x, y = curve1.evaluate(i / RES)
    xs[i] = x
    ys[i] = y

print(curve1.evaluate(0.), curve1.evaluate(1.), curve1.evaluate(2.))
plt.plot(xs, ys)
plt.show()