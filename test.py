import numpy as np
import scipy.integrate as sci_int
import math

a = np.array([[1, 2, 5, 2, 4, 10], [5, 9, 0, 9, 3, 54], [5, 20, -4, 3, 1.5, 2]])
b = a[np.array([0, 1, 1, 2]), np.array([4, 1, 2, 0])]
print(b)
print(a[0:-1, 0:-1])

f = lambda x: np.exp(-x**2)
res = sci_int.quad(f, 0, 1)[0]
print(res)

gaussX = gaussianXFunction(5, 1.0)

print(gaussX)
