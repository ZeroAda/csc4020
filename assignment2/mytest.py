import numpy as np
import sys
import matplotlib.pyplot as plt

a = np.array([[1, 2, 3], [2, 3, 4]])
b = np.array([[1, 2, 3]])
# elementwise multiplication
r1 = a * b
# ele
r3 = np.multiply(a, b)
# matrix multi
r2 = a.dot(b.T)


# print(r1,r2,r3)

class A:
    def __init__(self, a):
        self.a = a


q = A(10)
print(q.a)


q = np.array([[1,0,0]])
p = np.array([[5.78463502e-12,9.93881710e-01,6.11828978e-03]])
print(-1 * (q.dot(np.log(p.T))))
print(-1 * np.log(5.78463502e-12))
print(np.argmax(q))
print(np.argmax(p))

