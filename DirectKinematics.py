import numpy as np
import matplotlib.pyplot as plt


def matriceR(a, b, g):
    # a = radians(a)
    # b = radians(b)
    # g = radians(g)

    x = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    y = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
    z = np.array([[np.cos(g), -np.sin(g), 0], [np.sin(g), np.cos(g), 0], [0, 0, 1]])

    return x @ y @ z


def matriceT(x, y, z, a, b, g):
    T = np.concatenate((matriceR(a, b, g), np.array([[x, y, z]]).T), axis=1)
    return np.concatenate((T, np.array([[0, 0, 0, 1]])), axis=0)


def DirectKinematics(l1, l2, l3, t1, t2, t3):
    T01 = matriceT(0, 0, 0, 0, 0, t1)
    T12 = matriceT(0, 0, l1, np.pi / 2, 0, t2)
    T23 = matriceT(l2, 0, 0, 0, 0, t3)
    T34 = matriceT(l3, 0, 0, 0, 0, 0)

    T04 = T01 @ T12 @ T23 @ T34

    P = np.array([[0, 0, 0, 1]]).T
    P1 = np.round(T01 @ P, 3)
    P2 = np.round(T01 @ T12 @ P, 3)
    P3 = np.round(T01 @ T12 @ T23 @ P, 3)
    P4 = np.round(T04 @ P, 3)
    X = [0, P1[0, 0], P2[0, 0], P3[0, 0], P4[0, 0]]
    Y = [0, P1[1, 0], P2[1, 0], P3[1, 0], P4[1, 0]]
    Z = [0, P1[2, 0], P2[2, 0], P3[2, 0], P4[2, 0]]

    return [X[4], Y[4], Z[4]], X, Y, Z


l1 = 0.5
l2 = 0.5
l3 = 0.5
t1 = np.radians(11.30993247)
t2 = np.radians(-290.21064297)
t3 = np.radians(602.6128925)

thetas = [[t1, t2, t3], [np.radians(11.30993247), np.radians(-47.59775046), np.radians(117.38710749)]]

ax = plt.figure(1)
ax = plt.axes(projection='3d')
ax.set_xlim3d(0, 1)
ax.set_ylim3d(0, 1)
ax.set_zlim3d(0, 1)

j = 0
for i in thetas:
    P, X, Y, Z = DirectKinematics(l1, l2, l3, i[0], i[1], i[2])
    ax.scatter3D(X, Y, Z)
    plt.plot(X, Y, Z, label=j)
    j += 1

plt.legend()
plt.show()
