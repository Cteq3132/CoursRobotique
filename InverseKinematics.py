# Dans ce programme sont proposées 3 solutions permettant de résoudre un problème de cinématique inverse pour un
# robot à 3 degrés de libertés.

import numpy as np


# Définition des fonctions permettant de résoudre les différents types d'équation

def resolution_type_2(x, y, z):
    """
    Fonction permettant de résoudre une équation de la forme
    x.sin(t)+y.cos(t)=z
    Retourne les valeurs des angles en radians.
    """
    if x == 0 and y != 0:
        t1 = np.arctan2(np.sqrt(1 - (z / y) ** 2), z / y)
        t2 = np.arctan2(-np.sqrt(1 - (z / y) ** 2), z / y)
    elif x != 0 and y == 0:
        t1 = np.arctan2(z / x, np.sqrt(1 - (z / x) ** 2))
        t2 = np.arctan2(z / x, -np.sqrt(1 - (z / x) ** 2))
    elif x != 0 and y != 0 and z == 0:
        t1 = np.arctan2(-y, x)
        t2 = t1 + np.pi
    else:
        t1 = np.arctan2((x * z + y * np.sqrt(x ** 2 + y ** 2 - z ** 2)) / (x ** 2 + y ** 2),
                        (y * z + x * np.sqrt(x ** 2 + y ** 2 - z ** 2)) / (x ** 2 + y ** 2))
        t2 = np.arctan2((x * z - y * np.sqrt(x ** 2 + y ** 2 - z ** 2)) / (x ** 2 + y ** 2),
                        (y * z - x * np.sqrt(x ** 2 + y ** 2 - z ** 2)) / (x ** 2 + y ** 2))
    return t1, t2


def resolution_type_3(x1, y1, z1, x2, y2, z2):
    """
    Fonction permettant de résoudre un système d'équation de la forme
    x1.sin(t)+y1.cos(t)=z1
    x2.sin(t)+y2.cos(t)=z2
    Retourne les valeurs des angles en radians.
    """
    if x1 * y1 - x2 * y2 != 0:
        t = np.arctan2((z1 * y2 - z2 * y1) / (x1 * y2 - x2 * y1), (z2 * x1 - z1 * x2) / (x1 * y2 - x2 * y1))
    else:
        t = np.arctan2(z1 / x1, z2 / y2)
    return t


def resolution_type_6(w, x, y, z1, z2):
    """
    Fonction permettant de résoudre un système d'équation de la forme
    w.sin(t2) = x.cos(t1)+y.sin(t1)+z1
    w.cos(t2) = x.sin(t1)-y.cos(t1)+z2
    Retourne t1, t1', t2, t2' en radians.
    """
    # Résolution par un système de type 2 en t1
    b1 = 2 * (z1 * y + z2 * x)
    b2 = 2 * (z1 * x - z2 * y)
    b3 = w ** 2 - x ** 2 - y ** 2 - z1 ** 2 - z2 ** 2
    t1, t2 = resolution_type_2(b1, b2, b3)

    # Résolution par un système de type 3 en t2
    t1_1 = resolution_type_3(w, 0, x * np.cos(t1) + y * np.sin(t1) + z1, 0, w, x * np.sin(t1) - y * np.cos(t1) + z2)
    t1_2 = resolution_type_3(w, 0, x * np.cos(t2) + y * np.sin(t2) + z1, 0, w, x * np.sin(t2) - y * np.cos(t2) + z2)

    return t1, t2, t1_1, t1_2


def inverse_kinematics_analytic(l1, l2, l3, X):
    """
    :param l1, l2, l3: paramètre du robot
    :param X: vecteur représentant le point d'arrivé = [x, y, z]
    :return: les 4 couples [t1, t2, t3] permettant d'atteindre X
    """
    x, y, z = X[0], X[1], X[2]

    # Calcul des theta1
    t1_1, t1_2 = resolution_type_2(x, -y, 0)

    # Calcul des theta2 et theta3 pour theta1 = t1_1
    t2_1, t2_2, t3_1, t3_2 = resolution_type_6(l3, z-l1, -(x*np.cos(t1_1)+y*np.sin(t1_1)), 0, -l2)

    # Calcul des theta2 et theta3 pour theta1 = t1_2
    t2_3, t2_4, t3_3, t3_4 = resolution_type_6(l3, z - l1, -(x * np.cos(t1_2) + y * np.sin(t1_2)), 0, -l2)

    return [[t1_1, t2_1, t3_1], [t1_1, t2_2, t3_2], [t1_2, t2_3, t3_3], [t1_2, t2_4, t3_4]]
