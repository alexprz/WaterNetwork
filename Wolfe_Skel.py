#!/usr/bin/python

import numpy as np
from numpy import dot

########################################################################
#                                                                      #
#          RECHERCHE LINEAIRE SUIVANT LES CONDITIONS DE WOLFE          #
#                                                                      #
#          Algorithme de Fletcher-Lemarechal                           #
#                                                                      #
########################################################################

#  Arguments en entree
#
#    alpha  : valeur initiale du pas
#    x      : valeur initiale des variables
#    D      : direction de descente
#    Oracle : nom de la fonction Oracle
#
#  Arguments en sortie
#
#    alphan : valeur du pas apres recherche lineaire
#    ok     : indicateur de reussite de la recherche
#             = 1 : conditions de Wolfe verifiees
#             = 2 : indistinguabilite des iteres

def Wolfe(alpha, x, D, Oracle):

    ##### Coefficients de la recherche lineaire

    omega_1 = 0.1
    omega_2 = 0.9

    alpha_min = 0
    alpha_max = np.inf

    ok = 0
    dltx = 0.00000001

    ##### Algorithme de Fletcher-Lemarechal

    # Appel de l'oracle au point initial
    argout = Oracle(x, 4)
    critere = argout[0]
    gradient = argout[1]

    # Initialisation de l'algorithme
    alpha_n = alpha
    xn = x

    # Boucle de calcul du pas
    while ok == 0:

        # xn represente le point pour la valeur courante du pas,
        # xp represente le point pour la valeur precedente du pas.
        xp = xn
        xn = x + alpha_n*D

        # Calcul des conditions de Wolfe
        #
        # ---> A completer...
        # ---> A completer...
        if Oracle(xn, 2) > Oracle(x, 2) + omega_1*alpha_n*dot(Oracle(xp, 3).T, D):
            alpha_max = alpha_n
            alpha_n = (alpha_min + alpha_max)/2.

        elif dot(Oracle(xn, 3).T, D) < omega_2*dot(Oracle(xp, 3).T, D):
            alpha_min = alpha
            if alpha_max == np.inf:
                alpha_n = 2*alpha_min
            else:
                alpha_n = (alpha_min + alpha_max)/2.

        else:
            ok = 1

        # Test des conditions de Wolfe
        # - si les deux conditions de Wolfe sont verifiees,
        #   faire ok = 1 : on sort alors de la boucle while
        # - sinon, modifier la valeur de alphan : on reboucle.
        #
        # ---> A completer...
        # ---> A completer...

        # Test d'indistinguabilite
        print("xn, xp = {}, {}".format(xn, xp))
        print("alpha_n = {}".format(alpha_n))
        if np.linalg.norm(xn - xp) < dltx:
            ok = 2

    return alpha_n, ok
