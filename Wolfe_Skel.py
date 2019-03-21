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

def Wolfe(alpha, x, D, Oracle, check_direction=True):

    ##### Coefficients de la recherche lineaire

    omega_1 = 0.1
    omega_2 = 0.9

    alpha_min = 0
    alpha_max = np.inf

    ok = 0
    dltx = 0.000001

    ##### Algorithme de Fletcher-Lemarechal

    # Appel de l'oracle au point initial
    critere, gradient = Oracle(x, 4)
    scalar_product = np.vdot(gradient, D)
    if check_direction:
        assert scalar_product<0

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
        critere_n, gradient_n = Oracle(xn, 4)
        condition1 = (critere_n<=critere+omega_1*alpha_n*np.vdot(gradient,D))
        condition2 = (np.vdot(gradient_n,D)>=omega_2*np.vdot(gradient,D))
    
        if not condition1:
            alpha_max = alpha_n
            alpha_n = 0.5*(alpha_min+alpha_max)
        elif not condition2:
            alpha_min=alpha_n
            if alpha_max == np.inf:
                alpha_n=2*alpha_min
            else:
                alpha_n=0.5*(alpha_min+alpha_max)
        else:
            ok = 1

        if np.linalg.norm(xn - xp) < dltx:
            ok = 2
    return alpha_n, ok


# def Wolfe(alpha, x, D, Oracle, check_direction=True):
    
#     ##### Coefficients de la recherche lineaire

#     omega_1 = 0.1
#     omega_2 = 0.9

#     alpha_min = 0
#     alpha_max = np.inf

#     ok = 0
#     dltx = 0.0000001

#     ##### Algorithme de Fletcher-Lemarechal
    
#     # Appel de l'oracle au point initial
#     critere, gradient = Oracle(x, ind=4)
#     scalar_product = np.vdot(gradient, D)
#     criteren = critere
#     gradientn = gradient

#     if check_direction:
#         assert scalar_product<0
    
#     # Initialisation de l'algorithme
#     alpha_n = alpha
#     xn = x
    
#     # Boucle de calcul du pas
#     while ok == 0:
        
#         # xn represente le point pour la valeur courante du pas,
#         # xp represente le point pour la valeur precedente du pas.
#         xp = xn
#         xn = x + alpha_n*D
        
#         # Calcul des conditions de Wolfe
        
#         argoutn = Oracle (xn)
#         criteren=argoutn[0]
#         gradientn = argoutn[1]
       
#         # Test des conditions de Wolfe
#         condition1 = (criteren<=critere+omega_1*alpha_n*np.vdot(gradient,D))
#         condition2 = (np.vdot(gradientn,D)>=omega_2*np.vdot(gradient,D))
#         alpha_p = alpha_n
#         # - si les deux conditions de Wolfe sont verifiees,
#         if condition1 and condition2:
#             ok=1
#         # - sinon, modifier la valeur de alphan : on reboucle.
#         else:
#             if not condition1:
#                 alpha_max=alpha_n
#                 alpha_n=0.5*(alpha_min+alpha_max)
#             else :
#                 if not condition2:
#                     alpha_min=alpha_n
#                     if alpha_min == np.inf:
#                         alpha_n=2*alpha_min
#                     else:
#                         alpha_n=0.5*(alpha_min+alpha_max)
        
#         # Test d'indistinguabilite
#         if abs(alpha_n-alpha_p)*np.linalg.norm(D)<dltx:
#             ok = 2
               
#     return alpha_n, ok

# def Wolfe(alpha, x, D, Oracle, check_direction=True):

#     critere, gradient = Oracle(x)


#     omega_1 = 0.1
#     omega_2 = 0.9

#     alpha_min = 0
#     alpha_max = np.inf
#     alpha_n = alpha

#     critere, gradient = Oracle(x)

#     ok = 0
#     dltx = 0.00000001

#     while ok==0:
#         alpha_p = alpha_n

#         xn = x + alpha_n*D

#         critere_n, gradient_n = Oracle(xn)
#         condition1 = (critere_n<=critere+omega_1*alpha_n*np.vdot(gradient, D))
#         condition2 = (np.vdot(gradient_n, D) >= omega_2*np.vdot(gradient, D))
        
#         if condition1 and condition2:
#             ok=1
#         if not condition1:
#             alpha_max = alpha_n
#             alpha_n = 0.5*(alpha_max+alpha_min)
#         else:
#             if not condition2:
#                 alpha_min = alpha_n
#             else:
#                 if (alpha_max==np.inf):
#                     alpha_n = 2*alpha_min
#                     alpha_n = 0.5*(alpha_max+alpha_min)

#        
#     return alpha_n, ok
