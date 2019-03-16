#!/usr/bin/python

import numpy as np
from numpy.linalg import norm
from numpy.linalg import inv
from time import process_time
from Wolfe_Skel import Wolfe

#############################################################################
#                                                                           #
#         RESOLUTION D'UN PROBLEME D'OPTIMISATION SANS CONTRAINTES          #
#                                                                           #
#         Methode du gradient a pas variable                                #
#                                                                           #
#############################################################################

from Visualg import Visualg

def Newton(Oracle, x0):

    ##### Initialisation des variables

    iter_max = 10000
    gradient_step = 0.0005
    threshold = 0.000001

    gradient_norm_list = []
    gradient_step_list = []
    critere_list = []

    time_start = process_time()

    x = x0
    alpha_n = 1


    ##### Boucle sur les iterations
    for k in range(iter_max):

        # Valeur du critere et du gradient
        critere, gradient, hess = Oracle(x, 7)
        gradient_norm = norm(gradient)

        # Direction de descente
        D = np.dot(-inv(hess), gradient)
        delta_k=1*(critere+4)
        alpha_0=-2*delta_k/np.vdot(gradient, D)
        alpha_p = alpha_n
        alpha_n, ok = Wolfe(alpha_0, x, D, Oracle, False)

        # print("alpha", alpha_n)
        # print("ok", ok)

        # Mise a jour des variables
        x = x + alpha_n*D

        # Evolution du gradient, du pas, et du critere
        gradient_norm_list.append(gradient_norm)
        gradient_step_list.append(alpha_n)
        critere_list.append(critere)

        # Condition d'arret
        # if gradient_norm<=threshold:
        if abs(alpha_n-alpha_p)*norm(D) <= threshold:
            break

    critere_opt = critere
    gradient_opt = gradient
    x_opt = x
    time_cpu = process_time() - time_start

    print()
    print('Newton')
    print('Iteration :', k)
    print('Temps CPU :', time_cpu)
    print('Critere optimal :', critere_opt)
    print('Norme du gradient :', norm(gradient_opt))

    # Visualisation de la convergence
    Visualg(gradient_norm_list, gradient_step_list, critere_list)

    return critere_opt, gradient_opt, x_opt
