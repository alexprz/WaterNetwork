#!/usr/bin/python

import numpy as np
from numpy.linalg import norm
from time import process_time
from Wolfe_Skel import Wolfe

#############################################################################
#                                                                           #
#         RESOLUTION D'UN PROBLEME D'OPTIMISATION SANS CONTRAINTES          #
#                                                                           #
#         Methode de Polak-Ribiere                                          #
#                                                                           #
#############################################################################

from Visualg import Visualg

def Beta(gradient_n, gradient_p):
    gradient_n = np.reshape(gradient_n, (len(gradient_n),1))
    gradient_p = np.reshape(gradient_p, (len(gradient_p),1))
    return (np.dot(gradient_n.T, gradient_n) - np.dot(gradient_n.T, gradient_p))[0][0]/(norm(gradient_p)**2)

def Polak_Ribiere(Oracle, x0):

    ##### Initialisation des variables

    iter_max = 10000
    threshold = 0.000001

    gradient_norm_list = []
    gradient_step_list = []
    critere_list = []

    time_start = process_time()

    x = x0

    ##### Boucle sur les iterations

    critere_n, gradient_n = Oracle(x, 4)
    D = -gradient_n
    alpha_n = 1

    for k in range(iter_max):

        delta_k=0.001*(critere_n+4)
        alpha_0=7#1#-2*delta_k/(np.vdot(gradient_n, D))

        alpha_p = alpha_n
        alpha_n, ok = Wolfe(alpha_0, x, D, Oracle, check_direction=False)
        #
        # print("alpha", alpha_n)
        # print("ok", ok)

        # Mise a jour des variables
        x = x + alpha_n*D

        gradient_p = gradient_n
        critere_n, gradient_n = Oracle(x, 4)

        if np.linalg.norm(gradient_n) <= threshold:
            break

        D = -gradient_n + Beta(gradient_n, gradient_p)*D

        # Evolution du gradient, du pas, et du critere
        gradient_norm_list.append(norm(gradient_n))
        gradient_step_list.append(alpha_n)
        critere_list.append(critere_n)


    critere_opt = critere_n
    gradient_opt = gradient_n
    x_opt = x
    time_cpu = process_time() - time_start

    print()
    print('Polak Ribiere')
    print('Iteration :', k)
    print('Temps CPU :', time_cpu)
    print('Critere optimal :', critere_opt)
    print('Norme du gradient :', norm(gradient_opt))

    # Visualisation de la convergence
    Visualg(gradient_norm_list, gradient_step_list, critere_list)

    return critere_opt, gradient_opt, x_opt
