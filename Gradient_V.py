#!/usr/bin/python

import numpy as np
from numpy.linalg import norm
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

def Gradient_V(Oracle, x0, dual=True):

    ##### Initialisation des variables

    iter_max = 10000
    threshold = 0.000001

    gradient_norm_list = []
    gradient_step_list = []
    critere_list = []

    time_start = process_time()
    x_n = x0


    ##### Boucle sur les iterations
    for k in range(iter_max):

        # Valeur du critere et du gradient
        critere, gradient = Oracle(x_n, 4)
        gradient_norm = norm(gradient)
        x_p = x_n

        # Direction de descente
        D = -gradient
        delta_k = 1*(critere+4)

        if dual:
            alpha_0 = 3#1#-2*delta_k/np.vdot(gradient, D)
        else:
            alpha_0 = 1

        alpha_n, ok = Wolfe(alpha_0, x_n, D, Oracle)

        # Mise a jour des variables
        x_n = x_p + alpha_n*D

        # Evolution du gradient, du pas, et du critere
        gradient_norm_list.append(gradient_norm)
        gradient_step_list.append(alpha_n)
        critere_list.append(critere)

        # Condition d'arret
        if gradient_norm <= threshold:
            break

    critere_opt = critere
    gradient_opt = gradient
    x_opt = x_n
    time_cpu = process_time() - time_start

    print()
    print('Pas variable')
    print('Iteration :', k)
    print('Temps CPU :', time_cpu)
    print('Critere optimal :', critere_opt)
    print('Norme du gradient :', norm(gradient_opt))

    # Visualisation de la convergence
    Visualg(gradient_norm_list, gradient_step_list, critere_list)

    return critere_opt, gradient_opt, x_opt
