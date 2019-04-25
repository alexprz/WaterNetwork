#!/usr/bin/python

import numpy as np
from numpy.linalg import norm
from time import process_time
from Wolfe_Skel import Wolfe

#############################################################################
#                                                                           #
#         RESOLUTION D'UN PROBLEME D'OPTIMISATION SANS CONTRAINTES          #
#                                                                           #
#         Methode BFGS                                                      #
#                                                                           #
#############################################################################

from Visualg import Visualg


def BFGS(Oracle, x0):

    ##### Initialisation des variables

    iter_max = 10000
    gradient_step = 0.0005
    threshold = 0.000001

    gradient_norm_list = []
    gradient_step_list = []
    critere_list = []

    time_start = process_time()

    x = x0
    Id = np.identity(len(x))
    W = Id

    ##### Boucle sur les iterations

    critere_n, gradient_n = Oracle(x, 4)
    D = -gradient_n
    alpha_n = 1

    for k in range(iter_max):

        alpha_p = alpha_n
        alpha_n, ok = Wolfe(8, x, D, Oracle)

        # Mise a jour des variables
        x_p = x
        x = x + alpha_n*D

        gradient_p = gradient_n
        critere_n, gradient_n = Oracle(x, 4)
        if norm(gradient_n) <= threshold:
            break


        delta_x = np.reshape(x-x_p, (len(x),1))
        delta_g = np.reshape(gradient_n-gradient_p, (len(x),1))
        denominateur = np.vdot(delta_g,delta_x)

        mat1 = Id - np.dot(delta_x,delta_g.T)/denominateur
        mat2 = Id - np.dot(delta_g,delta_x.T)/denominateur
        mat3 = np.dot(delta_x,delta_x.T)/denominateur

        W = np.dot(mat1, np.dot(W, mat2)) + mat3
        D = -np.dot(W, gradient_n)

        # Evolution du gradient, du pas, et du critere
        gradient_norm_list.append(norm(gradient_n))
        gradient_step_list.append(alpha_n)
        critere_list.append(critere_n)



    critere_opt = critere_n
    gradient_opt = gradient_n
    x_opt = x
    time_cpu = process_time() - time_start

    print()
    print('BFGS')
    print('Iteration :', k)
    print('Temps CPU :', time_cpu)
    print('Critere optimal :', critere_opt)
    print('Norme du gradient :', norm(gradient_opt))

    # Visualisation de la convergence
    Visualg(gradient_norm_list, gradient_step_list, critere_list)

    return critere_opt, gradient_opt, x_opt
