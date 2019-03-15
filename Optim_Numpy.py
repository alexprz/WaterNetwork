#!/usr/bin/python

import numpy as np
from scipy.optimize import minimize

#############################################################################
#                                                                           #
#         RESOLUTION D'UN PROBLEME D'OPTIMISATION SANS CONTRAINTES          #
#                                                                           #
#         Methode d'optimisation de scipy                                   #
#                                                                           #
#############################################################################

def Optim_Numpy(Oracle, x0):

    results = minimize(lambda x: Oracle(x)[0], x0, jac=lambda x: Oracle(x)[1])

    critere_opt = results.fun
    gradient_opt = results.jac
    x_opt = results.x
    
    mess = results.message
    print(mess)
    
    return critere_opt, gradient_opt, x_opt
