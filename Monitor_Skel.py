#!/usr/bin/python

import numpy as np

#############################################################################
#                                                                           #
#  MONITEUR D'ENCHAINEMENT POUR LE CALCUL DE L'EQUILIBRE D'UN RESEAU D'EAU  #
#                                                                           #
#############################################################################

##### Fonctions fournies dans le cadre du projet

# Donnees du probleme
from Probleme_R import *
from Structures_N import *

# Affichage des resultats
from Visualg import Visualg

# Verification des resultats
from HydrauliqueP import HydrauliqueP
from HydrauliqueD import HydrauliqueD
from Verification import Verification

##### Fonctions a ecrire dans le cadre du projet

# ---> Charger les fonctions associees a l'oracle du probleme,
#      aux algorithmes d'optimisation et de recherche lineaire
#
#      Exemple 1 - le gradient a pas fixe :
#
from OraclePG import OraclePG, OraclePH
from Gradient_F import Gradient_F
from Optim_Numpy import Optim_Numpy
from Newton_F import Newton_F
x0 = 0.1 * np.random.normal(size=n-md)
# lbd0 = 0.1 * np.random.normal(size=md)
lbd0 = np.random.randn(md)
# qc[0] = 1
# print(OraclePG(qc, 2))
# Gradient_F(lambda x:OraclePG(x, 4), qc)
# print(Optim_Numpy(lambda x:OraclePG(x, 4), qc))
# print(Newton_F(lambda x:OraclePH(x, 7), qc))

#
#      Exemple 2 - le gradient a pas variable :
#
# from OraclePG import OraclePG
# from Gradient_V import Gradient_V
# from Wolfe import Wolfe
#
# ---> A modifier...
# ---> A modifier...
# ---> A modifier...
from OraclePG import OraclePG, OraclePH
from OracleD import OracleDG, OracleDH
from Gradient_V import Gradient_V
from Polak_Ribiere import Polak_Ribiere
from BFGS import BFGS
from Gradient_Newton import Newton
from Wolfe_Skel import Wolfe

# Gradient_F(OraclePG, x0)
# Gradient_V(OraclePG, x0)
# Polak_Ribiere(OraclePG, x0)
# BFGS(OraclePG, x0)
# Newton(OraclePH, x0)
# Gradient_F(OracleDG, lbd0)
# Gradient_V(OracleDG, lbd0)
# Polak_Ribiere(OracleDG, lbd0)
# BFGS(OracleDG, lbd0)
Newton(OracleDH, lbd0)

##### Initialisation de l'algorithme

# ---> La dimension du vecteur dans l'espace primal est n-md
#      et la dimension du vecteur dans l'espace dual est md
#
#      Probleme primal :
#
#                        x0 = 0.1 * np.random.normal(size=n-md)
#
#      Probleme dual :
#
#                        x0 = 100 + np.random.normal(size=md)
#
# ---> A modifier...
# ---> A modifier...
# ---> A modifier...

##### Minimisation proprement dite

# ---> Executer la fonction d'optimisation choisie
#
#      Exemple 1 - le gradient a pas fixe :
#
#                  print()
#                  print("ALGORITHME DU GRADIENT A PAS FIXE")
#                  copt, gopt, xopt = Gradient_F(OraclePG, x0)
#
#      Exemple 2 - le gradient a pas variable :
#
#                  print()
#                  print("ALGORITHME DU GRADIENT A PAS VARIABLE")
#                  copt, gopt, xopt = Gradient_V(OraclePG, x0)
#
# ---> A modifier...
# ---> A modifier...
# ---> A modifier...

##### Verification des resultats

# ---> La fonction qui reconstitue les variables hydrauliques
#      du reseau a partir de la solution du probleme s'appelle
#      HydrauliqueP pour le probleme primal, et HydrauliqueD
#      pour le probleme dual
#
#      Probleme primal :
#
#                        qopt, zopt, fopt, popt = HydrauliqueP(xopt)
#
#
#                        qopt, zopt, fopt, popt = HydrauliqueDxopt)
#
# ---> A modifier...
# ---> A modifier...
# ---> A modifier...

# Verification(qopt, zopt, fopt, popt)
