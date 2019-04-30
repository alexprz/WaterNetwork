#!/usr/bin/python

#############################################################################
#                                                                           #
#  MONITEUR D'ENCHAINEMENT POUR LE CALCUL DE L'EQUILIBRE D'UN RESEAU D'EAU  #
#                                                                           #
#############################################################################

# Donnees du probleme
from Probleme_R import *
from Structures_N import *

# Verification des resultats
from HydrauliqueP import HydrauliqueP
from HydrauliqueD import HydrauliqueD
from Verification import Verification

# Oracles
from OracleP import OraclePG, OraclePH
from OracleD import OracleDG, OracleDH

# Méthodes d'optimisation
from Gradient_F import Gradient_F
from Newton_F import Newton_F
from Gradient_V import Gradient_V
from Polak_Ribiere import Polak_Ribiere
from BFGS import BFGS
from Gradient_Newton import Newton
from Optim_Numpy import Optim_Numpy

# Points initiaux
x0 = 0.1 * np.random.normal(size=n-md)
lbd0 = 100 + np.random.normal(size=md)

# Primal
Gradient_F(OraclePG, x0)
Newton_F(OraclePH, x0)
Gradient_V(OraclePG, x0)
Polak_Ribiere(OraclePG, x0)
BFGS(OraclePG, x0)
copt, gopt, xopt = Newton(OraclePH, x0)

# Dual
Gradient_F(OracleDG, lbd0, dual=True)
Gradient_V(OracleDG, lbd0, dual=True)
Polak_Ribiere(OracleDG, lbd0, dual=True)
BFGS(OracleDG, lbd0, dual=True)
copt_dual, gopt_dual, xopt_dual = Newton(OracleDH, lbd0)

# Scipy optimizationation
copt_scipy, gopt_scipy, xopt_scipy = Optim_Numpy(lambda x:OraclePG(x, 4), x0)
copt_scipy_dual, gopt_scipy_dual, xopt_scipy_dual = Optim_Numpy(lambda x:OracleDG(x, 4), lbd0)

# Vérification
qopt, zopt, fopt, popt = HydrauliqueP(xopt)
Verification(qopt, zopt, fopt, popt)
qopt, zopt, fopt, popt = HydrauliqueD(xopt_dual)
Verification(qopt, zopt, fopt, popt)

print('Vérification Scipy primal')
qopt, zopt, fopt, popt = HydrauliqueP(xopt_scipy)
Verification(qopt, zopt, fopt, popt)
print('Vérification Scipy dual')
qopt, zopt, fopt, popt = HydrauliqueD(xopt_scipy_dual)
Verification(qopt, zopt, fopt, popt)
