#!/usr/bin/python

import numpy as np
from numpy import dot, transpose as t

##############################################################################
#                                                                            #
#  VERIFICATION DES EQUATIONS D'EQUILIBRE D'UN RESEAU DE DISTRIBUTION D'EAU  #
#                                                                            #
##############################################################################

# On suppose determinee la solution du probleme d'optimisation et reconstituee
# les variables hydrauliques du reseau. On calcule le plus grand ecart sur les
# 2 series d'equations qui caracterisent l'equilibre du reseau.
#
# Variables en entree
#
#    - q : vecteur des debits des arcs
#    - z : vecteur des pertes de charge des arcs
#    - f : vecteur des flux aux noeuds
#    - p : vecteur des pressions aux noeuds

from Probleme_R import *
from Structures_N import *

def Verification(q, z, f, p):
    
    # Ecarts maximaux sur les lois de Kirschoff
    tol_debits = max(abs(dot(A, q) - f))
    tol_pression = max(abs(dot(t(A), p) + z))
    
    # Affichage
    print()
    print("Vérification des équations d'équilibre du réseau")
    print("Sur les débits : {}".format(tol_debits))
    print("Sur les pressions : {}".format(tol_pression))

