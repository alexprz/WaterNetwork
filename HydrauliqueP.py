#!/usr/bin/python

import numpy as np
from numpy import dot
from numpy import transpose as t

###############################################################################
#                                                                             #
# CALCUL DES VARIABLES HYDRAULIQUES DU RESEAU A PARTIR DES DEBITS DU CO-ARBRE #
#                                                                             #
###############################################################################

# Supposant connu le vecteur des debits sur le co-arbre, on calcule l'ensemble
# des variables  hydrauliques du reseau ; on dispose pour cela des matrices et
# du debit admissible qui ont ete calcules precedemment.
#
# Variables en entree
#
#    - qc   : vecteur des debits des arcs du co-arbre
#
# Variables en sortie
#
#    - q    : vecteur des debits des arcs
#    - z    : vecteur des pertes de charge des arcs
#    - f    : vecteur des flux des noeuds
#    - p    : vecteur des pressions des noeuds

from Probleme_R import *
from Structures_N import *

def HydrauliqueP(qc):
    
    # Debits des arcs
    q = q0 + dot(B, qc)
    
    # Pertes de charge des arcs
    z = r * abs(q) * q
    
    # Flux des noeuds
    f = np.zeros(m)
    f[:mr] = dot(Ar, q)
    f[mr:] = fd
    
    # Pression aux noeuds
    p = np.zeros(m)
    p[:mr] = pr
    p[mr:] = -dot(t(AdI), (dot(t(Ar), pr) + z)[:md])

    return q, z, f, p
