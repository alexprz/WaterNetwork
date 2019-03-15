#!/usr/bin/python

import numpy as np
from numpy import dot
from numpy import zeros
from numpy import sqrt
from numpy import transpose as t

#################################################################################
#                                                                               #
# CALCUL DES VARIABLES HYDRAULIQUES DU RESEAU A PARTIR DES PRESSIONS AUX NOEUDS #
#                                                                               #
#################################################################################

# Supposant connues les pressions aux noeuds de demande, on calcule l'ensemble
# des variables hydrauliques du reseau ; on dispose pour cela des matrices qui
# ont ete calcules precedemment.
#
# Variables en entree
#
#    - pd   : vecteur des pressions aux noeuds de demande
#
# Variables en sortie
#
#    - q    : vecteur des debits des arcs
#    - z    : vecteur des pertes de charge des arcs
#    - f    : vecteur des flux des noeuds
#    - p    : vecteur des pressions des noeuds

from Probleme_R import *
from Structures_N import *

def HydrauliqueD(pd):
    
    # Pressions aux noeuds
    p = zeros(m)
    p[:mr] = pr
    p[mr:m] = pd
    
    # Pertes de charge des arcs
    z = - dot(t(A), p)
    
    # Debits des arcs
    q = z / sqrt(r*abs(z))
    
    # Flux aux noeuds
    f = zeros(m)
    f[:mr] = dot(Ar,q)
    f[mr:m]= fd
    
    return q, z, f, p
    
