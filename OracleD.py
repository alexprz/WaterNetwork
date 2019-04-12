from Probleme_R import *
from Structures_N import *

def J(q):
    return 1./3*np.vdot(q, r*q*np.absolute(q)) + np.vdot(pr, np.dot(Ar, q))

def OracleDG(lbd, ind=4):

    q_hash = np.zeros(n)

    x = np.dot(Ar.T, pr) + np.dot(Ad.T, lbd)

    for i in range(n):
        y = x[i]/r[i]
        q_hash[i] = np.sqrt(abs(y))*np.sign(y)

    F, G = 0, 0

    if ind == 2 or ind==4:
        F = J(q_hash) + np.vdot(lbd, np.dot(Ad, q_hash) - fd)
    if ind == 3 or ind==4:
        G = np.dot(Ad, q_hash) - fd

    if ind == 2:
        return F
    if ind == 3:
        return G
    if ind == 4:
        return F, G

def OracleDH(lbd, ind):

    if ind == 2:
        return OracleDG(lbd, 2)
    if ind == 3:
        return OracleDG(lbd, 3)
    if ind == 4:
        return OracleDG(lbd, 4)


    H = np.zeros((md, n))
    x = np.dot(Ar.T, pr) + np.dot(Ad.T, lbd)

    for i in range(md):
        for j in range(n):
            H[i, j] = Ad[i, j]/(2*np.sqrt(r[j]*abs(x[j])))

    if ind == 5:
        return H

    if ind == 6:
        G = OracleDG(lbd, 3)
        return G, H

    if ind == 7:
        F, G = OracleDG(lbd, 4)
        return F, G, H

if __name__=='__main__':
    lbd = np.zeros(md)
    print(OracleDH(lbd, 7))
