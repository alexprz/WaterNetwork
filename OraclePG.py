from Probleme_R import *
from Structures_N import *

def OraclePG(qc, ind):
    q0Bqc = q0 + dot(B, qc)

    F, G = 0, 0

    if ind == 2 or ind==4:
        F = 1/3.*np.vdot(q0Bqc, r * q0Bqc * np.absolute(q0Bqc)) + np.vdot(pr, dot(Ar, (q0Bqc)))
    if ind == 3 or ind==4:
        G = np.dot(B.T, r*q0Bqc*np.absolute(q0Bqc) + dot(Ar.T, pr))

    if ind == 2:
        return F
    if ind == 3:
        return G
    if ind == 4:
        return F, G

def OraclePH(qc, ind):
    q0Bqc = q0 + dot(B, qc)

    if ind == 2:
        return OraclePG(qc, 2)
    if ind == 3:
        return OraclePG(qc, 3)
    if ind == 4:
        return OraclePG(qc, 4)


    H = np.zeros((n-md, n-md))

    for i in range(n-md):
        for j in range(n-md):
            for k in range(n):
                H[i,j] += 2*B[k,i]*r[k]*np.absolute(q0Bqc)[k]*B[k,j]

    if ind == 5:
        return H

    if ind == 6:
        G = OraclePG(qc, 3)
        return G, H

    if ind == 7:
        F, G = OraclePG(qc, 4)
        return F, G, H

if __name__=='__main__':
    qc = np.zeros(n-md)
    print(OraclePH(qc, 7))
