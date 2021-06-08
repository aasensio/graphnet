import numpy as np

def parabcoef(dtm, dtp):
    if (dtm > 0.001):
        exu = np.exp(-dtm)
        u0 = 1.0 - exu
        u1 = dtm - 1.0 + exu
        u2 = dtm**2 - 2.0*dtm + 2.0 - 2.0*exu
    else:
        u0 = dtm - dtm**2 / 2.0
        u1 = dtm**2 / 2.0 - dtm**3 / 6.0
        u2 = dtm**3 / 3.0 - dtm**4 / 12.0

    out0 = (u2 - u1 * (dtp + 2.0 * dtm)) / (dtm*(dtm+dtp)) + u0
    out1 = (u1 * (dtm + dtp) - u2) / (dtm*dtp)
    out2 = (u2 - dtm*u1) / (dtp * (dtm+dtp))

    return out0, out1, out2

def linearcoef(dtm):
    if (dtm > 0.001):
        exu = np.exp(-dtm)
        u0 = 1.0 - exu
        u1 = dtm - 1.0 + exu
        out0 = u1 / dtm
        out1 = u0 - out0
    else:
        out0 = dtm / 2.0 - dtm**2 / 6.0
        out1 = dtm / 2.0 - dtm**2 / 3.0

    return out0, out1

def shortcar(tau, S, I0=0.0):
    n = len(tau)
    stI = np.zeros(n)
    stI[0] = I0
    for i in range(1, n-1):
        dtm = np.abs(tau[i] - tau[i-1])
        dtp = np.abs(tau[i+1] - tau[i])

        if (dtm > 0.001):
            exu = np.exp(-dtm)
        else:
            exu = 1.0 - dtm + dtm**2 / 2.0

        psim, psi0, psip = parabcoef(dtm, dtp)

        stI[i] = stI[i-1] * exu + psim * S[i-1] + psi0 * S[i] + psip * S[i+1]

    dtm = np.abs(tau[n-1] - tau[n-2])

    if (dtm > 0.001):
        exu = np.exp(-dtm)
    else:
        exu = 1.0 - dtm + dtm**2 / 2.0

    psim, psi0 = linearcoef(dtm)

    stI[n-1] = stI[n-2] * exu + psim * S[n-2] + psi0 * S[n-1]

    return stI