import numpy as np
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

from rtfunctions import one_full_fs, sc_2nd_order, calc_lambda_full, calc_lambda_monoc

def solve_RT(ltau_min, ltau_max, ND, eps_const, line_ratio, niter=200, out_every=20):
    # We will define a discrete grid for our spacial coordinate, that is logtau 
    # This is continuum
    logtau = np.linspace(ltau_min, ltau_max, ND)
    tau = 10.0**logtau

    # Planck Function, so the atmosphere is isothermal:
    B = np.zeros(ND)
    B[:] = 1.0

    # Photon destruction probability, smaller this is, more NLTE we are.
    eps = np.zeros(ND)
    eps[:] = eps_const

    # Wavelength grid, in Doppler width units.
    # Profile is constant with depth, for simplicity
    NL = 21
    x = np.linspace(-4,4,NL)
    profile = 1./np.sqrt(np.pi) * np.exp(-(x**2.0))

    # Wavelength integration weights
    prof_norm = np.sum(profile)
    wx = np.zeros(NL)
    wx[0] = (x[1] - x[0]) * 0.5
    wx[-1] = (x[-1] - x[-2]) * 0.5
    wx[1:-1] = (x[2:NL] - x[0:-2]) * 0.5
    norm =  (np.sum(profile*wx))
    wx/= norm

    # Angle integration:
    mu=([1./np.sqrt(3.0)])
    wmu=[1.0]
    mu=np.cos([0.4793425352,1.0471975512,1.4578547042])
    wmu=[.2777777778,0.4444444444,0.2777777778]
    NM = mu.shape[0]
    mu = np.asarray(mu)
    wmu = np.asarray(wmu)

    #Boundary conditions and starting value for the source function
    I_boundary_lower = B[-1]
    I_boundary_upper = 0.0
    S = np.copy(B)

    outtau = []
    outS = []
    outJ = []
    outL = []
    outeps = []
    outratio = []

    # Iteration part
    for iter in range(niter):
        
        # Initialize the scattering integral and local lambda operator
        J = np.zeros(ND)
        L = np.zeros(ND)

        # For each direction and wavelength, calculate the specific monochromatic intensity
        # and add contributions to the mean intensity and the local operator
        for m in range(0,NM):
            for l in range(0,NL):

                #outward
                ILambda = sc_2nd_order(tau*profile[l]*line_ratio,S,mu[m],B[-1])

                J+=ILambda[0]*profile[l]*wx[l]*wmu[m]*0.5
                L+=ILambda[1]*profile[l]*wx[l]*wmu[m]*0.5

                #inward
                ILambda = sc_2nd_order(tau*profile[l]*line_ratio,S,-mu[m],0)

                J+=ILambda[0]*profile[l]*wx[l]*wmu[m]*0.5
                L+=ILambda[1]*profile[l]*wx[l]*wmu[m]*0.5

        if (out_every is not None):
            if (iter // out_every == iter / out_every):
                outtau.append(tau)
                outS.append(S.copy())
                outJ.append(J)
                outL.append(L)
                outeps.append(eps)
                outratio.append(line_ratio)
            
        # Correct the source function using local ALI approach:		
        dS = (eps * B + (1.-eps) * J - S) / (1.-(1.-eps)*L)

        # Check for change
        max_change  = np.max(np.abs(dS / S))
        # print(iter, iter // out_every, max_change)

        # Correct the source function
        S += dS
        if (max_change<1E-6):
            outtau.append(tau)
            outS.append(S.copy())
            outJ.append(J)
            outL.append(L)
            outeps.append(eps)
            outratio.append(line_ratio)
            break
    
    outtau = np.vstack(outtau)
    outS = np.vstack(outS)
    outJ = np.vstack(outJ)
    outL = np.vstack(outL)
    outeps = np.vstack(outeps)
    outratio = np.vstack(outratio)

    return outtau, outS, outJ, outL, outeps, outratio


if (__name__ == '__main__'):

    # Generate 300 random 2-level problems for training        
    nsolutions = 300

    tau_all = []
    S_all = []
    J_all = []
    L_all = []
    eps_all = []
    ratio_all = []
    kernel = 1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-1, 10.0))
    gp = GaussianProcessRegressor(kernel=kernel)

    for i in tqdm(range(nsolutions)):
        eps = 10.0**np.random.uniform(low=-6, high=-2, size=1)[0]
        ltau_min = np.random.uniform(low=-8, high=-6, size=1)[0]
        ltau_max = np.random.uniform(low=np.log10(1.0/eps), high=np.log10(1.0/eps)+3, size=1)[0]
        ND = np.random.randint(low=70, high=100, size=1)[0]

        eps = 10.0**np.random.uniform(low=-6, high=-2, size=1)[0]
        line_ratio = 10.0**np.random.uniform(low=1, high=3, size=1)[0]
        tau, S, J, L, epsilon, ratio = solve_RT(ltau_min, ltau_max, ND, eps_const=eps, line_ratio=line_ratio, niter=2000, out_every=None)

        tau_all.append(tau)
        S_all.append(S)
        J_all.append(J)
        L_all.append(L)
        eps_all.append(epsilon)
        ratio_all.append(ratio)
        
    with open(f'training.pk', 'wb') as filehandle:
        pickle.dump(tau_all, filehandle)
        pickle.dump(S_all, filehandle)
        pickle.dump(J_all, filehandle)
        pickle.dump(L_all, filehandle)
        pickle.dump(eps_all, filehandle)
        pickle.dump(ratio_all, filehandle)


    # Generate 300 random 2-level problems for validation
    nsolutions = 50

    tau_all = []
    S_all = []
    J_all = []
    L_all = []
    eps_all = []
    ratio_all = []
    for i in tqdm(range(nsolutions)):
        eps = 10.0**np.random.uniform(low=-6, high=-2, size=1)[0]
        ltau_min = np.random.uniform(low=-8, high=-6, size=1)[0]
        ltau_max = np.random.uniform(low=np.log10(1.0/eps), high=np.log10(1.0/eps)+3, size=1)[0]
        ND = np.random.randint(low=70, high=100, size=1)[0]
        
        line_ratio = 10.0**np.random.uniform(low=1, high=3, size=1)[0]
        tau, S, J, L, epsilon, ratio = solve_RT(ltau_min, ltau_max, ND, eps_const=eps, line_ratio=line_ratio, niter=2000, out_every=None)

        tau_all.append(tau)
        S_all.append(S)
        J_all.append(J)
        L_all.append(L)
        eps_all.append(epsilon)
        ratio_all.append(ratio)
        
    with open(f'test.pk', 'wb') as filehandle:
        pickle.dump(tau_all, filehandle)
        pickle.dump(S_all, filehandle)
        pickle.dump(J_all, filehandle)
        pickle.dump(L_all, filehandle)
        pickle.dump(eps_all, filehandle)
        pickle.dump(ratio_all, filehandle)