import os
import numpy as np
from numpy import linalg as la
from scipy import interpolate
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import lsmr
from scipy.optimize import minimize




def svd(A, b):
    symTol = 1e-8
    u, s, v = la.svd(A)

    st = np.zeros((A.shape[1], A.shape[0]))
    for i in range(len(s)):
        if s[i] < symTol: s[i] = 0
        if s[i] != 0:
            st[i,i] = 1/s[i]
        if s[i] == 0:
            st[i,i] = 0
    
    ut = u.T
    vt = v.T
  
    x =  vt * st * ut * b
    return x

def recast(wave, flux, wave_low, wave_max, delta):
    # put wave into ln_space
    Ln_wave = np.log10(wave)
    y = interpolate.interp1d(Ln_wave, flux, kind='linear')
    #construction
    ln_wave = np.arange(np.log10(wave_low), np.log10(wave_max), delta)
    ff = y(ln_wave)
    return ln_wave, ff

def get_matrix_L(Nplx, gamma):
    dia= np.ones(Nplx)
    L = np.diag(dia)
    for i in range(len(L)-1):
        L[i,i+1] = -0.5
        L[i+1,i] = -0.5
    L = np.sqrt(gamma)*L
    return L

def get_matrix_N(Ln_wave, V, delta):
    #if not shitf for A and B
    if V == 0:
        C = np.ones(len(Ln_wave))
        N = np.diag(C)

    #if red-shift 
    if V > 0:
        N = np.zeros((len(Ln_wave), len(Ln_wave)))
        R = np.log10(1 + V/299792.458) / delta  
        m = int(R)
        for i in range(len(Ln_wave)):  
            if m+i == len(Ln_wave): break
            N[m+i][i] = np.mod(R, 1)
            if m+i < len(Ln_wave)-1:
                N[m+i+1][i] = 1 - np.mod(R, 1)

    #if blue-shift 
    if V < 0:
        N = np.zeros((len(Ln_wave), len(Ln_wave)))
        R = np.log10(1 + V/299792.458) / delta  
        m = int(-R)
        for i in range(len(Ln_wave)): 
            if m+i == len(Ln_wave): break
            N[i][m+i] = 1 - np.mod(R, 1)
            if m+i < len(Ln_wave)-1:
                N[i][m+i+1] = np.mod(R, 1)   
    return N

def get_matrix_M(lw, vA, q, v0, delta):
    vB = -q * (vA-v0) + v0
    NA = get_matrix_N(lw, vA, delta)
    NB = get_matrix_N(lw, vB, delta)
    M = np.concatenate((NA, NB), axis=1)  
    A = np.sum(NA, axis=1) * np.sum(NB, axis=1)
    index = A == 0
    M[index] = 0
    M = np.array(M)
    return M

def dist(q, gamma, v0, rvs, ln_wave, fobs, delta=1e-5, solver='lsmr'):

    Nplx = len(ln_wave)
    l = get_matrix_L(2*Nplx, gamma)
    f_regular = np.zeros(2*Nplx)
    f = np.concatenate((fobs, f_regular))

    for i in range(len(rvs)):
        m = get_matrix_M(ln_wave, rvs[i], q, v0, delta)
        if i > 0:
            m = np.concatenate((m, mpre))
        mpre = m
    M = np.concatenate((m, l))

    if solver == 'svd':
        M = np.matrix(M)
        f = np.matrix(f).T
        x = svd(M, f)

        flux_A = np.array(x[0:Nplx].T)[0]
        flux_B = np.array(x[Nplx:2*Nplx].T)[0]
    
    if solver == 'lsmr':
        A = csc_matrix(M, dtype=float)
        b = np.array(f, dtype=float)
        x, istop, itn, normr = lsmr(A, b)[:4]

        flux_A = np.array(x[0:Nplx].T)
        flux_B = np.array(x[Nplx:2*Nplx].T)

    return flux_A, flux_B

def opt_function(q, gamma, rvs, v0, ln_wave, fobs, delta=1e-5):

    for i in range(len(rvs)):
        m = get_matrix_M(ln_wave, rvs[i], q, v0, delta)
        if i > 0:
            m = np.concatenate((m, mpre))
        mpre = m

    flux_A, flux_B = dist(q, gamma, v0, rvs, ln_wave, fobs)
    tolf = np.concatenate((flux_A, flux_B))
    tf = np.matrix(tolf).T
    m = np.matrix(m)
    fpred = m*tf

    N = len(fobs)
    f = np.concatenate((fobs, np.array(fpred.T)[0]))
    ff = f.reshape(2, N)
    cov = np.cov(ff, rowvar=False)
    cov = np.matrix(cov)
    r = fobs - np.array(fpred.T)[0]
    R = np.matrix(r)

    return np.array(R*R.T)[0][0]


def optimisation(q, gamma, rvs, v0, ln_wave, fobs, xatol=1e-8, fatol=1e-8, maxiter=1000, method='nelder-mead', disp=False):
    res = minimize(opt_function, q, args=(gamma, rvs, v0, ln_wave, fobs), method=method, options={'xatol': xatol, 'fatol': fatol, 'disp': disp, 'maxiter': maxiter})
    return res.x


