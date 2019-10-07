# My attempt at learning and using python for research
"""
The script here performs a static WLS state estimation using the Gauss-Newton 
method. 

@author: achattopadhyay
"""

#%% RELEVANT LIBRARIES FOR IMPORT
import numpy as np
#import scipy.sparse.linalg.spsolve as sparse_solve
import scipy.sparse as sparse
import pandas as pd
import time
import os
import matplotlib.pyplot as plt

# Import user made functions
import matpower_scripts as mpc_f

from indices import F_BUS, T_BUS, BR_R, BR_X, BR_B, TAP, SHIFT, BR_STATUS, PF, PT, QF, QT
from indices import PQ, PV, REF, NONE
from indices import BUS_I, BUS_TYPE, PD, QD, GS, BS, VM, VA
from indices import GEN_BUS, PG, QG, QMAX, QMIN, VG, MBASE, GEN_STATUS, PMAX, PMIN

#%% CASE DATA INTIAILIZATIONS
###############################################################################
## Case information
casename = 'WECC2470_int_idx.m'
bus, gen, branch, baseMVA = mpc_f.read_case(casename)
Ybus, Yf, Yt = mpc_f.makeYbus(baseMVA, bus, branch)

# Sort branch to make sure it is the same as the read CSV data
branch = np.array(sorted(branch, key=lambda x: (x[0], x[1])))

#%% Functions
##=============================================================================
def makeYbus(baseMVA, bus, branch):
    # This function evaluates the Ybus matrix and returns Ybus, Yf, and Yt
    nb = bus.shape[0]       # Number of buses
    nl = branch.shape[0]    # Number of lines
    
    status = branch[:, BR_STATUS]       # ones at in-service branches
    Ys = status / (branch[:, BR_R] + 1j * branch[:, BR_X])  # Series admittance
    Bc = status * branch[:, BR_B]       # Line charging susceptance
    tap = np.ones(nl)                   # default tap raito = 1
    i = np.nonzero(branch[:, TAP])      # indices of non-zero tap ratios
    tap[i] = branch[i, TAP]
    tap = tap * np.exp(1j * np.pi /180 * branch[:, SHIFT])  # Add phase shifters
    
    Ytt = Ys + 1j * Bc / 2
    Yff = Ytt / (tap * np.conj(tap))
    Yft = - Ys / np.conj(tap)
    Ytf = -Ys /tap
    
    Ysh = (bus[:, GS] + 1j * bus[:, BS]) / baseMVA
    
    # The -1 in the indices for the sparse are for zero-based indexing in python
    f = branch[:, F_BUS] - 1       # list of "from" buses
    t = branch[:, T_BUS] - 1        # list of "to" buses
 
    Cf = sparse.csr_matrix((np.ones(nl), (np.arange(nl), f)), (nl, nb))
    Ct = sparse.csr_matrix((np.ones(nl), (np.arange(nl), t)), (nl, nb))
    
    i = np.r_[range(nl), range(nl)]
    
    Yf = sparse.csr_matrix((np.r_[Yff, Yft], (i, np.r_[f, t])), (nl, nb))
    Yt = sparse.csr_matrix((np.r_[Ytf, Ytt], (i, np.r_[f, t])), (nl, nb))
    
    Ybus = Cf.T @ Yf + Ct.T @ Yt + sparse.csr_matrix((Ysh, (range(nb), range(nb))), (nb, nb))
    
    return Ybus, Yf, Yt
##=============================================================================

#==============================================================================
# Derivatives of S_inj w.r.t V 
def dSbus_dV(Ybus, V):
    # This function computes partial derivatives of power injection w.r.t voltage
    ib = range(len(V))
    
    Ibus = Ybus * V
    
    diagV = sparse.csr_matrix((V, (ib, ib)))
    diagIbus = sparse.csr_matrix((Ibus, (ib, ib)))
    diagVnorm = sparse.csr_matrix((V / abs(V), (ib, ib)))
    
    dS_dVm = diagV @ np.conj(Ybus @ diagVnorm) + np.conj(diagIbus) @ diagVnorm
    dS_dVa = 1j * diagV @ np.conj(diagIbus - Ybus @ diagV)
    
    return dS_dVm, dS_dVa
##=============================================================================

##=============================================================================
# Derivatives of S_br w.r.t V  
def dSbr_dV(branch, Yf, Yt, V):
    # This function computes partial derivatives of power flows w.r.t voltage
    nb = V.shape[0]       # Number of buses
    nl = branch.shape[0]    # Number of lines
    v_ones = np.ones(nl)    # number of 1's to be in the connection matrix, equal to number of lines
    
    il = np.arange(nl)      # Row indices for branches
    ib = np.arange(nb)      # Row indices for buses
    
    # Make "from" connection matrix
    # If branch 'i' is connected to bus 'j', then element c_ij of Cf = 1, else = 0
    f = branch[:, F_BUS].astype(int) - 1    # list of "from" buses
    If = Yf @ V
    Vf = V[f]
    Cf = sparse.csr_matrix((v_ones, (il, f)), (nl, nb))
    
    # Make "to" connection matrix
    # If branch 'i' is connected to bus 'j', then element c_ij of Ct = 1, else = 0
    t = branch[:, T_BUS].astype(int) - 1    # list of "to" buses
    It = Yt @ V
    Vt = V[t]
    Ct = sparse.csr_matrix((v_ones, (il, t)), (nl, nb))
    
    diagVf = sparse.csr_matrix((Vf, (il, il)))
    diagIf = sparse.csr_matrix((If, (il, il)))
    
    diagVt = sparse.csr_matrix((Vt, (il, il)))
    diagIt = sparse.csr_matrix((It, (il, il)))
    
    diagV = sparse.csr_matrix((V, (ib, ib)), (nb, nb))
    diagVnorm = sparse.csr_matrix((V / abs(V), (ib, ib)), (nb, nb))
    
    dSf_dVa = 1j * (np.conj(diagIf) @ Cf @ diagV - diagVf @ np.conj(Yf) @ np.conj(diagV))
    dSf_dVm = np.conj(diagIf) @ Cf @ diagVnorm + diagVf @ np.conj(Yf) @ np.conj(diagVnorm)
    
    dSt_dVa = 1j * (np.conj(diagIt) @ Ct @ diagV - diagVt @ np.conj(Yt) @ np.conj(diagV))
    dSt_dVm = np.conj(diagIt) @ Ct @ diagVnorm + diagVt @ np.conj(Yt) @ np.conj(diagVnorm)
    
    Sf = V[f] * np.conj(If)
    St = V[t] * np.conj(It)
    
    return dSf_dVa, dSf_dVm, dSt_dVa, dSt_dVm, Sf, St
##=============================================================================

##=============================================================================
def bus_types(bus, gen):
# This function obtains the indices of the REF, PV, and PQ buses
    nb = bus.shape[0]
    ng = gen.shape[0]
    Cg = sparse.csr_matrix((gen[:, GEN_STATUS]>0, (gen[:, GEN_BUS]-1, range(ng))), (nb, ng))
    
    gen_bus_status = (Cg @ (np.ones(ng, int).T).astype(bool))
    
    # form index list for slack, PV, and PQ buses
    bus_types = bus[:, BUS_TYPE]
    
    ref = np.where((bus_types==REF) & gen_bus_status)
    pv = np.where((bus_types==PV) & gen_bus_status)
    pq = np.where((bus_types==PQ) | ~gen_bus_status)
    
    if len(ref) == 0:
        ref = np.zeros(1, dtype=int)
        ref[0] = pv[0]
        pv = pv[1:]
     
    return ref, pv, pq
##=============================================================================

##=============================================================================
def read_line_flows(file):
# This function reads the csv file of line flows
    df = pd.read_csv(file, sep=',', skiprows = [0])
    
    a = df.sort_values(by=['From Number', 'To Number'])
    line_flows = a.values
    
    return line_flows
##=============================================================================

#%% STATE ESTIMATION INITIALIZATIONS
## Initializations for State Estimation
converged = False
iteration = 0
max_it = 5
st_dev = 0.02
tolerance = 1e-3

# File name of measurements
file_directory = os.path.dirname(os.path.realpath(__file__))
folder_name = 'Extended_csv_files_2500'
file_name = 't00_00.csv'
file = (file_directory + '/' + folder_name + '/' + file_name)

# Find number of buses and lines
nb = bus.shape[0]
nl = branch.shape[0]

# Find bus index lists
ref, pv, pq = bus_types(bus, gen)
nref = list(np.append(pv, pq))
nref.sort()

#%% Assemble state estimation data structures
# Intial bus voltage profile
V = np.ones(nb)   # Using flat start
Fnorm = []

# read line flows
line_flows = read_line_flows(file)

# Create measurement vectors of Pf, Pt, Qf, and Qt from line flows read from csv
z_Pf = line_flows[:, 0]/baseMVA
z_Pt = line_flows[:, 1]/baseMVA
z_Qf = line_flows[:, 2]/baseMVA
z_Qt = line_flows[:, 3]/baseMVA

# Make z vector of measurements
z = np.concatenate([z_Pf, z_Pt, z_Qf, z_Qt])

# Make co-variance matrix
# st. dev vector for each type of measurements
sigma_Pf = np.transpose(np.ones(len(z_Pf))) * st_dev
sigma_Pt = np.transpose(np.ones(len(z_Pf))) * st_dev
sigma_Qf = np.transpose(np.ones(len(z_Pf))) * st_dev
sigma_Qt = np.transpose(np.ones(len(z_Pf))) * st_dev

# st.dev vector for all measurements
sigma = np.concatenate([sigma_Pf, sigma_Pt, sigma_Qf, sigma_Qt])
# Co-variance matrix: contains elements of sigma^2 along diagonals
Rinv = sparse.diags((1/sigma**2))

# Create indices of measurements that will be passed. 
# This is the real and reactive flows for all lines
idx_Pf = list(range(nl))
idx_Pt = list(range(nl))
idx_Qf = list(range(nl))
idx_Qt = list(range(nl))

# Create index grid of Jacobian elements that will be retrieved
ij_Pf = np.ix_(idx_Pf, nref)
ij_Pt = np.ix_(idx_Pt, nref)
ij_Qf = np.ix_(idx_Qf, nref)
ij_Qt = np.ix_(idx_Qt, nref)

# make system admittance matrices
Ybus, Yf, Yt = makeYbus(baseMVA, bus, branch)

f = branch[:, F_BUS].astype(int) - 1
t = branch[:, T_BUS].astype(int) - 1

converter_bus = 333 - 1
inverter_bus = 1124 - 1

#%% Do Gauss-Newton iterations

# Start timer for S.E 
start_time = time.time()

while (not converged) & (iteration < max_it):
    
    iteration = iteration + 1

    # Get the voltage magnitude and angles
    Vm = abs(V)
    Va = np.angle(V)
    
    # Compute estimated measurement based on initial guess of state
    Sfe = V[f] * np.conj(Yf @ V)
    Ste = V[t] * np.conj(Yt @ V)
    
    # separate real and imaginary parts
    Pfe = np.real(Sfe)              # real components
    Pte = np.real(Ste)
    Qfe = np.imag(Sfe)              # imaginary components
    Qte = np.imag(Ste)
    
    # Assemble into single vector of estimated measurements
    z_est = np.concatenate([Pfe, Pte, Qfe, Qte])
    
    # Compute derivatives needed to make H matrix
    # Derivatives of line flows
    dSf_dVa, dSf_dVm, dSt_dVa, dSt_dVm, Sf, St = dSbr_dV(branch, Yf, Yt, V)
    
    # For sub-blocks of H related to line flows
    dPf_dVa = np.real(dSf_dVa)  # "from" end, wrt angle
    dQf_dVa = np.imag(dSf_dVa)
    
    dPt_dVa = np.real(dSt_dVa)  # "to" end, wrt angle
    dQt_dVa = np.imag(dSt_dVa)
    
    dPf_dVm = np.real(dSf_dVm)  # "from" end, wrt magnitude
    dQf_dVm = np.imag(dSf_dVm)
    
    dPt_dVm = np.real(dSt_dVm) # "to" end, wrt magnitude
    dQt_dVm = np.imag(dSt_dVm)

    # select rows and columns of sub-blocks given by [rows=idx, col=nref]
    J11 = dPf_dVa[ij_Pf]    # J11 = dPf_dVa
    J12 = dPf_dVm[ij_Pf]    # J11 = dPf_dVm
    J21 = dPt_dVa[ij_Pt]    # J11 = dPt_dVa
    J22 = dPt_dVm[ij_Pt]    # J11 = dPt_dVm
    J31 = dQf_dVa[ij_Qf]    # J11 = dQf_dVa
    J32 = dQf_dVm[ij_Qf]    # J11 = dQf_dVm
    J41 = dQt_dVa[ij_Qt]    # J11 = dQf_dVa
    J42 = dQt_dVm[ij_Qt]    # J11 = dQf_dVm
    
    # The state vector is X = [Vmag; Vang], so create H in that order 
    H = sparse.vstack((
            sparse.hstack((J11, J12)),
            sparse.hstack((J21, J22)),
            sparse.hstack((J31, J32)),
            sparse.hstack((J41, J42)),           
            )).tocsr()
    
    # Compute gain matrix J and Residual F so that J * delta x = F
    # Gain matrix J = H' x Rinv x H
    J = H.T @ Rinv @ H
    
    # Residual matrix F = H' x Rinv x (z - z_est)
    F = H.T @ Rinv @ (z - z_est)
    
    dx = sparse.linalg.spsolve(J, F)

    normF = np.linalg.norm(F, np.inf)
    
    Fnorm.append(normF)
    print('Iteration = {0} \t\t Inf norm of residual matrix = {1:.2E}'.format(iteration, normF))
    
    # If infinite norm of residual is less than tolerance
    if normF < tolerance:
        converged = True
    
    # Update the voltage magnitude and angles
    Va[nref] = Va[nref] + dx[0 : len(nref)]
    Vm[nref] = Vm[nref] + dx[len(nref) : ]
    
    V = Vm * np.exp(1j * Va)
    
    S_inj = V * np.conj( Ybus @ V)
#    print('Estimate of injection at converter = {0:4.0f} MW'. format(- np.real(S_inj[converter_bus]) * baseMVA))
#    print('Estimate of injection at inverter = {0:4.0f} MW'. format(- np.real(S_inj[inverter_bus]) * baseMVA))
    
plt.plot(Fnorm)
plt.yscale('log')
    
elapsed_time = time.time() - start_time
print('')
print('Total elapsed time for Gauss-Newton method {0:.2E} sec'.format(elapsed_time))