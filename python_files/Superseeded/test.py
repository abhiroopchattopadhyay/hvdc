# My attempt at learning and using python for research
"""
My attempt at using python for my research 
"""

#%% RELEVANT LIBRARIES FOR IMPORT
import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as spsl
import pandas as pd
import matplotlib.pylab as plt
import read_mpc

from openpyxl import load_workbook

#%% COLUMN HEADER VALUES INITIALIZATIONS

# Branches
F_BUS       = 0    # f, from bus number
T_BUS       = 1    # t, to bus number
BR_R        = 2    # r, resistance (p.u.)
BR_X        = 3    # x, reactance (p.u.)
BR_B        = 4    # b, total line charging susceptance (p.u.)
RATE_A      = 5    # rateA, MVA rating A (long term rating)
RATE_B      = 6    # rateB, MVA rating B (short term rating)
RATE_C      = 7    # rateC, MVA rating C (emergency rating)
TAP         = 8    # ratio, transformer off nominal turns ratio
SHIFT       = 9    # angle, transformer phase shift angle (degrees)
BR_STATUS   = 10   # initial branch status, 1 - in service, 0 - out of service
ANGMIN      = 11   # minimum angle difference, angle(Vf) - angle(Vt) (degrees)
ANGMAX      = 12   # maximum angle difference, angle(Vf) - angle(Vt) (degrees)

# included in power flow solution, not necessarily in input
PF          = 13   # real power injected at "from" bus end (MW)
QF          = 14   # reactive power injected at "from" bus end (MVAr)
PT          = 15   # real power injected at "to" bus end (MW)
QT          = 16   # reactive power injected at "to" bus end (MVAr)

# Buses
# Types
PQ      = 1
PV      = 2
REF     = 3
NONE    = 4

BUS_I       = 0    # bus number (1 to 29997)
BUS_TYPE    = 1    # bus type
PD          = 2    # Pd, real power demand (MW)
QD          = 3    # Qd, reactive power demand (MVAr)
GS          = 4    # Gs, shunt conductance (MW at V = 1.0 p.u.)
BS          = 5    # Bs, shunt susceptance (MVAr at V = 1.0 p.u.)
BUS_AREA    = 6    # area number, 1-100
VM          = 7    # Vm, voltage magnitude (p.u.)
VA          = 8    # Va, voltage angle (degrees)
BASE_KV     = 9    # baseKV, base voltage (kV)
ZONE        = 10   # zone, loss zone (1-999)
VMAX        = 11   # maxVm, maximum voltage magnitude (p.u.)
VMIN        = 12   # minVm, minimum voltage magnitude (p.u.)

# Generators
GEN_BUS     = 0    # bus number
PG          = 1    # Pg, real power output (MW)
QG          = 2    # Qg, reactive power output (MVAr)
QMAX        = 3    # Qmax, maximum reactive power output at Pmin (MVAr)
QMIN        = 4    # Qmin, minimum reactive power output at Pmin (MVAr)
VG          = 5    # Vg, voltage magnitude setpoint (p.u.)
MBASE       = 6    # mBase, total MVA base of this machine, defaults to baseMVA
GEN_STATUS  = 7    # status, 1 - machine in service, 0 - machine out of service
PMAX        = 8    # Pmax, maximum real power output (MW)
PMIN        = 9    # Pmin, minimum real power output (MW)
PC1         = 10   # Pc1, lower real power output of PQ capability curve (MW)
PC2         = 11   # Pc2, upper real power output of PQ capability curve (MW)
QC1MIN      = 12   # Qc1min, minimum reactive power output at Pc1 (MVAr)
QC1MAX      = 13   # Qc1max, maximum reactive power output at Pc1 (MVAr)
QC2MIN      = 14   # Qc2min, minimum reactive power output at Pc2 (MVAr)
QC2MAX      = 15   # Qc2max, maximum reactive power output at Pc2 (MVAr)
RAMP_AGC    = 16   # ramp rate for load following/AGC (MW/min)
RAMP_10     = 17   # ramp rate for 10 minute reserves (MW)
RAMP_30     = 18   # ramp rate for 30 minute reserves (MW)
RAMP_Q      = 19   # ramp rate for reactive power (2 sec timescale) (MVAr/min)
APF         = 20   # area participation factor

#%% CASE DATA INTIAILIZATIONS
# case 9 information
baseMVA = 100.0

bus = np.array([
	[1,	2,	0,	0,	0,	0,	1,	1.04,	-3.970866,	16.5,	1,	1.1,	0.9],
	[2,	3,	0,	0,	0,	0,	1,	1.025,	0,	18,	1,	1.1,	0.9],
	[3,	2,	0,	0,	0,	0,	1,	1.025,	-0.663341,	13.8,	1,	1.1,	0.9],
	[4,	1,	0,	0,	0,	0,	1,	1.0214855,	-6.196574,	230,	1,	1.1,	0.9],
	[5,	1,	125,	50,	0,	0,	1,	0.9951484,	-9.29143,	230,	1,	1.1,	0.9],
	[6,	1,	90,	30,	0,	0,	1,	1.0061533,	-5.324264,	230,	1,	1.1,	0.9],
	[7,	1,	0,	0,	0,	0,	1,	1.0138507,	-5.600873,	230,	1,	1.1,	0.9],
	[8,	1,	100,	35,	0,	0,	1,	1.0072369,	-6.787188,	230,	1,	1.1,	0.9],
	[9,	1,	0,	0,	0,	0,	1,	1.0276423,	-3.373753,	230,	1,	1.1,	0.9],
])

#bus = np.array([
#	[1,	2,	0,	    0,	    0,	0,	1,	1,	0,	230,	1,	1.1,	0.9],
#	[2,	1,	300,	98.61,	0,	0,	1,	1,	0,	230,	1,	1.1,    0.9],
#	[3,	2,	300,	98.61,	0,	0,	1,	1,	0,	230,	1,	1.1,    0.9],
#	[4,	3,	400,	131.47,	0,	0,	1,	1,	0,	230,	1,	1.1,    0.9],
#	[5,	2,	0,	    0,	    0,	0,	1,	1,	0,	230,	1,	1.1,	0.9],
#])

#gen = np.array([
#    [1, 71.63,   34.82, 300, -300, 1, 100, 1, 250, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#    [2, 162.28, 26.22, 300, -300, 1, 100, 1, 300, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#    [3, 85,  -2.61, 300, -300, 1, 100, 1, 270, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
#    ])

branch = np.array([
	[2,	7,	0,	     0.0625,	0,	0,	0,	0,	1,	0,	1,	0,	0,	162.2800,	26.2200,	-162.2800,	-10.1500],
	[4,	1,	0,	     0.0576,	0,	0,	0,	0,	1,	0,	1,	0,	0,	-71.6300,	-31.4400,	71.6300,	34.8200],
	[5,	4,	0.01,	0.068,	0.176,	0,	0,	0,	0,	0,	1,	0,	0,	-84.2300,	-32.6900,	85.0100,	20.0600],
	[6,	4,	0.017,	0.092,	0.158,	0,	0,	0,	0,	0,	1,	0,	0,	13.4700,	-27.1300,	-13.3800,	11.3800],
	[7,	5,	0.032,	0.161,	0.306,	0,	0,	0,	0,	0,	1,	0,	0,	41.3000,	-10.8600,	-40.7700,	-17.3100],
	[7,	8,	0.0085,	0.0576,	0.149,	0,	0,	0,	0,	0,	1,	0,	0,	37.6600,	-1.1900,	-37.5400,	-13.2000],
	[8,	9,	0.0119,	0.1008,	0.209,	0,	0,	0,	0,	0,	1,	0,	0,	-62.4600,	-21.8000,	62.9300,	4.1600],
	[9,	3,	0,	     0.0586,	0,	0,	0,	0,	1,	0,	1,	0,	0,	-85.0000,	6.6400,	85.0000,	-2.6100],
	[9,	6,	0.039,	0.1738,	0.358,	0,	0,	0,	0,	0,	1,	0,	0,	22.0700,	-10.8000,	-21.8600,	-25.3100]
])

#branch = np.array([
#	[1,	2,	0.00281,	0.0281,	0.00712,	400,  400,	400,  0,	0,	1,	-360,	360],
#	[1,	4,	0.00304,	0.0304,	0.00658,	0,	   0,	0,	  0,	0,	1,	-360,	360],
#	[1,	5,	0.00064,	0.0064,	0.03126,	0,	   0,	0,	  0,	0,	1,	-360,	360],
#	[2,	3,	0.00108,	0.0108,	0.01852,	0,	   0,	0,	  0,	0,	1,	-360,	360],
#	[3,	4,	0.00297,	0.0297,	0.00674,	0,	   0,	0,	  0,	0,	1,	-360,	360],
#	[4,	5,	0.00297,	0.0297,	0.00674,	240,  240,	240,  0,	0,	1,	-360,	360],
#])


# Sort branch to make sure it is the same as the read CSV data
#branch = np.array(sorted(branch, key=lambda x: (x[0], x[1])))

#%% Functions
##=======================================================
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
    
    f = branch[:, F_BUS]        # list of "from" buses
    t = branch[:, T_BUS]        # list of "to" buses
    
    # The -1 in the indices for the sparse are for zero-based indexing in python
    Cf = sparse.csr_matrix((np.ones(nl), (np.arange(nl), f-1)), (nl, nb))
    Ct = sparse.csr_matrix((np.ones(nl), (np.arange(nl), t-1)), (nl, nb))
    
    i = np.r_[range(nl), range(nl)]
    
    Yf = sparse.csr_matrix((np.r_[Yff, Yft], (i, np.r_[f-1, t-1])), (nl, nb))
    Yt = sparse.csr_matrix((np.r_[Ytf, Ytt], (i, np.r_[f-1, t-1])), (nl, nb))
    
    Ybus = Cf.T @ Yf + Ct.T @ Yt + sparse.csr_matrix((Ysh, (range(nb), range(nb))), (nb, nb))
    
    return Ybus, Yf, Yt
##=======================================================

#V = np.ones(bus.shape[0])
V = np.array([1.0242 - 0.6780j,
              1.0000 + 0.0000j,
              0.9971 - 0.0861j,
              1.0040 - 0.1082j,
              0.9761 - 0.1346j,
              0.9856 - 0.1286j,
              0.9890 - 0.1004j,
              0.9777 - 0.1315j,
              1.0032 - 0.0866j,
])

# make system admittance matrices
Ybus, Yf, Yt = makeYbus(baseMVA, bus, branch)

##=======================================================

nb = V.shape[0]       # Number of buses
nl = branch.shape[0]    # Number of lines
v_ones = np.ones(nl)    # number of 1's to be in the connection matrix, equal to number of lines

il = np.arange(nl)      # Row indices for branches, 0, 1, ... (nl-1)
ib = np.arange(nb)      # Row indices for buses, 0, 1, , ...(nb-1)

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
##=======================================================