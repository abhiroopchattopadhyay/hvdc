#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 09:39:40 2019

The tracking state estimation program initially implemented in MATLAB 
ported to Python.

@author: achattopadhyay
"""
#%% Import section
# Import Python modules
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import timeit

# Import user made functions
import matpower_scripts as mpc_f

from indices import F_BUS, T_BUS
from indices import PQ, PV, REF, NONE
from indices import BUS_I, BUS_TYPE, PD, QD, GS, BS, VM, VA
from indices import GEN_BUS, PG, QG, QMAX, QMIN, VG, MBASE, GEN_STATUS, PMAX, PMIN

#%% Initialize necessary empty structures
se_iterCount = []
se_success = []
se_etime = []
Pconv = []
Pinv = []
Pconv_error = []
Pinv_error = []
Pconv_pct_error = []
Pinv_pct_error = []
line_flows = []

#%% Initialize state estimation parameters
#max_iter = 3
tol = 1e-3
stdev = 0.02
poll_freq = 120  # Polling freq. in sec. Other choices include 120, 60, 30, 15

#%% Boolean conditions for testing threats and adding errors
test_threat = True
add_error = False

## Case information
casename = 'WECC2470_int_idx.m'
bus, gen, branch, baseMVA = mpc_f.read_case(casename)
branch = np.array(sorted(branch, key=lambda x: (x[0], x[1])))
Ybus, Yf, Yt = mpc_f.makeYbus(baseMVA, bus, branch)

nb = bus.shape[0]
nl = branch.shape[0]
ng = gen.shape[0]

converter_bus = 333 - 1
inverter_bus = 1124 - 1

f = branch[:, F_BUS].astype(int) - 1
t = branch[:, T_BUS].astype(int) - 1

#%% Read .csv files for load flow, PTDF values, and line flows

# Find the current program directory
file_directory = os.path.dirname(os.path.realpath(__file__))

# .csv file name of load flow results
lf_result_file = 'loadFlowResults.csv'

# Read the .csv file for the LF results 
lf_results = pd.read_csv(str(file_directory) +('/') + lf_result_file, sep=',')

# Extract the relevant DC values
PW_Idc = np.array(lf_results.loc[ :, 'DC setpt'])
PW_conv_inj = np.array(lf_results.loc[ :, 'PW Conv Inj'])
PW_inv_inj = np.array(lf_results.loc[ :, 'PW Inv Inj'])
time_stamp = np.array(lf_results.loc[ :, 'time'])

# Read the .csv file for PTDF values
ptdf_file = 'PTDF_2470bus.csv'
df1 = pd.read_csv(str(file_directory) +('/') + ptdf_file, sep=',', skiprows = [0])
df1 = df1.sort_values(by = ['From Number', 'To Number'])
ptdf_data = np.asarray(df1)

# Extract the necessary "from" and "to" PTDF values
Hf = ptdf_data[ :, 0]
Ht = ptdf_data[:, 1]

# Folder location of line flow csv files
csvFolder = "Extended_csv_files_2500"
full_directory = os.path.join(file_directory, csvFolder)
flow_files = os.listdir(full_directory)
flow_files.sort()

# Read all the line flow .csv files
for k in range(0, len(flow_files)):
     file = flow_files[k]
     file_to_read = full_directory + '/' + file
     df = mpc_f.read_line_flows(file_to_read)
     line_flows.append(df)
line_flows = np.asarray(line_flows)

#%% Trim data 
n = np.arange(0, 80, int(poll_freq/15))

PW_Idc = PW_Idc[n]
PW_conv_inj = PW_conv_inj[n]
PW_inv_inj = PW_inv_inj[n]
line_flows = line_flows[n, :, :]
time_stamp = time_stamp[n]

#%% Define state estimation function as a function of it_max and ptdf threshold
def se_variation(it_max, ptdf_threshold):
    
    #%% Prioritization of line flows
    
    # Find the indices of the branches that have PTDF value higher than threshold
    from_lines = [i for i in range(len(Hf)) if abs(Hf[i]) >= ptdf_threshold]
    to_lines = [i for i in range(len(Ht)) if abs(Ht[i]) >= ptdf_threshold]
    
    #%% Perform state estimation for every time scan
    
    for k in range(0, len(line_flows)):
        print('=========================================================')
        print('Running S.E. at time scan = {0} min'.format(time_stamp[k]))
        print('DC current on the HVDC line = {0} A'.format(PW_Idc[k]))
    
        # k = 0 is for the initial state
        if k == 0:          
            # For estimating initial state, use a flat start
            V0 = np.ones(nb)
            # Define the line flow measurements using all measurements
            Pf = line_flows[k, :, 0] / baseMVA
            Pt = line_flows[k, :, 1] / baseMVA
            Qf = line_flows[k, :, 2] / baseMVA
            Qt = line_flows[k, :, 3] / baseMVA
            
            if add_error:
                Pf = Pf + np.random.normal(0, stdev, nl)
                Pt = Pt + np.random.normal(0, stdev, nl)
                Qf = Qf + np.random.normal(0, stdev, nl)
                Qt = Qt + np.random.normal(0, stdev, nl)
            
            # Run state estimation
            V_est, success, iter_count, e_time = mpc_f.run_se(baseMVA, bus, gen, branch, \
                                            Pf, Pt, Qf, Qt, it_max, V0, stdev, tol)
            
        else:
            # For every other state estimate, use the previous estimate as starting point
            V0 = V_est
            # Compute estimated measurement based on previous estimate of state
            Sfe = V0[f] * np.conj(Yf @ V0)
            Ste = V0[t] * np.conj(Yt @ V0)
            
            # Set Pf, Pt, Qf, and Qt from estimated values computed
            Pf = np.real(Sfe)
            Pt = np.real(Ste)
            Qf = np.imag(Sfe)
            Qt = np.imag(Ste)
    
            # Replace the flows of the prioritized lines from the time scan .csv
            Pf[from_lines] = line_flows[k, from_lines, 0] / baseMVA
            Pt[to_lines] = line_flows[k, to_lines, 1] / baseMVA
            Qf[from_lines] = line_flows[k, from_lines, 2] / baseMVA
            Qt[to_lines] = line_flows[k, to_lines, 3] / baseMVA
    
            if add_error:
                Pf[from_lines] = Pf[from_lines] + np.random.normal(0, stdev, len(from_lines))
                Pt[to_lines] = Pt[to_lines] + np.random.normal(0, stdev, len(to_lines))
                Qf[from_lines] = Qf[from_lines] + np.random.normal(0, stdev, len(from_lines))
                Qt[to_lines] = Qt[to_lines] + np.random.normal(0, stdev, len(to_lines))
            
            # Run state estimation
            V_est, success, iter_count, e_time = mpc_f.run_se(baseMVA, bus, gen, branch, \
                                        Pf, Pt, Qf, Qt, it_max, V0, stdev, tol)
    
        # Compute the bus injections
        S_inj = V_est * np.conj( Ybus @ V_est)
        
        # Store all relevant output values 
        se_iterCount.append(iter_count)
        se_success.append(success)
        se_etime.append(e_time)
        
        # Store the converter and inverter estimate
        Pconv.append( - np.real(S_inj[converter_bus]) * baseMVA)
        Pinv.append( - np.real(S_inj[inverter_bus]) * baseMVA)
        
        # Store the error in converter and inverter estimate
        Pconv_error.append(abs(Pconv[-1] - PW_conv_inj[k]))
        Pinv_error.append(abs(Pinv[-1] - PW_inv_inj[k]))
        
        # Store the error percent in converter and inverter estimate
        Pconv_pct_error.append(abs(Pconv_error[-1]/ PW_conv_inj[k]) * 100)
        Pinv_pct_error.append(abs(Pinv_error[-1]/ PW_inv_inj[k]) * 100)
        
        print('Estimate at converter = {0:4.0f} MW'. format(Pconv[-1]))
        print('Estimate at inverter = {0:4.0f} MW'. format(Pinv[-1]))
    
    #Pconv_pct_error = np.asarray(Pconv_pct_error)
    
    Pconv_error_avg = sum(Pconv)/len(Pconv)
    
    return Pconv_error_avg
    #return np.asarray(Pconv)

#%% Variation study
it_max = np.arange(1, 11, 1)
ptdf_low = np.arange(0, 55, 5)

conv_error = np.empty((len(it_max), len(ptdf_low)))

start = timeit.timeit()

for m in range(len(it_max)):
    
    for n in range(len(ptdf_low)):
        
        conv_error[m, n] = se_variation(it_max[m], ptdf_low[n])
        
end = timeit.timeit()

print (end - start)

#%% Plot 3D results
hf = plt.figure()
ha = hf.add_subplot(111, projection='3d')        
X, Y = np.meshgrid(it_max, ptdf_low)
ha.plot_surface(X, Y, conv_error)
plt.show()