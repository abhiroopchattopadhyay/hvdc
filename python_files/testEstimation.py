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
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

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
max_iter = 3
tol = 1e-3
stdev = 0.02
poll_freq = 15  # Polling freq. in sec. Other choices include 120, 60, 30, 15

#%% Boolean conditions for testing threats and adding errors
test_threat = False
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

#%% Prioritization of line flows
ptdf_low = 10

# Find the indices of the branches that have PTDF value higher than threshold
from_lines = [i for i in range(len(Hf)) if abs(Hf[i]) >= ptdf_low]
to_lines = [i for i in range(len(Ht)) if abs(Ht[i]) >= ptdf_low]

#%% Threat simulation modification here
# If threat is being simulated, then modify the expected and actual profile
if test_threat:
    # Define the start and end of the genuine order and the corrupted order
    P_export_start_index = 26
    P_export_end_index = 49
    P_threat_end_index = 60
    
    print('Initial flow on line = {0:4.0f} MW'. format(PW_conv_inj[P_export_start_index]))
    print('Final flow after power order = {0:4.0f} MW'. format(PW_conv_inj[P_export_end_index]))
    print('Final flow corrupted to = {0:4.0f} MW'. format(PW_conv_inj[P_threat_end_index]))
    
    PW_Idc[: P_export_start_index] = PW_Idc[P_export_start_index]
    PW_Idc[P_export_end_index :] = PW_Idc[P_export_end_index]
    
    PW_conv_inj[: P_export_start_index] = PW_conv_inj[P_export_start_index]
    PW_conv_inj[P_export_end_index :] = PW_conv_inj[P_export_end_index]
    
    PW_inv_inj[: P_export_start_index] = PW_inv_inj[P_export_start_index]
    PW_inv_inj[P_export_end_index :] = PW_inv_inj[P_export_end_index]
        
    # Modify the line flows to reflect the corrupted order
    line_flows[: P_export_start_index] = line_flows[P_export_start_index]
    line_flows[P_threat_end_index :] = line_flows[P_threat_end_index]

# If add_error, then add uniform error with deviation of 2% and 0.0 mean
#if add_error:
#    PW_conv_inj = PW_conv_inj + np.random.normal(0, 0.02 * PW_conv_inj, PW_conv_inj.shape)
#    PW_inv_inj = PW_inv_inj + np.random.normal(0, 0.02 * PW_inv_inj, PW_inv_inj.shape)

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
                                        Pf, Pt, Qf, Qt, max_iter, V0, stdev, tol)
        
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
                                    Pf, Pt, Qf, Qt, max_iter, V0, stdev, tol)

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

Pconv_pct_error = np.asarray(Pconv_pct_error)

#%% Plot the results

fig, ax = plt.subplots()
ax.scatter(time_stamp, Pconv_error, label='Error at Converter Bus')
ax.scatter(time_stamp, Pinv_error, label='Error at Inverter Bus')
plt.grid(True)
plt.xlabel('Time [min]')
plt.ylabel('Error in the State Estimate [MW]')
leg = ax.legend()

#%% Create Animation
#frame_rate = 1/poll_freq
#frame_count = int(1200 * frame_rate)
frame_count = 80 # 15 frames total 
frame_rate = 5.3  # 1 per sec => 15 sec video

Writer = animation.writers['ffmpeg']
writer = Writer(fps = frame_rate, metadata = dict(artist='Me'), bitrate = 1800)

#if test_threat:
#    P_init = Pconv[P_export_start_index]
#    P_final = Pconv[P_export_end_index]
#    P_threat = Pconv[P_threat_end_index]
#    text_str = ('Initial Power Flow :     {:4.0f} MW'. format(P_init),
#                'Final Power Flow :       {:4.0f} MW'. format(P_final),
#                'Compromised Final Flow : {:4.0f} MW'. format(P_threat))
#else:
#    text_str = ('Initial Power Flow :     {:4.0f} MW'. format(PW_conv_inj[0]),
#                'Final Power Flow :       {:4.0f} MW'. format(PW_conv_inj[-1]))
    
# Create subplots for estimate and error in estimate
fig = plt.figure(figsize = (15,15))
gs = fig.add_gridspec(2,2)
ax1 = fig.add_subplot(gs[0, :])
ax2 = fig.add_subplot(gs[1, 0])
ax3 = fig.add_subplot(gs[1, 1])
fig.suptitle('Results at Converter End', fontsize = 20)

ax1.set_xlabel('Time [min]', fontsize = 20)
ax1.set_ylabel('Injection Estimate [MW]', fontsize = 20, wrap = True)
ax1.set_xlim(time_stamp[0], time_stamp[-1])
ax1.set_ylim(0, 3500)
ax1.grid(True)
#ax1.text(0.1, 0.1, text_str, fontsize = 15)

ax2.set_xlabel('Time [min]', fontsize = 20)
ax2.set_ylabel('Estimate Error [MW]', fontsize = 20)
ax2.set_xlim(time_stamp[0], time_stamp[-1])
ax2.set_ylim(0, Pconv_error[-1])
ax2.grid(True)

ax3.set_xlabel('Time [min]', fontsize = 20)
ax3.set_ylabel('Estimate Error [%]', fontsize = 20)
ax3.set_xlim(time_stamp[0], time_stamp[-1])
ax3.set_ylim(0, np.nanmax(Pconv_pct_error[Pconv_pct_error != np.inf]))
ax3.grid(True)

def init():
    # do nothing
    pass

def animate(i):    
    # Plot the load flow results
    p1 = sns.scatterplot(x = time_stamp[: int(i+1)], y = PW_conv_inj[: int(i+1)], \
            label = 'Expected Power Profile' if i==0 else "", color = 'b', ax = ax1)
    # Plot converter estimate
    p2 = sns.scatterplot(x = time_stamp[: int(i+1)], y = Pconv[: int(i+1)], \
            label = 'Estimate' if i==0 else "", color = 'r', ax = ax1)
    # Plot converter estimate error
    p3 = sns.scatterplot(x = time_stamp[: int(i+1)], y = Pconv_error[: int(i+1)], \
            color = 'y', ax = ax2)
    # Plot percentage error in converter
    p4 = sns.scatterplot(x = time_stamp[: int(i+1)], y = Pconv_pct_error[: int(i+1)],\
            color = 'g', ax = ax3)

    p1.tick_params(labelsize = 20)
    p2.tick_params(labelsize = 20)
    p3.tick_params(labelsize = 20)
    p4.tick_params(labelsize = 20)
    
    p1.legend(loc = 'upper left')
    
#ani = matplotlib.animation.FuncAnimation(fig, animate, frames = frame_count,\
#                                         init_func = init, repeat=False)
ani = matplotlib.animation.FuncAnimation(fig, animate, frames = frame_count,\
                                         init_func = init, repeat=False)

ani.save('test_file.mp4', writer=writer)