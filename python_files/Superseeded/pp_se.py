#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 13:21:10 2019

@author: achattopadhyay

State estimation using pandapower

See the link below for example
https://github.com/e2nIEE/pandapower/blob/master/tutorials/state_estimation.ipynb
"""

#%%############################################################################
# Import all the necessary modules

import pandapower as pp
import numpy as np
import pandas

#%%############################################################################
# Case information for case 9, from MATPOWER

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

gen = np.array([
    [1, 71.63,   34.82, 300, -300, 1, 100, 1, 250, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [2, 162.28, 26.22, 300, -300, 1, 100, 1, 300, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [3, 85,  -2.61, 300, -300, 1, 100, 1, 270, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
])

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

# Sort the branches
branch = np.array(sorted(branch, key=lambda x: (x[0], x[1])))
#%%############################################################################
# The relevant columns of the branch and bus arrays
BASE_KV     = 9    # baseKV, base voltage (kV)
BUS_TYPE    = 1    # bus type

F_BUS       = 0    # f, from bus number
T_BUS       = 1    # t, to bus number
BR_R        = 2    # r, resistance (p.u.)
BR_X        = 3    # x, reactance (p.u.)
BR_B        = 4    # b, total line charging susceptance (p.u.)
BR_STATUS   = 10   # initial branch status, 1 - in service, 0 - out of service

#%%############################################################################
# Create the network for the above given case information

# Find number of buses, branches, and generators
nb, _ = bus.shape
ng, _ = gen.shape
nl, _ = branch.shape

busList =[]
lineList = []

# Create blank network
net = pp.create_empty_network()

# Add the buses in the network
for i in range(nb):
    busKV = bus[i, BASE_KV]
    busIndex = i + 1
    busName = "bus" + str(busIndex)
    busList.append(busName)
    
    b1 = pp.create_bus(net, name = busName, vn_kv = busKV, index=busIndex)

# Assign slack bus to the network
bus_types = bus[:, BUS_TYPE]
refIndex =  np.where((bus_types==3))
refIndex = np.asscalar(refIndex[0]) + 1
pp.create_ext_grid(net, refIndex, name = "slack")

# Add the lines to the network
for i in range(nl):
    f_Bus = int(branch[i, F_BUS])
    t_Bus = int(branch[i, T_BUS])
    length = 1
    lineIndex = i + 1
    lineName = "line" + str(lineIndex)
    lineList.append(lineIndex)
    
    r = branch[i, BR_R]
    x = branch[i, BR_X]
    c = branch[i, BR_B]/(2 * np.pi * 60 * 1e-9)
    Imax = np.inf
    
    l1 = pp.create_line_from_parameters(net, f_Bus, t_Bus, length,\
                        index = lineIndex, r_ohm_per_km = r, x_ohm_per_km = x,\
                        c_nf_per_km = c, name = lineName, max_i_ka = Imax)
#%%############################################################################
# Read measurement from .csv file

# File name of measurements
file = '9bus+100.csv'

# This function reads the csv file of line flows
df = pandas.read_csv(file, sep=',', skiprows = [0])

# Sort the line flows by "From Bus" followed by "To Bus"
line_flows = df.sort_values(by=['From Number', 'To Number'])

#%%############################################################################
# Add each measurement to the network
nr, _ = df.shape

# The measurement standard deviations
stDevPf = 0.02
stDevPt = 0.02
stDevQf = 0.02
stDevQt = 0.02

# For every row in the .csv, add the Pf, Pt, Qf, and Qt measurements
for i in range(nr):

    Pf = line_flows.iloc[i][0]
    Pt = line_flows.iloc[i][1]
    Qf = line_flows.iloc[i][2]
    Qt = line_flows.iloc[i][3]
    fromBus = int(line_flows.iloc[i][4])
    toBus = int(line_flows.iloc[i][5])

    lineIndex = lineList[i]
    
    #print(lineIndex, Pf, fromBus)
    pp.create_measurement(net, "p", "line", Pf, stDevPf, \
                      element = lineIndex, side=fromBus)
    #print(lineIndex, Pt, toBus)
    pp.create_measurement(net, "p", "line", Pt, stDevPt, \
                      element = lineIndex, side=toBus)
    #print(lineIndex, Qf, fromBus)
    pp.create_measurement(net, "q", "line", Qf, stDevQf, \
                      element = lineIndex, side=fromBus)
    #print(lineIndex, Qt, toBus)
    pp.create_measurement(net, "q", "line", Qt, stDevQt, \
                      element = lineIndex, side=toBus)

#%%############################################################################
# Perform state estimation

success = pp.estimation.state_estimation.estimate(net, init = "flat")

#V, delta = net.res_bus_est.vm_pu, net.res_bus_est.va_degree