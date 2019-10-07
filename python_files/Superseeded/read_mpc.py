#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 15:51:13 2019

Script to read a MATPOWER case file.
This script will be useful to read the case data into numpy arrays.
Outputs are baseMVA, bus, gen, branch.

@author: achattopadhyay
"""

import re
import os
import numpy as np

FileName = 'WECC2470_int_idx.m'

##=============================================================================
def read_case(FileName):

    # Obtain the current directory
    # FileDirectory = os.path.dirname(os.path.realpath(__file__))
    
    #for file in os.listdir(FileDirectory):
    #    print (file)
    
    #for file in os.listdir(FileDirectory):
    #    if file.endswith(".m"):
    #        FileName = os.path.join(FileDirectory, file)
            
    FileObject = open(FileName, "r")
    lines = FileObject.readlines()
    
    # remove comments
    lines = [line.split('%')[0] for line in lines]
    
    # put everythin into a single string
    case = '\n'.join(lines)
    
    try:
        version = re.search(r"mpc.version\s*=\s*'(\S+)'", case).group(1)
        baseMVA = re.search(r'mpc.baseMVA\s*=\s*(\d*\.?\d*)', case).group(1)
        bus_str = re.search(r'mpc.bus\s*=\s*\[([-\d\se.;]+)\]', case).group(1)
        gen_str = re.search(r'mpc.gen\s*=\s*\[([-\d\se.;]+)\]', case).group(1)
        branch_str = re.search(r'mpc.branch\s*=\s*\[([-\d\se.;]+)\]', case).group(1)
        
    except:
        raise Exception('Failed to parse Matpower file{0}'.format(FileName))
       
    # Split the strings to make the bus, gen, and branch lists    
    bus_data = [d.split() for d in bus_str.strip(';\n\t ').split(';')]
    gen_data = [d.split() for d in gen_str.strip(';\n\t ').split(';')]
    branch_data = [d.split() for d in branch_str.strip(';\n\t ').split(';')]
    
    # Convert to float values
    baseMVA = float(baseMVA)
    bus = np.array(bus_data, dtype = np.float32)
    gen = np.array(gen_data, dtype = np.float32)
    branch = np.array(branch_data, dtype = np.float32)
    
    return bus, gen, branch, baseMVA
##=============================================================================