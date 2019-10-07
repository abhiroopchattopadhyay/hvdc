#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 15:09:35 2019

@author: achattopadhyay
"""

import numpy as np
import re

# Script for reading text data

with open('Hdata.txt', "r") as f:
    list = [[float(num) for num in line.split(',')] for line in f]

data = np.array(l)

#rows = [int(i[0]) for i in list]
#columns = [int(i[1]) for i in list]
#v = [float(i[2]) for i in list]

#file = open('Hdata.txt',"r" )
#lines = file.readlines()
#
#l = [num for num in line.split for line in lines]

#case = '\n'.join(lines)

#blankRows = re.finditer('\t\t\n\n', case)

#version = re.search(r"mpc.version\s*=\s*'(\S+)'", case).group(1)

#bus_data = [d.split() for d in bus_str.strip(';\n\t ').split(';')]
#gen_data = [d.split() for d in gen_str.strip(';\n\t ').split(';')]
#branch_data = [d.split() for d in branch_str.strip(';\n\t ').split(';')]


#lines = [line for line in file.readlines() if line.strip()]

#FileObject = open(FileName, "r")
#H = [d.split() for d in ]

