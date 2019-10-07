import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

#%% Data to create animation

# File directory
file_directory = os.path.dirname(os.path.realpath(__file__))
# .csv file name of load flow results
result_file = 'animation_test_threat.csv'

# Read the .csv file for the LF results 
data = pd.read_csv(str(file_directory) +('/') + result_file, sep=',')

# Extract the relevant DC values
time_stamp = np.array(data.loc[ :, 'Time_stamp'])
PW_conv_inj = np.array(data.loc[ :, 'PW_conv_inj'])
Pconv = np.array(data.loc[:, 'P_conv'])
se_time = np.array(data.loc[:, 'se_time'])
Pconv_error = np.array(data.loc[:, 'Pconv_error'])
Pconv_pct_error = np.array(data.loc[:, 'Pconv_pct_error'])

#%% Video Properties
poll_freq = 15

frame_rate = 1/poll_freq
frame_count = int(1200 * frame_rate)

Writer = animation.writers['ffmpeg']
writer = Writer(fps = frame_rate, metadata = dict(artist='Me'), bitrate = 1800)

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
ax1.set_ylim(Pconv[0], Pconv[-1])
ax1.grid(True)

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

def animate(i):

    # Plot the load flow results
    #p1 = sns.lineplot(x = time_stamp, y = PW_conv_inj, color = 'b', ax = ax1)
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
    
ani = matplotlib.animation.FuncAnimation(fig, animate, frames = frame_count, repeat=False)
ani.save('test_file.mp4', writer=writer)