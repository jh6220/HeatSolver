import matplotlib.pyplot as plt

from functions import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# future function inputs
m_dot = 3.0519
T_n2o = 300
Tg = 2.9193e+03
hg = 6.2961e+03

# #define geometry
n_channels = 30
r_channel = 0.0015
t_gap = 0.001
r_chamber = 0.0322
spacing = 0.0002
h_channel = r_channel + t_gap
t_wall = 2*h_channel

# define heat transfer constants
dm_dot = m_dot / n_channels
Cp_n2o = 2000
k_n2o = 0.08
mu_n2o = 1.37e-5
# rho_n2o = 770
# k = 20
k = [0.0161, 5.2269]
t = 1
h_n2o, Re = h_Re(r_channel, mu_n2o, dm_dot, Cp_n2o, k_n2o)

points, triangle, bc, pg = GetMesh(n_channels, r_channel, h_channel, t_wall, r_chamber, spacing, h_n2o, T_n2o, hg, Tg,showPlot=False)

# Instantiate the heat transfer problem object
hf = HeatFlow(points, triangle, bc, k, pg)
# Solve the problem iteratively (non-linear conductivity)
hf.Solve_itt()
# Plot temperature distribution
hf.PlotTemp()
# Plot mesh
hf.PlotMesh()

# Plot the temperature along the thrust chamber wall
ang_arr = np.linspace(0,np.pi/n_channels,100)
x = np.sin(ang_arr)*r_chamber
y = np.cos(ang_arr)*r_chamber
T = hf.SamplePoints(x,y)
plt.plot(ang_arr,T)
plt.show()

# Solve the transient problem and produce animated plot
hf.SolveTransient(1.5, 0.001)
hf.Animate(showAnim = False,saveFile = "AnimationCircle.gif")
