import matplotlib.pyplot as plt

from functions import *
import N2O
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# future function inputs
m_dot = 3.950352631578947
T_n2o = 300
P_n2o = 6000
Tg = 2625.8519651374
hg = 7214.21926430911

# define geometry
n_channels = 50
t_gap_bot = 0.001
t_gap_top = 0.001
r_chamber = 0.0539473786567391
spacing = 0.0001
ratio_channel = 1
t_gap_side = 0.001
ang_channel = ratio_channel*((r_chamber+t_gap_bot)*np.pi/n_channels - t_gap_side/2)/((r_chamber+t_gap_bot)*np.pi/n_channels)
a_channel = 0.0015
A = np.pi * ((r_chamber + t_gap_bot + a_channel) ** 2 - (r_chamber + t_gap_bot) ** 2) * ang_channel / n_channels
P = 2 * a_channel + (2 * np.pi * ang_channel / n_channels) * (r_chamber + t_gap_bot + a_channel + r_chamber + t_gap_bot)
r_channel = 2 * A / P
# print(r_channel)

# define heat transfer constants
dm_dot = m_dot / n_channels
# Cp_n2o = 2000
# k_n2o = 0.08
# mu_n2o = 1.37e-5
Cp_n2o, mu_n2o, k_n2o, rho_n2o = N2O.GetData(T_n2o, P_n2o)

# k = 20
k = [0.0161, 5.2269]
t = 1
h_n2o, Re = h_Re(r_channel, mu_n2o, dm_dot, Cp_n2o, k_n2o)

print(h_n2o)

h_channel = 0.0015
t_wall = 0.001

points, triangle, bc, pg = GetMeshS(n_channels, a_channel, ang_channel, t_gap_bot, t_gap_top, r_chamber, spacing, h_n2o,
                                    T_n2o, hg, Tg, showPlot=False)

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
hf.Animate(showAnim = False,saveFile = "AnimationSquare.gif")



# hf = HeatFlow(points, triangle, bc, k, pg)
# # hf.Solve_itt()
# hf.Solve()
#
# # area = hf.area
# # plt.hist(area)
# # print(area[area<10**-10])
#
# hf.PlotTemp()
# #
# # hf.SolveTransient(1.5, 0.001)
#
# # hf.Animate()
#
#
# # hf.PlotTemp()
# # print(hf.MaxTemp())
#
# hf.PlotMesh()
#
# # ang_arr = np.linspace(0,np.pi/n_channels,100)
# # x = np.sin(ang_arr)*r_chamber
# # y = np.cos(ang_arr)*r_chamber
# # T = hf.SamplePoints(x,y)
# # plt.plot(ang_arr,T)
# # plt.show()


