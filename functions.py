import numpy as np
import matplotlib.pyplot as plt
from scipy import spatial
from shapely import geometry
from HeatFlow import HeatFlow
from N2O import *


def GetBC(boundary_cc, boundary_tc, boundary_is, triangle, neighbors, boundary, h_cc, T_cc, h_tc, T_tc):
    # define (N_surface_nodes x 6) matrix containing boundary conditions, each row represents edge on surface between
    # two nodes (vertexes)
    #   first and second column are indexes of the two nodes
    #   third column is the index of the triangle this edge is part of
    #   fourth column is the type of boundary condition (1-convection_channel, 2-convection_trust_chamber, 3-insulation)
    #   fifth column is the convection coefficient value of zero if insulation
    #   sixth column is the fluid temperature value of zero if insulation

    bc = np.zeros((len(boundary), 6), int)
    i_bc = 0
    for i in range(triangle.shape[0]):
        # loop through all triangles
        for j in range(3):
            # loop though each side of the triangle
            if neighbors[i, j] == -1:
                # define the indexes of the two vectexes of the currect side of the triangle
                j1 = (j + 1) % 3
                j2 = (j + 2) % 3

                # populate the row of the bc matrix
                bc[i_bc, [0, 1]] = np.sort(triangle[i, [j1, j2]])
                bc[i_bc, 2] = i
                if all(np.isin(triangle[i, [j1, j2]], boundary_cc)):
                    bc[i_bc, 3] = 1
                    bc[i_bc, 4] = h_cc
                    bc[i_bc, 5] = T_cc
                elif all(np.isin(triangle[i, [j1, j2]], boundary_tc)):
                    bc[i_bc, 3] = 2
                    bc[i_bc, 4] = h_tc
                    bc[i_bc, 5] = T_tc
                elif all(np.isin(triangle[i, [j1, j2]], boundary_is)):
                    bc[i_bc, 3] = 3
                else:
                    print("boundary not found in any list")

                i_bc += 1

    return bc


def GetMesh(n_channels, r_channel, h_channel, t_wall, r_chamber, spacing, h_cc, T_cc, h_tc, T_tc, showPlot=False):
    ang = 2 * np.pi / n_channels
    ang_arr_top = np.linspace(ang / 2, 0, int(np.ceil((r_chamber + t_wall) * ang / (2 * spacing))) + 1)
    top_points = np.zeros((len(ang_arr_top), 2))
    top_points[:, 0] = (r_chamber + t_wall) * np.sin(ang_arr_top)
    top_points[:, 1] = (r_chamber + t_wall) * np.cos(ang_arr_top)

    ang_arr_bot = np.linspace(0, ang / 2, int(np.ceil((r_chamber) * ang / (2 * spacing))) + 1)
    bot_points = np.zeros((len(ang_arr_bot), 2))
    bot_points[:, 0] = r_chamber * np.sin(ang_arr_bot)
    bot_points[:, 1] = r_chamber * np.cos(ang_arr_bot)

    side_r_arr = np.linspace(r_chamber, r_chamber + t_wall, int(np.ceil(t_wall / spacing)) + 1)
    side_points = np.zeros((len(side_r_arr) - 2, 2))
    side_points[:, 0] = np.sin(ang / 2) * side_r_arr[1:-1]
    side_points[:, 1] = np.cos(ang / 2) * side_r_arr[1:-1]

    edge_out = np.concatenate([bot_points, side_points, top_points])

    ang_arr_in = np.linspace(0, np.pi, np.shape(edge_out)[0])
    edge_in = np.zeros((np.shape(edge_out)[0], 2))
    edge_in[:, 0] = np.sin(ang_arr_in) * r_channel
    edge_in[:, 1] = -np.cos(ang_arr_in) * r_channel + r_chamber + h_channel

    points = np.transpose(np.array([[], []]))
    boundary_cc = []
    boundary_tc = []
    boundary_is = []
    for i in range(np.shape(edge_in)[0]):
        length = np.linalg.norm(edge_in[i, :] - edge_out[i, :])
        scaler = np.reshape(np.linspace(0, 1, int(np.ceil(length / spacing)) + 1) ** 1,
                            (int(np.ceil(length / spacing)) + 1, 1))  # power of 1.4 seems to work well
        radial = edge_in[i, :] + scaler * (edge_out[i, :] - edge_in[i, :])
        points = np.append(points, radial, axis=0)

        if (i in [0, np.shape(edge_in)[0] - 1]):
            boundary_is += list(range(points.shape[0] - len(scaler), points.shape[0]))

        boundary_cc.append(points.shape[0] - len(scaler))

        if (i < len(bot_points) - 1):
            boundary_tc.append(points.shape[0] - 1)
        elif (i == len(bot_points) - 1):
            boundary_tc.append(points.shape[0] - 1)
            boundary_is.append(points.shape[0] - 1)
        else:
            boundary_is.append(points.shape[0] - 1)

    boundary = np.unique(np.array(boundary_cc + boundary_tc + boundary_is))
    triang = spatial.Delaunay(points)
    outside_triangles = []
    p = [(x, y) for x, y in np.concatenate((edge_in, np.flip(edge_out, axis=0)), axis=0)]
    pg = geometry.Polygon(p)

    for i in range(triang.simplices.shape[0]):
        if not (pg.contains(geometry.Point(np.mean(points[triang.simplices[i, :], 0]),
                                           np.mean(points[triang.simplices[i, :], 1])))):
            outside_triangles.append(i)

    triangle = np.delete(triang.simplices, outside_triangles, axis=0)
    neighbors = np.delete(triang.neighbors, outside_triangles, axis=0)
    neighbors[np.isin(neighbors, outside_triangles)] = -1

    # define boundary conditions 1-convection_channel, 2-convection_trust_chamber, 3-insulation
    bc = GetBC(boundary_cc, boundary_tc, boundary_is, triangle, neighbors, boundary, h_cc, T_cc, h_tc, T_tc)

    if showPlot:
        plt.axis('equal')
        plt.triplot(points[:, 0], points[:, 1], triangle)
        c = ["g", "b", "r", "k"]
        for i in range(bc.shape[0]):
            x = [points[bc[i, 0], 0], points[bc[i, 1], 0]]
            y = [points[bc[i, 0], 1], points[bc[i, 1], 1]]
            plt.plot(x, y, color=c[bc[i, 3]])

    return points, triangle, bc, pg


# def GetMeshSquare(n, showPlot=False):
#     points = np.zeros([n ** 2, 2])
#     triangle = np.zeros([2 * (n - 1) ** 2, 3], int)
#     index = 0
#     for i in range(n):
#         for j in range(n):
#             points[index, :] = [j, i]
#             index += 1
#
#     points = points / (n - 1)
#
#     for i in range(n - 1):
#         for j in range(n - 1):
#             for k in range(2):
#                 if (k == 0):
#                     triangle[2 * ((n - 1) * i + j) + k, 0] = n * i + j
#                     triangle[2 * ((n - 1) * i + j) + k, 1] = n * i + j + 1
#                     triangle[2 * ((n - 1) * i + j) + k, 2] = n * i + j + n
#                 else:
#                     triangle[2 * ((n - 1) * i + j) + k, 0] = n * i + j + 1
#                     triangle[2 * ((n - 1) * i + j) + k, 1] = n * i + j + n + 1
#                     triangle[2 * ((n - 1) * i + j) + k, 2] = n * i + j + n
#
#     boundary = np.array(list(range(n)) + list(range(0, n * n, n)) + list(range(n - 1, n * n + n - 1, n)) + list(range(n * (n - 1), n * n)), int)
#     boundary = np.unique(boundary)
#     boundary_in = np.array(list(range(0, n * n, n)), int)
#     boundary_out = np.array(list(range(n - 1, n * n + n - 1, n)), int)
#     boundary_is = np.array(list(range(n)) + list(range(n * (n - 1), n * n)), int)
#
#     # bc = GetBC(boundary_in, boundary_out, boundary_is, triangle, points, boundary)
#     bc = GetBC(boundary_in, boundary_out, boundary_is, triangle, points, boundary)
#
#     if (showPlot):
#         plt.axis('equal')
#         plt.triplot(points[:, 0], points[:, 1], triangle)
#         c = ["g", "b", "r", "k"]
#         for i in range(bc.shape[0]):
#             x = [points[bc[i, 0], 0], points[bc[i, 1], 0]]
#             y = [points[bc[i, 0], 1], points[bc[i, 1], 1]]
#             plt.plot(x, y, color=c[bc[i, 3]])
#
#     return points, triangle, bc


def GetMeshS(n_channels, a_channel, ang_channel, t_gap_bot, t_gap_top, r_chamber, spacing, h_cc, T_cc, h_tc, T_tc,
             showPlot=False):
    # Define mesh and boundary conditions of the wall section with square cooling channel

    # angle of the trust chamber wall section
    ang = np.pi / n_channels
    ang_channel = ang_channel * ang

    # Define points - 2Darray (Nx2) that contains all of the vertexes
    r_out_chamber = r_chamber + t_gap_bot + t_gap_top + a_channel  # outer diameter of the trust chamber wall (above the channel there are 3*t_gap of the wall)
    N_channel = int(np.ceil(
        ang_channel * r_out_chamber / spacing) + 1)  # number of points in tangential direction in the section of the wall with the cooling channel
    N_wall = int(np.ceil((ang - ang_channel) * r_out_chamber / spacing) + 1)  # number of points in tangential direction in the section of the wall without the cooling channel

    # array of angle along which the the points in the mesh will be defined for the channel and wall part
    ang_arr_channel = np.expand_dims(np.linspace(0, ang_channel, N_channel), axis=(0, 2))
    ang_arr_wall = np.expand_dims(np.linspace(ang_channel, ang, N_wall), axis=(0, 2))

    boundary_cc = []  # array of indexes of points with cooling channel boundary condition
    boundary_tc = []  # array of indexes of points with trust chamber boundary condition
    boundary_is = []  # array of indexes of points with insulation boundary condition
    polygon = []  # ordered list of indexes of points that form the polygon defining the boundary

    # array of radius's for the bottom part of the cooling channel part
    scaler_bot = np.expand_dims(np.linspace(r_chamber, r_chamber + t_gap_bot, int(np.ceil(t_gap_bot / spacing) + 1)),
                                axis=(1, 2))
    # array of radius's for the top part of the cooling channel part
    scaler_top = np.expand_dims(
        np.linspace(r_out_chamber - t_gap_top, r_out_chamber, int(np.ceil(t_gap_top / spacing) + 1)), axis=(1, 2))
    # array of radius's for the wall part
    scalar_wall = np.expand_dims(
        np.linspace(r_chamber, r_out_chamber, int(np.ceil((r_out_chamber - r_chamber) / spacing) + 1)), axis=(1, 2))

    # define points and BC for bottom piece of the wall bellow the cooling channel
    points_bot = np.concatenate((np.sin(ang_arr_channel), np.cos(ang_arr_channel)), axis=2) * scaler_bot
    points_bot = points_bot.reshape(points_bot.shape[0] * points_bot.shape[1], 2)
    boundary_tc += list(range(ang_arr_channel.shape[1]))
    boundary_is += list(range(0, points_bot.shape[0], ang_arr_channel.shape[1]))
    boundary_cc += list(range(points_bot.shape[0] - ang_arr_channel.shape[1], points_bot.shape[0]))
    start_index = points_bot.shape[0]

    # polygon boundary appended
    polygon += list(np.flip(boundary_tc))
    polygon += boundary_cc

    # define points and BC for bottom piece of the wall above the cooling channel
    points_top = np.concatenate((np.sin(ang_arr_channel), np.cos(ang_arr_channel)), axis=2) * scaler_top
    points_top = points_top.reshape(points_top.shape[0] * points_top.shape[1], 2)
    boundary_cc += list(range(start_index, start_index + ang_arr_channel.shape[1]))
    boundary_is += list(range(start_index, start_index + points_top.shape[0], ang_arr_channel.shape[1]))
    boundary_is += list(
        range(start_index + points_top.shape[0] - ang_arr_channel.shape[1], start_index + points_top.shape[0]))

    # polygon boundary appended
    polygon += list(np.flip(list(range(start_index, start_index + ang_arr_channel.shape[1]))))
    polygon += list(
        range(start_index + points_top.shape[0] - ang_arr_channel.shape[1], start_index + points_top.shape[0]))

    # define the points and BC of the radial wall of the cooling channel
    start_index += points_top.shape[0]
    scaler_rad_cc_wall = np.expand_dims(
        np.linspace(r_chamber + t_gap_bot, r_chamber + t_gap_bot + a_channel, int(np.ceil(a_channel / spacing) + 1)),
        axis=1)
    scaler_rad_cc_wall = scaler_rad_cc_wall[1:-1, 0:1]
    points_rad_cc_wall = np.expand_dims([np.sin(ang_channel), np.cos(ang_channel)], axis=0) * scaler_rad_cc_wall
    boundary_cc += list(range(start_index, start_index + points_rad_cc_wall.shape[0]))

    # define points and BC for the part of the wall without cooling channel
    start_index += points_rad_cc_wall.shape[0]
    ang_arr_wall = ang_arr_wall[0:1, 1:, 0:1]
    points_wall = np.concatenate((np.sin(ang_arr_wall), np.cos(ang_arr_wall)), axis=2) * scalar_wall
    points_wall = points_wall.reshape(points_wall.shape[0] * points_wall.shape[1], 2)
    N_ang_wall = ang_arr_wall.shape[1]
    boundary_tc += list(range(start_index, start_index + N_ang_wall))
    boundary_is += list(range(start_index + N_ang_wall - 1, start_index + points_wall.shape[0], N_ang_wall))
    boundary_is += list(range(start_index + points_wall.shape[0] - N_ang_wall, start_index + points_wall.shape[0]))

    # delete duplicates in boundary list
    boundary_is = list(set(boundary_is))

    # polygon boundary appended
    polygon += list(range(start_index + points_wall.shape[0] - N_ang_wall, start_index + points_wall.shape[0]))
    polygon += list(np.flip(list(range(start_index, start_index + N_ang_wall))))

    # concatenate points from parts
    points = np.concatenate((points_bot, points_top, points_rad_cc_wall, points_wall), axis=0)

    # array of indexes of points that require boundary condition
    boundary = np.unique(np.array(boundary_cc + boundary_tc + boundary_is))
    # triangulation of all points
    triang = spatial.Delaunay(points)

    # transform the edge into a list of doublets that are readable by shapely and create polygon defining geometry
    p = [(x, y) for x, y in points[polygon, :]]
    pg = geometry.Polygon(p)

    # find and delete all triangles that lie beyond the polygon
    outside_triangles = []
    for i in range(triang.simplices.shape[0]):
        if not (pg.contains(geometry.Point(np.mean(points[triang.simplices[i, :], 0]),
                                           np.mean(points[triang.simplices[i, :], 1])))):
            outside_triangles.append(i)

    triangle = np.delete(triang.simplices, outside_triangles, axis=0)
    neighbors = np.delete(triang.neighbors, outside_triangles, axis=0)
    neighbors[np.isin(neighbors, outside_triangles)] = -1

    # define boundary conditions 1-convection_channel, 2-convection_trust_chamber, 3-insulation
    bc = GetBC(boundary_cc, boundary_tc, boundary_is, triangle, neighbors, boundary, h_cc, T_cc, h_tc, T_tc)

    if (showPlot):
        plt.axis('equal')
        plt.triplot(points[:, 0], points[:, 1], triangle)
        c = ["g", "b", "r", "k"]
        for i in range(bc.shape[0]):
            x = [points[bc[i, 0], 0], points[bc[i, 1], 0]]
            y = [points[bc[i, 0], 1], points[bc[i, 1], 1]]
            plt.plot(x, y, color=c[bc[i, 3]])
    plt.show()

    return points, triangle, bc, pg


def h_Re(r, mu, m_dot, Cp, k):
    d = 2 * r
    Re = 4 * m_dot / (np.pi * d * mu)
    Pr = Cp * mu / k
    f = (0.79 * np.log(Re) - 1.64) ** (-2)
    Nu = ((f / 8) * (Re - 1000) * Pr) / (1 + 12.7 * (f / 8) ** 0.5 * (Pr ** (2 / 3) - 1))
    h = Nu * k / d
    return h, Re

def h_Re2(r,A,mu,m_dot,Cp,k):
    d = 2 * r
    Re = d*m_dot/(mu*A)
    Pr = Cp * mu / k
    f = (0.79 * np.log(Re) - 1.64) ** (-2)
    Nu = ((f / 8) * (Re - 1000) * Pr) / (1 + 12.7 * (f / 8) ** 0.5 * (Pr ** (2 / 3) - 1))
    h = Nu * k / d
    return h, Re


def SolveHeatTransferCircle(r_channel, n_channels, r_chamber, m_dot, Tg, hg, spacing=0.0005, T_n2o=300, t_gap=0.001, showPlot=False):
    h_channel = r_channel + t_gap
    t_wall = 2 * h_channel

    dm_dot = m_dot / n_channels
    Cp_n2o = 2000
    k_n2o = 0.08
    mu_n2o = 1.37e-5
    # rho_n2o = 770
    k = 20
    t = 1
    h_n2o, Re = h_Re(r_channel, mu_n2o, dm_dot, Cp_n2o, k_n2o)

    points, triangle, bc, pg = GetMesh(n_channels, r_channel, h_channel, t_wall, r_chamber, spacing, h_n2o, T_n2o, hg, Tg)
    hf = HeatFlow(points, triangle, bc, k, pg)
    if (showPlot):
        hf.PlotTemp()
    return hf.MaxTemp()

def SolveHeatTransferSquare(a_channel,ratio_channel, n_channels, r_chamber, m_dot, Tg, hg, Cp_n2o, mu_n2o, k_n2o, rho_n2o, T_n2o, spacing=0.0002, t_gap_bot=0.001,t_gap_top=0.003, t_gap_side = 0.001, showPlot=False):
    ang_channel = ratio_channel * ((r_chamber + t_gap_bot) * np.pi / n_channels - t_gap_side / 2) / (
                (r_chamber + t_gap_bot) * np.pi / n_channels)
    dm_dot = m_dot / n_channels
    k = [0.0161, 5.2269]
    A = np.pi * ((r_chamber + t_gap_bot + a_channel) ** 2 - (r_chamber + t_gap_bot) ** 2) * ang_channel / n_channels
    P = 2 * a_channel + (2 * np.pi * ang_channel / n_channels) * (
                r_chamber + t_gap_bot + a_channel + r_chamber + t_gap_bot)
    r_channel = 2 * A / P
    t = 1
    h_n2o,_ = h_Re(r_channel, mu_n2o, dm_dot, Cp_n2o, k_n2o)

    points, triangle, bc, pg = GetMeshS(n_channels, a_channel, ang_channel, t_gap_bot, t_gap_top, r_chamber, spacing,
                                        h_n2o, T_n2o, hg, Tg, showPlot=False)
    hf = HeatFlow(points, triangle, bc, k, pg)
    hf.Solve_itt()
    if showPlot:
        hf.PlotTemp()

    v = m_dot/(A*rho_n2o)
    Re = rho_n2o*2*r_channel*v/mu_n2o
    dp = deltaP(2*r_channel,1000,v,Re,rho_n2o)

    return hf.MaxTemp(), dp, r_channel

def SolveHeatTransferSquare2(a_channel,ratio_channel, n_channels, r_chamber, m_dot, Tg, hg, P_n2o, T_n2o, spacing=0.0002, t_gap_bot=0.001,t_gap_top=0.003, t_gap_side = 0.001, showPlot=False):
    ang_channel = ratio_channel * ((r_chamber + t_gap_bot) * np.pi / n_channels - t_gap_side / 2) / (
                (r_chamber + t_gap_bot) * np.pi / n_channels)
    dm_dot = m_dot / n_channels
    k = [0.0161, 5.2269]
    A = np.pi * ((r_chamber + t_gap_bot + a_channel) ** 2 - (r_chamber + t_gap_bot) ** 2) * ang_channel / n_channels
    P = 2 * a_channel + (2 * np.pi * ang_channel / n_channels) * (
                r_chamber + t_gap_bot + a_channel + r_chamber + t_gap_bot)
    r_channel = 2 * A / P
    t = 1
    Cp_n2o, mu_n2o, k_n2o, rho_n2o = GetData(T_n2o, P_n2o)
    h_n2o, Re = h_Re2(r_channel, A, mu_n2o, dm_dot, Cp_n2o, k_n2o)

    points, triangle, bc, pg = GetMeshS(n_channels, a_channel, ang_channel, t_gap_bot, t_gap_top, r_chamber, spacing,
                                        h_n2o, T_n2o, hg, Tg, showPlot=False)
    hf = HeatFlow(points, triangle, bc, k, pg)
    hf.Solve_itt()
    if showPlot:
        hf.PlotTemp()

    v = m_dot/(A*rho_n2o)
    dp = deltaP(2*r_channel, 1000, v, Re, rho_n2o)

    return hf.MaxTemp(), dp

def SliceOptimization(n_channels,r_chamber,m_dot_ox,Tg,hg,P_n2o,T_n2o):
    T_wall = 1100

    ratio_channel = 1
    a_channel_min = 0.0015
    a_channel_max = 0.01
    spacing = 0.0001

    T_min, dp_min = SolveHeatTransferSquare2(a_channel_min, ratio_channel, n_channels, r_chamber, m_dot_ox, Tg, hg, P_n2o, T_n2o,
                                        spacing=spacing, t_gap_bot=0.001, t_gap_top=0.003, t_gap_side=0.001,
                                        showPlot=False)
    T_max, dp_max = SolveHeatTransferSquare2(a_channel_max, ratio_channel, n_channels, r_chamber, m_dot_ox, Tg, hg, P_n2o, T_n2o,
                                        spacing=spacing, t_gap_bot=0.001, t_gap_top=0.003, t_gap_side=0.001,
                                        showPlot=False)

    if T_min>=T_wall:
        return a_channel_min , ratio_channel, dp_min, T_min
    elif T_max<=T_wall:
        return a_channel_max , ratio_channel, dp_max, T_max

    itt_max = 10
    itt = 0
    T_err = 1
    while True:
        a_channel = a_channel_min + (T_wall - T_min) * (a_channel_max - a_channel_min) / (T_max - T_min)
        T, dp = SolveHeatTransferSquare2(a_channel, ratio_channel, n_channels, r_chamber, m_dot_ox, Tg, hg, P_n2o, T_n2o,
                                         spacing=spacing, t_gap_bot=0.001, t_gap_top=0.003, t_gap_side=0.001,
                                         showPlot=False)

        if (itt < itt_max) and (np.abs(T - T_wall) > T_err):
            if (T < T_wall):
                a_channel_min = a_channel
                T_min = T
            else:
                a_channel_max = a_channel
                T_max = T
        else:
            return a_channel, ratio_channel, dp, T

        itt += 1


