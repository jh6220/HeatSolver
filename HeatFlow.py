import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy import interpolate
import scipy as sp
from shapely import geometry

class HeatFlow:
    t = 1 # thickness of the 2d geometry (does not matter right now)
    rho = 8192 # kg/m^3 for Inconel
    Cp = 500  #J/(kg*K) for Inconel

    def __init__(self, points, triangles, bc, k, boundary=None):
        self.points = points # [x values vector, y values vector]
        self.triangles = triangles # [N_triangles x 3] matrix - each row is are indexes of three points that make triangle together
        self.bc = bc # [N_points x 6] matrix - each row defines boundary condition between two points on the surface
        self.k = k # list of two items... thermal conductivity = k[1]*T + k[2]
        self.boundary = boundary
        self.area = self.CalcArea()
        self.T = np.ones([self.points.shape[0], 1]) * 700
        self.T_grid = np.zeros([self.points.shape[0],1])
        self.anim_time = []

        # self.Solve_itt()

        # self.T = np.ones([self.points.shape[0], 1]) * 700
        # self.Solve()

    def PlotMesh(self):
        plt.axis('equal')
        plt.triplot(self.points[:, 0], self.points[:, 1], self.triangles)
        c = ["g", "b", "r", "k"]
        for i in range(self.bc.shape[0]):
            x = [self.points[self.bc[i, 0], 0], self.points[self.bc[i, 1], 0]]
            y = [self.points[self.bc[i, 0], 1], self.points[self.bc[i, 1], 1]]
            plt.plot(x, y, color=c[self.bc[i, 3]])
        plt.show()

    # def CalcArea(self):
    #     area = np.zeros(len(self.triangles))
    #     for i in range(len(self.triangles)):
    #         x = self.points[self.triangles[i, :], 0]
    #         y = self.points[self.triangles[i, :], 1]
    #         area[i] = np.abs(x[0] * (y[1] - y[2]) + x[1] * (y[2] - y[0]) + x[2] * (y[0] - y[1])) / 2
    #     return area

    def CalcArea(self):
        # Calculates the area of each triangle
        area = np.zeros(len(self.triangles))
        i = 0
        while i<self.triangles.shape[0]:
            x = self.points[self.triangles[i, :], 0]
            y = self.points[self.triangles[i, :], 1]
            area[i] = np.abs(x[0] * (y[1] - y[2]) + x[1] * (y[2] - y[0]) + x[2] * (y[0] - y[1])) / 2
            # sometime triangles can be defined outside the geometry, they have very small area and are deleted
            if area[i] < 10**(-15):
                self.triangles = np.delete(self.triangles,i,axis=0)
                area = np.delete(area,i)
            else:
                i = i+1
        return area

    def GetK(self):
        self.K = np.zeros([self.points.shape[0], self.points.shape[0]])
        for index in range(len(self.triangles)):
            x = self.points[self.triangles[index, :], 0]
            y = self.points[self.triangles[index, :], 1]
            for i in range(3):
                for j in range(i, 3):
                    k_elem = self.k[0] * np.mean(self.T[self.triangles[index, :]]) + self.k[1]
                    if (i != j):
                        bi = y[(i + 1) % 3] - y[(i + 2) % 3]
                        bj = y[(j + 1) % 3] - y[(j + 2) % 3]
                        ci = - x[(i + 1) % 3] + x[(i + 2) % 3]
                        cj = - x[(j + 1) % 3] + x[(j + 2) % 3]
                        self.K[self.triangles[index, i], self.triangles[index, j]] += \
                            (bi * bj + ci * cj) * k_elem / self.area[index]
                        self.K[self.triangles[index, j], self.triangles[index, i]] += \
                            (bi * bj + ci * cj) * k_elem / self.area[index]
                    else:
                        bi = y[(i + 1) % 3] - y[(i + 2) % 3]
                        ci = - x[(i + 1) % 3] + x[(i + 2) % 3]
                        self.K[self.triangles[index, i], self.triangles[index, i]] += \
                            (bi * bi + ci * ci) * k_elem / self.area[index]
        self.K *= self.t / 4

        for index in range(self.bc.shape[0]):
            if ((self.bc[index, 3] == 1) or (self.bc[index, 3] == 2)):
                length = np.linalg.norm(self.points[self.bc[index, 0], :] - self.points[self.bc[index, 1], :])
                self.K[np.ix_(self.bc[index, [0, 1]], self.bc[index, [0, 1]])] += \
                    (self.bc[index, 4] * self.t * length / 6) * np.array([[2, 1], [1, 2]])

        return self.K

    def Getf(self):
        self.f = np.zeros((self.points.shape[0], 1))
        for index in range(self.bc.shape[0]):
            if ((self.bc[index, 3] == 1) or (self.bc[index, 3] == 2)):
                length = np.linalg.norm(self.points[self.bc[index, 0], :] - self.points[self.bc[index, 1], :])
                self.f[self.bc[index, [0, 1]]] += self.bc[index, 4] * self.bc[index, 5] * self.t * length / 2

        return self.f

    def GetC(self):
        self.C = np.zeros([self.points.shape[0], self.points.shape[0]])
        temp_mat = np.array([[1/6, 1/12, 1/12],
                             [1/12, 1/6, 1/12],
                             [1/12, 1/12, 1/6]]) #derived in matlab (refer to git repository Propulsion>Regenerative Cooling>SymbolicCalculationsFE.m
        for index in range(len(self.triangles)):
            self.C[np.ix_(self.triangles[index, :], self.triangles[index, :])] += \
                self.rho*self.Cp*self.t*self.area[index]*temp_mat

        return self.C

    def Solve(self):
        self.GetK()
        self.Getf()
        self.T = np.linalg.solve(self.K, self.f)

    def Solve_itt(self):
        # self.T = np.ones([self.points.shape[0], 1])*700
        for i in range(3):
            self.Solve()

    def SolveTransient(self,t_end,dt,T0 = 300, plotTmax = False):
        self.dt = dt
        self.time = np.arange(0,t_end,dt)
        self.T_transient = np.zeros([self.points.shape[0], self.time.shape[0]])
        self.T = 700*np.ones([self.points.shape[0],1])
        self.GetK()
        self.Getf()
        self.GetC()
        self.T_transient[:, [0]] = T0*np.ones([self.points.shape[0],1])
        T_max_arr = np.zeros([self.time.shape[0],1])
        T_max_arr[0] = T0
        lhs = (self.C + 0.5 * dt * self.K)
        inv_lhs = np.linalg.inv(lhs)
        for i in range(1,self.time.shape[0]):
            print("\r{:.2f}%".format(100*self.time[i-1]/t_end),end='')

            # self.GetK()
            # lhs = (self.C + 0.5 * dt * self.K)

            rhs = np.dot(self.C - 0.5 * dt * self.K, self.T_transient[:, [i - 1]]) + dt * self.f

            self.T_transient[:,[i]] = np.dot(inv_lhs,rhs)

            # self.T = np.linalg.solve(lhs, rhs)
            # self.T_transient[:, [i]] = self.T

            T_max_arr[i] = np.max(self.T_transient[:,i])

        print('\r100.00%')

        if plotTmax:
            plt.plot(self.time,T_max_arr)
            plt.show()

    def Animate(self,showAnim = True, saveFile = ""):
        fig, self.ax = plt.subplots()
        X,Y,_,_ = self.GetColourMapData()
        self.cax = self.ax.pcolormesh(X,Y,self.T_grid[:,:,-1],cmap="plasma")
        plt.colorbar(self.cax)
        anim = animation.FuncAnimation(fig,self.animate_func,interval=100,frames=len(self.anim_time))
        if showAnim:
            plt.show()

        if not (saveFile == ""):
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=20, metadata=dict(artist='Jakub Horsky'), bitrate=1800)
            anim.save(saveFile, writer=writer)

    def animate_func(self,i):
        self.cax.set_array(self.T_grid[:,:,i])
        self.ax.set_title("{:.2f} seconds".format(self.anim_time[i]))

    def GetColourMapData(self,n_skip=10):
        x = np.linspace(np.min(self.points[:, 0]), np.max(self.points[:, 0]), 200)
        y = np.linspace(np.min(self.points[:, 1]), np.max(self.points[:, 1]), 200)
        X_grid, Y_grid = np.meshgrid(x, y)
        X = (X_grid[:-1, :-1] + X_grid[1:, 1:]) / 2
        Y = (Y_grid[:-1, :-1] + Y_grid[1:, 1:]) / 2

        i_nan = []
        j_nan = []
        if (self.boundary != None):
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    if not (self.boundary.contains(geometry.Point(X[i, j], Y[i, j]))):
                        i_nan.append(i)
                        j_nan.append(j)

        self.T_grid = np.zeros([X.shape[0],X.shape[1],len(range(0,self.T_transient.shape[1],n_skip))])
        self.anim_time = []
        index = 0
        for i in range(0,self.T_transient.shape[1],n_skip):
            self.anim_time.append(self.time[i])
            print("\r{:.2f}%".format(100 * i / self.T_transient.shape[1]), end='')
            f = sp.interpolate.LinearNDInterpolator(self.points, self.T_transient[:, i])
            self.T_grid[:,:,index] = f(X, Y)
            index += 1

        print('\r100.00%')

        for i in range(len(i_nan)):
            self.T_grid[i_nan[i],j_nan[i],:] = np.nan

        return X_grid,Y_grid,self.T_grid,self.anim_time

    def PlotTemp(self):
        f = sp.interpolate.LinearNDInterpolator(self.points, self.T)
        x = np.linspace(np.min(self.points[:, 0]), np.max(self.points[:, 0]), 200)
        y = np.linspace(np.min(self.points[:, 1]), np.max(self.points[:, 1]), 200)
        X_grid, Y_grid = np.meshgrid(x, y)
        X = (X_grid[:-1, :-1] + X_grid[1:, 1:]) / 2
        Y = (Y_grid[:-1, :-1] + Y_grid[1:, 1:]) / 2
        TempGrid = f(X, Y)
        TempGrid = np.squeeze(TempGrid)
        if (self.boundary != None):
            for i in range(TempGrid.shape[0]):
                for j in range(TempGrid.shape[1]):
                    if not (TempGrid[i, j] == np.nan) and not (self.boundary.contains(geometry.Point(X[i, j], Y[i, j]))):
                        TempGrid[i, j] = np.nan
        plt.axis('equal')
        plt.pcolormesh(X, Y, TempGrid, cmap="plasma")  # cmap = "plasma";"bwr"
        plt.colorbar()
        plt.show()

    def SamplePoints(self, x, y):
        f = sp.interpolate.LinearNDInterpolator(self.points, self.T)
        TempSampled = f(x, y)
        return TempSampled

    def MaxTemp(self):
        return np.max(self.T)