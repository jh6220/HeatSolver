import pandas as pd
import numpy as np
import scipy.interpolate as interp

class N2O:
    #Class for calculating properties of liquid n2o from temperature and pressure

    def __init__(self):
        #importing data tables of values for interpolation
        #   first column (V_1) is temperature in Kelvin
        #   second column (V_2) is pressure in kiloPascals
        #   third column (V_3) is value
        #   forth column (V_4) is uncertainty
        self.tCp = pd.read_csv("heatCapacity.csv")
        self.tmu = pd.read_csv("viscosity.csv")
        self.tk = pd.read_csv("conductivity.csv")
        self.trho = pd.read_csv("density.csv")
        self.tcompr = pd.read_csv("compressibilityL.csv")

        #creates 2d interpolating function handles for all values
        self.Cp = self.GetInterpFunc(self.tCp)
        self.mu = self.GetInterpFunc(self.tmu)
        self.k = self.GetInterpFunc(self.tk)
        self.rho = self.GetInterpFunc(self.trho)
        self.compr = self.GetInterpFunc(self.tcompr)

    def GetInterpFunc(self, t):
        #creates 2d interpolating function handle of a table
        data = t[["V_1", "V_2"]].to_numpy() #[temperature vector, pressure vector]
        Z = t["V_3"] #vector of values to be interpolated
        f = interp.LinearNDInterpolator(data, Z)
        return f

    def GetCp(self, T,P):
        return self.Cp(T,P)

    def GetMu(self, T,P):
        return self.mu(T,P)

    def GetK(self, T,P):
        return self.k(T,P)

    def GetRho(self, T,P):
        return self.rho(T,P)

    def GetData(self, T, P):
        #returns all n2o properties that are needed for convection coeffcient calculation
        return self.Cp(T,P), self.mu(T,P), self.k(T,P), self.rho(T,P)

    def GetCompr(self, T,P):
        return self.compr(T,P)


n2o = N2O()

def GetData(T,P):
    return n2o.GetData(T,P)






