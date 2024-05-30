from enum import Enum
from tqdm import tqdm
import numpy as np


__author__ = "Wei Ma"
__copyright__ = "Copyright (C) 2024 Wei Ma"
__license__ = ""
__version__ = "1.0"

'''
Author: Wei Ma
Date: 2024/05/30

Description:
This code aims to provide numerical solutions for 1D radial flow (for ideal gas) in a homogeneous and isotropic media.

These are the assumptions:
1. constant b
2. constant porosity phi
3. constant viscosity mu
4. constant Z = 1
5. constant T = Tsc
6. permeability k(r,t) = k_0 * (1 + b / P(r,t)), where P is pressure at time t and radius r

Although there are two RunMode, "variable_k" is always preferred.

IMPORTANT Note:
Since the simulation is based on explict formula, one has to be very careful selecting a proper dt and dr.
Generally, dt < Constant * (dr)^2, one may try to find a good dt for each dr selection.

In the main function, two Cases are provided to demonstrate how to use this code.
'''


class RunMode(Enum):
    CONSTANT_K = "constant_k"
    VARIABLE_K = "variable_k"


class RadialFlow1D:
    def __init__(self, config) -> None:
        self.config = config
        self.p_up = float(config['p_up'])
        self.p_down = float(config['p_down'])
        self.p_sc = float(config['p_sc'])
        self.k0 = float(config['k0'])
        self.phi = float(config['phi'])
        self.mu = float(config['mu'])
        self.b = float(config['b'])
        self.T = float(config['T'])
        self.nt = int(config['nt'])
        self.r0 = float(config['r0'])
        self.nx = int(config['nx'])
        self.NX = self.nx + 2
        self.dt = self.T / self.nt
        self.dr = self.r0 / self.NX

    
    def get_perm(self, p):
        return self.k0 * (1.0 + 1.0 * self.b / (p + 1.0e-17))
    

    def iterate_constant_k(self, p0):
        # initialization
        p1 = np.ones_like(p0)

        # no flow boundary ==> P(-1) = P(0), or P(0) = p_0 as always for the first block (r_0)
        p1[0] = self.p_up

        # for the inner cells
        dt = self.dt
        dr = self.dr
        
        nx = len(p0)

        for k in range(1, nx - 1):
            r_k = self.r0 - k * self.dr
            c = 1.0 * self.get_perm(p0[k]) * dt / (self.mu * self.phi * r_k * dr)
            part_1 = (r_k + 0.5 * self.dr) * 0.5 * (p0[k+1] + p0[k]) * (p0[k+1] - p0[k]) / dr
            part_2 = (r_k - 0.5 * self.dr) * 0.5 * (p0[k] + p0[k-1]) * (p0[k] - p0[k-1]) / dr
            p1[k] = p0[k] + c * (part_1 - part_2)
        
        # no flow boundary ==> P(n+2) = P(n+1)
        p1[nx-1] = p1[nx-2]

        return p1
        

    def iterate_variable_k(self, p0):
        # initialization
        p1 = np.ones_like(p0)

        # no flow boundary ==> P(-1) = P(0), or P(0) = p_0 as always for the first block (r_0)
        p1[0] = self.p_up

        # for the inner cells
        dt = self.dt
        dr = self.dr
        nx = self.NX
        for k in range(1, nx - 1):
            r_k = self.r0 - k * self.dr
            c = 1.0 * self.k0 * dt / (self.mu * self.phi * r_k * dr)
            part_1 = (r_k + 0.5 * self.dr) * (0.5 * (p0[k+1] + p0[k]) + self.b) * (p0[k+1] - p0[k]) / dr
            part_2 = (r_k - 0.5 * self.dr) * (0.5 * (p0[k] + p0[k-1]) + self.b) * (p0[k] - p0[k-1]) / dr
            p1[k] = p0[k] + c * (part_1 - part_2)
        
        # no flow boundary ==> P(n+2) = P(n+1)
        p1[nx-1] = p1[nx-2]

        return p1
    

    def run(self, mode=RunMode.VARIABLE_K.value):
        nt = self.nt
        nx = self.NX
        print("\n\t=== nt = {}, nx = {}".format(nt, nx))

        p = np.zeros((nt, nx)) # save the simulation solutions for all grid cells at each time step

        p0 = np.ones(nx) * self.p_down
        print("\n\t=== initial condition pressure : {}".format(p0))
        p0[0] = self.p_up
        print("\n\t=== initial condition pressure : {}".format(p0))

        p[0] = p0
        for i in tqdm(range(1, self.nt)):
            if mode == RunMode.VARIABLE_K.value:
                p1 = self.iterate_constant_k(p0)
            elif mode == RunMode.CONSTANT_K.value:
                p1 = self.iterate_variable_k(p0)
            p[i] = p1
            p0 = p1

        return p
    


if __name__ == "__main__":
    '''
    If you want to run a simulation with custom parameters
    1. edit the configuration file "radial_flow_1d_config.py" and then import it as Case 1
    2. edit the configuration directly as Case 2
    '''
    from radial_flow_1d_config import radial_flow_1d_config

    '''
    Case 1
    Replace "False" with "True" will active this case
    '''
    if False:
        simulator = RadialFlow1D(config=radial_flow_1d_config)
        p = simulator.run(mode=RunMode.VARIABLE_K.value)

    '''
    Case 2
    Replace "False" with "True" will active this case
    '''
    if True:
        config = {
            "p_up": 10,
            "p_down": 1,
            "p_sc": 1,
            "k0": 1.0e-3,
            "phi": 0.1,
            "mu": 1,
            "b": 1,
            "T": 10,
            "nt": 100000,
            "r0": 10,
            "nx": 998,
        }

        simulator = RadialFlow1D(config=config)
        p = simulator.run(mode=RunMode.VARIABLE_K.value)
        print("\n\t=== The pressue distribution at the last time step = {}".format(p[-1]))