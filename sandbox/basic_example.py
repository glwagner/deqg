import sys; sys.path.append("..")

import matplotlib.pyplot as plt
import numpy as np

from numpy import pi
from deqg import TwoLayerModel, mpiprint, random_noise

mpiprint("executable: {}".format(sys.executable))

# Calculation of ν (4th order hyperviscosity):
#
# >> ν = C * qmax / k^4, k=(dissipation scale).
#
# with C=10, k=(nx/2)/Lx=16, qmax=1e-4,
# => ν = 1.5e-8.

model = TwoLayerModel(
    nx = 64,
    ny = 64,
    Lx = 2.0,
    Ly = 2.0,
    F1 = 25.0,
    F2 = 6.25,
    β = 0.0,
    ν = 1.5e-8,
    U = "1/cosh(5*y)**2",
    η = "10*cos(10*pi*x)*cos(10*pi*y)",
    pi = pi,
)

model.build_solver()

# initial condition...
a = 1e-3
model.set_fields(
    q1 = a*random_noise(model.domain),
    q2 = a*random_noise(model.domain)
)

# T = q^{-1}. Or growth rate ... ?
τ = 1/a
dt = 1e-5*τ

fig, ax = plt.subplots()
for i in range(100):
    model.stop_at(iteration=200)
    model.run(dt=dt, log_cadence=100)

    plt.imshow(model.q1['g']) 
    plt.pause(0.1)

    mean_q = np.mean(model.q1['g'])
    mpiprint("mean pv: {}".format(mean_q))
