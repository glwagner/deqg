import numpy as np
import time, logging

from mpi4py import MPI
from dedalus import public as de
from numpy import pi

from .quasigeostrophy import QGModel
from .utils import add_parameters, add_first_derivative_substitutions, bind_parameters
from .utils import bind_state_variables

logger = logging.getLogger(__name__)

class TwoLayerModel(QGModel):
    """
    A model for two-layer quasi-geostrophic flow.

    Args
    ----
        nx : (int) 
            Grid resolution in :math:`x`

        ny : (int)
            Grid resolution in :math:`y`

        Lx : (float)
            Domain extent in :math:`x`

        Ly : (float)
            Domain extent in :math:`y`

        F1 : (float)
            Pedlosky bullshit 1

        F2 : (float)
            Pedlosky bullshit 2

        ν : (float)
            Hyperviscosity

        β : (float)
            Planetary gradient

        η : (str)
            Bottom topography (somehow)

        **params : (any)
            Additional parameters to be added to the dedalus problem.
    """
    def __init__(self,
        nx = 64,
        ny = 64,
        Lx = 2.0,       
        Ly = 2.0,       
        F1 = 25.0,       
        F2 = 6.25,
        β = 0.0,
        ν = 0.0,
        U = "0",
        η = "0",
        **params         
        ):

        # Create bases and domain
        self.xbasis = xbasis = de.Fourier('x', nx, interval=(-Lx/2, Lx/2), dealias=3/2)
        self.ybasis = ybasis = de.Fourier('y', ny, interval=(-Ly/2, Ly/2), dealias=3/2)
        self.domain = domain = de.Domain([xbasis, ybasis], grid_dtype=np.float64)
        
        self.variables = variables = ['q1', 'q2', 'ψ1', 'ψ2']
        self.problem = problem = de.IVP(domain, variables=variables, time='t')
        
        add_parameters(problem, β=β, F1=F1, F2=F2, ν=ν, **params)
        bind_parameters(self, β=β, F1=F1, F2=F2, ν=ν, U=U, η=η, **params)

        problem.substitutions['η'] = η
        add_first_derivative_substitutions(problem, ['q1', 'q2', 'ψ1', 'ψ2', 'η'], ['x', 'y'])

        problem.substitutions['J(a, b)'] = "dx(a)*dy(b) - dy(a)*dx(b)"
        problem.substitutions['lap(a)'] = "dx(dx(a)) + dy(dy(a))"
        problem.substitutions['U'] = U
        problem.substitutions['Uyy'] = "dy(dy(U))"

        # Equations
        problem.add_equation("dt(q1) + β*ψ1x + ν*lap(lap(q1)) = - J(ψ1, q1) + (Uyy - F1*U)*ψ1x - U*q1x")
        problem.add_equation("dt(q2) + β*ψ2x + ν*lap(lap(q2)) = - J(ψ1, q1+η)      + F2*U*ψ2x")

        problem.add_equation("q1 - lap(ψ1) - F1*(ψ2-ψ1) = 0", condition="(nx != 0) or (ny != 0)")
        problem.add_equation("q2 - lap(ψ2) - F2*(ψ1-ψ2) = 0", condition="(nx != 0) or (ny != 0)")
        problem.add_equation("ψ1 = 0", condition="(nx == 0) and (ny == 0)")
        problem.add_equation("ψ2 = 0", condition="(nx == 0) and (ny == 0)")

        self.x = domain.grid(0)
        self.y = domain.grid(1)
        
    def build_solver(self, timestepper='RK443'):
        """Build a dedalus solver for the model with `timestepper`.

        Args
        ----
            timestepper : The name of the timestepper to be used by the solver. 
        """
        QGModel.build_solver(self, timestepper=timestepper)
        bind_state_variables(self, self.variables)
