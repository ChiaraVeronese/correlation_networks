import numpy as np
import numpy.random as rm
from scipy.integrate import odeint

class Kuramoto:

    def __init__(self, coupling=1, dt=0.01, T=10, n_nodes=None, natfreqs=None, init_angles=None):
        '''
        coupling: float
            Coupling strength. Default = 1. Typical values range between 0.4-2
        dt: float
            Delta t for integration of equations.
        T: float
            Total time of simulated activity.
            From that the number of integration steps is T/dt.
        n_nodes: int, optional
            Number of oscillators.
            If None, it is inferred from len of natfreqs.
            Must be specified if natfreqs is not given.
        natfreqs: 1D ndarray, optional
            Natural oscillation frequencies.
            If None, then new random values will be generated and kept fixed
            for the object instance.
            Must be specified if n_nodes is not given.
            If given, it overrides the n_nodes argument.
        init_angles: 1D ndarray, optional
            Random initial random angles (position, "theta").
            states vector of nodes representing the position in radians.
            If not specified, random initialization [0, 2pi].
        '''
        self.dt = dt
        self.T = T
        self.coupling = coupling

        if natfreqs is not None:
            self.natfreqs = natfreqs
            self.n_nodes = len(natfreqs)
        else:
            self.n_nodes = n_nodes
            self.natfreqs = np.random.normal(size=self.n_nodes)#np.random.uniform(-20*0.27/12,20*0.27/12,size=self.n_nodes)

        if init_angles is None:
            self.init_angles=2 * np.pi * np.random.random(size=self.n_nodes)
   

    def drift(self, angles_vec, t, adj_mat, coupling):
        '''
        Compute derivative of all nodes for current state, defined as
        dx_i    natfreq_i + k  sum_j ( Aij* sin (x_j - x_i) )
        ---- =             ---
         dt                M_i
        t: for compatibility with scipy.odeint
        '''
        angles_i, angles_j = np.meshgrid(angles_vec, angles_vec)
        interactions = adj_mat * np.sin(angles_j - angles_i)  # Aij * sin(j-i)

        dxdt = self.natfreqs + coupling * interactions.sum(axis=0)  # sum over incoming interactions
        return dxdt
    
    

    def euler_maruyama(self, drift, init_angles, t, eps, adj_mat, coupling):
        '''
        x_i: oscillator phase at node i
        Compute derivative of all nodes for current state, defined as
        dx_i    drift_i(x,Aij) + D dW_i
        ---- =                     ----
         dt                         dt
        '''
        N = len(t)
        dt = t[1]-t[0]
        x = np.zeros((init_angles.shape[0],N))
        x[:,0] = init_angles
        
        for i in range(N-1):
            dWt = rm.normal(0, np.sqrt(dt), init_angles.shape)
            x[:,i+1] = x[:,i] + drift(x[:,i],t,adj_mat,coupling) * dt + np.sqrt(2*eps) * dWt

        return x


    def integrate(self, adj_mat, eps):
        '''Updates all states by integrating state of all nodes'''
        # Coupling term (k / Mj) is constant in the integrated time window.
        # Compute it only once here and pass it to the derivative function
        n_interactions = (adj_mat != 0).sum(axis=0)  # number of incoming interactions #np.ones(len(adj_mat))#
        coupling = self.coupling/ n_interactions  # normalize coupling by number of interactions
       
        
        t = np.linspace(0, self.T, int(self.T/self.dt))
        if eps>0:
            timeseries = self.euler_maruyama(self.drift, self.init_angles, t, eps, adj_mat, coupling)
            return timeseries
        else:
            timeseries = odeint(self.drift, self.init_angles, t, args=(adj_mat, coupling))
            return timeseries.T  # transpose for consistency (act_mat:node vs time)

    def run(self, adj_mat=None, eps=0):
        '''
        adj_mat: 2D nd array
            Adjacency matrix representing connectivity.
        eps: float
            noise intensity. When eps=0 the integration is done by odeint, 
            whereas in case of SDE Euler-Maruyama integration
        Returns
        -------
        act_mat: 2D ndarray
            Activity matrix: node vs time matrix with the time series of all
            the nodes.
        '''
        assert (adj_mat == adj_mat.T).all(), 'adj_mat must be symmetric'
        
        return self.integrate(adj_mat, eps)

