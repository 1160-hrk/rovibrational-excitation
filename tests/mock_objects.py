import numpy as np

class MockDipole:
    def __init__(self, basis):
        self.basis = basis
    
    def get_mu_in_units(self, axis, units):
        dim = self.basis.size()
        if axis == 'x':
            return np.ones((dim, dim))
        else:
            return np.zeros((dim, dim))
            
    def get_mu_x_SI(self):
        return self.get_mu_in_units("x", "C*m")

    def get_mu_y_SI(self):
        return self.get_mu_in_units("y", "C*m")
        
    @property
    def mu_x(self):
        return self.get_mu_in_units("x", "C*m")
        
    @property
    def mu_y(self):
        return self.get_mu_in_units("y", "C*m")

class MockEfield:
    def __init__(self):
        self.dt = 0.1

    def get_E_components(self, axes, out_units, out_time_units):
        return np.ones(10), np.zeros(10)
    
    @property
    def tlist_s(self):
        return np.linspace(0, 1, 21) # 10ステップになるように調整

    def get_pol(self):
        return np.array([1.0, 0.0])

    def get_Efield(self):
        Ex, Ey = self.get_E_components(None, None, None)
        return np.array([Ex, Ey])

    def get_scalar_and_pol(self):
        return np.ones(10), self.get_pol()

class DummyDipole:
    def get_mu_in_units(self, axis, units):
        return np.zeros((2, 2)) 