import numpy as np
from scipy import signal


class BackwardEulerDiff:
    def __init__(self, dt):
        self.x_prev = None
        self.xd = None
        self.dt = dt

        self._init = False

    def __call__(self, x):
        assert isinstance(x, np.ndarray), "Input must be an instance of np.ndarray"

        if not self._init:
            self.x_prev = np.empty_like(x)
            self.xd = np.empty_like(x)

            self.x_prev[:] = x[:]
            
            self._init = True

        np.subtract(x, self.x_prev, out=self.xd)
        self.xd /= self.dt

        self.x_prev[:] = x[:]

        return self.xd

class LowPassFilter:
    """
    Low pass filter implementation based on Euler discretisation
    method.
    """
    def __init__(self, f_cutoff, dt):
        self.dt = dt
        self.f_cutoff = f_cutoff
        
        self.tau = 1.0 / self.f_cutoff

        self.alpha = self.dt / (self.tau + self.dt)
        self.alpha_inv = 1.0 - self.alpha

        self.y = None
        self.temp_1 = None
        self.temp_2 = None

    def __call__(self, x):
        assert isinstance(x, (np.ndarray, int, float)), "Input must be an instance of np.ndarray, int or float"

        if isinstance(x, np.ndarray):
            if self.y is None:
                self.y = x.copy()

                self.temp_1 = np.empty_like(x)
                self.temp_2 = np.empty_like(x)
                return self.y
            
            np.multiply(self.alpha_inv, self.y, out=self.temp_1)
            np.multiply(self.alpha, x, out=self.temp_2)

            np.add(self.temp_1, self.temp_2, out=self.y)

        elif isinstance(x, int) or isinstance(x, float):
            if self.y is None:
                self.y = x
                return self.y

            self.y = (1.0 - self.alpha) * self.y + self.alpha * x

        return self.y