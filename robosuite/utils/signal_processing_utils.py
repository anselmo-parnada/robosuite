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
    Low pass filter implementation based on Tustin's bilinear discretisation
    method.
    """
    def __init__(self, f_cutoff, dt):
        self.dt = dt
        self.f_cutoff = f_cutoff
        omega_cutoff = 2.0 * np.pi * self.f_cutoff
        
        # Compute transfer function
        num = omega_cutoff
        den = [1,omega_cutoff] 
        lowPass = signal.TransferFunction(num,den)

        # Discretise and obtain coefficients
        discreteLowPass = lowPass.to_discrete(dt,method='gbt',alpha=0.5)
        self.a1 = discreteLowPass.den[1]
        self.b0 = discreteLowPass.num[0]
        self.b1 = discreteLowPass.num[1]

        self.y = None
        self.y_prev = None
        self.x_prev = None
        self.temp_1 = None
        self.temp_2 = None
        self.temp_3 = None

    def __call__(self, x):
        assert isinstance(x, np.ndarray), "Input must be an instance of np.ndarray"

        if self.y is None:
            self.y = x.copy()

            self.y_prev = x.copy()
            self.x_prev = x.copy()

            self.temp_1 = np.empty_like(x)
            self.temp_2 = np.empty_like(x)
            self.temp_3 = np.empty_like(x)
            return self.y
        
        np.multiply(self.a1, self.y_prev, out=self.temp_1)
        np.multiply(self.b0, x, out=self.temp_2)
        np.multiply(self.b1, self.x_prev, out=self.temp_3)

        np.add(self.temp_1, self.temp_2, out=self.y)
        np.add(self.y, self.temp_3, out=self.y)

        self.y_prev[:] = self.y[:]
        self.x_prev = x[:]

        return self.y