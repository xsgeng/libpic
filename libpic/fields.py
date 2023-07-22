import numpy as np
from .maxwell_2d import update_bfield_2d, update_efield_2d

class Fields2D:
    @classmethod
    def attrs(self):
        return ["ex", "ey", "ez", "bx", "by", "bz", "jx", "jy", "jz", "rho"]

    def __init__(self, nx, ny, dx, dy, x0, y0, n_guard) -> None:
        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.dy = dy
        self.n_guard = n_guard

        shape = (nx+2*n_guard, ny+2*n_guard)
        self.ex = np.zeros(shape)
        self.ey = np.zeros(shape)
        self.ez = np.zeros(shape)
        self.bx = np.zeros(shape)
        self.by = np.zeros(shape)
        self.bz = np.zeros(shape)
        self.jx = np.zeros(shape)
        self.jy = np.zeros(shape)
        self.jz = np.zeros(shape)
        self.rho = np.zeros(shape)


        xaxis = np.arange(nx+n_guard*2, dtype=float)
        xaxis[-n_guard:] = np.arange(-n_guard, 0)
        xaxis *= dx
        self.xaxis = xaxis[:, None] + x0

        yaxis = np.arange(ny+2*n_guard, dtype=float)
        yaxis[-n_guard:] = np.arange(-n_guard, 0)
        yaxis *= dy
        self.yaxis = yaxis[None, :] + y0

    def __getitem__(self, key):
        return self.ex[key], self.ey[key], self.ez[key], self.bx[key], self.by[key], self.bz[key], self.jx[key], self.jy[key], self.jz[key]

    def __setitem__(self, key, value):
        self.ex[key], self.ey[key], self.ez[key], self.bx[key], self.by[key], self.bz[key], self.jx[key], self.jy[key], self.jz[key] = value