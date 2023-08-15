import numpy as np


class Fields:
    nx: int
    ny: int
    nz: int
    n_guard: int
    dx: float
    dy: float
    dz: float
    shape: tuple

    attrs = ["ex", "ey", "ez", "bx", "by", "bz", "jx", "jy", "jz", "rho"]

    def _init_fields(self):
        self.ex = np.zeros(self.shape)
        self.ey = np.zeros(self.shape)
        self.ez = np.zeros(self.shape)
        self.bx = np.zeros(self.shape)
        self.by = np.zeros(self.shape)
        self.bz = np.zeros(self.shape)
        self.jx = np.zeros(self.shape)
        self.jy = np.zeros(self.shape)
        self.jz = np.zeros(self.shape)
        self.rho = np.zeros(self.shape)

    def __getitem__(self, key):
        return self.ex[key], self.ey[key], self.ez[key], self.bx[key], self.by[key], self.bz[key], self.jx[key], self.jy[key], self.jz[key]

    def __setitem__(self, key, value):
        self.ex[key], self.ey[key], self.ez[key], self.bx[key], self.by[key], self.bz[key], self.jx[key], self.jy[key], self.jz[key] = value


class Fields2D(Fields):

    def __init__(self, nx, ny, dx, dy, x0, y0, n_guard) -> None:
        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.dy = dy
        self.n_guard = n_guard

        self.shape = (nx+2*n_guard, ny+2*n_guard)
        self._init_fields()

        xaxis = np.arange(nx+n_guard*2, dtype=float)
        xaxis[-n_guard:] = np.arange(-n_guard, 0)
        xaxis *= dx
        self.x0 = x0
        self.xaxis = xaxis[:, None] + x0

        yaxis = np.arange(ny+2*n_guard, dtype=float)
        yaxis[-n_guard:] = np.arange(-n_guard, 0)
        yaxis *= dy
        self.y0 = y0
        self.yaxis = yaxis[None, :] + y0


class Fields3D(Fields):

    def __init__(self, nx, ny, nz, dx, dy, dz, x0, y0, z0, n_guard) -> None:
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.n_guard = n_guard

        self.shape = (nx+2*n_guard, ny+2*n_guard, nz+2*n_guard)
        self._init_fields()

        # x axis
        xaxis = np.arange(nx+n_guard*2, dtype=float)
        xaxis[-n_guard:] = np.arange(-n_guard, 0)
        xaxis *= dx
        self.x0 = x0
        self.xaxis = xaxis[:, None, None] + x0

        # y axis
        yaxis = np.arange(ny+2*n_guard, dtype=float)
        yaxis[-n_guard:] = np.arange(-n_guard, 0)
        yaxis *= dy
        self.y0 = y0
        self.yaxis = yaxis[None, :, None] + y0

        # z axis
        zaxis = np.arange(nz+2*n_guard, dtype=float)
        zaxis[-n_guard:] = np.arange(-n_guard, 0)
        zaxis *= dz
        self.z0 = z0
        self.zaxis = zaxis[None, None, :] + z0
