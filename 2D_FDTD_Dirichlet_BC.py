import numpy as np
import scipy as sp
from scipy import constants
import matplotlib.pyplot as plt
import matplotlib.animation as animation


plt.rcParams['text.usetex'] = True


# gaussian source - this is to be sampled in time for all the time steps
# in 2D case it's the amplitude of the Ez out of the xy plane
def g(t, t0, tau):
    return np.exp(-(t - t0)**2 / tau**2)

def src_no_dc(t, t0, tau):
    return ((t) / tau)**3 * np.exp(-(t) / tau) * (4 - (t)/tau)


### this code is for Ez mode -- we have magnetic field components in xy-plane: Hx and Hy
# so, we need 3 2-dimensional arrays

if __name__=="__main__":

    Nx = 200  # total number of spatial cells in x direction
    Ny = 200
    Nt = 300  # total num of time steps
    f = 3e+8  # target frequency we are interested in having in our model, [Hz]
    num_dz_per_wavelen = 10  # rule of thumb []
    tau = 1 / 2 / f  # this is the width of the Gaussian pulse that we calculate according to the maximum frequency we want to account for
    t0 = 6 * tau  # "ease" the source into the computational domain - start at the point

    lam = sp.constants.c / f  # wavelength [m]
    dx = lam / num_dz_per_wavelen
    dy = dx
    dt = dx / np.sqrt(2)/ sp.constants.c  # CFL stability condition in 2D [s]
    omega = 2 * np.pi * f  # angular frequency [Hz]
    k = omega / sp.constants.c  # wave number [1/m]



    Ez = np.zeros((Nx, Ny)) # E(0) .... E(Nz) # for PEC BC
    Dz = np.zeros((Nx, Ny))
    Hx = np.zeros((Nx, Ny)) # H(0) .... H(Nz) # for PMC BC
    Hy = np.zeros((Nx, Ny))
    Jz = np.zeros(Nt)  # electric current density -- variation in time at the source point


    T = np.arange(0, Nt*dt, dt)
    for n in range(Nt):
        Jz[n] = src_no_dc(T[n], t0, tau)

    X = np.arange(0, (Nx)*dx, dx)
    Y = np.arange(0, (Ny) * dy, dy)


    mu_xx = np.ones((Nx, Nx))
    mu_yy = np.ones((Ny, Ny))
    eps_zz = np.ones((Nx, Nx)) # Nz == Nx
    eta_0 = np.sqrt(sp.constants.mu_0 / sp.constants.epsilon_0)

    mHx = sp.constants.c * dt / mu_xx
    mHy = sp.constants.c * dt / mu_yy
    # mHz = sp.constants.c * dt / mu_zz

    CEx = np.zeros((Nx, Ny))
    CEy = np.zeros((Nx, Ny))
    CHz = np.zeros((Nx, Ny))

    x_src = int(Nx / 3)
    y_src = int(Ny / 4)




    def time_step_2D_Dirichlet(n):

        # curl terms are CEx, CEy, CHz
        # we update H and then D and then find E from D

        global Ez
        # x component of the curl of the electric field
        CEx[:, :Ny-1] = (Ez[:, 1:Ny] - Ez[:, :Ny-1]) / dy
        CEx[:, Ny-1] = (0 - Ez[:, Ny-1]) / dy

        # y-component of the curl of the electric field
        CEy[:Nx-1, :] = -(Ez[1:Nx, :] - Ez[:Nx-1, :]) / dx
        CEy[Nx-1, :] = -(0 - Ez[Nx-1, :]) / dx

        # update H field
        global Hx
        Hx += np.multiply(-mHx, CEx)
        global Hy
        Hy += np.multiply(-mHy, CEy)

        # calculate curls of magnetic field
        # it's PEC only now
        for nx in range(1, Nx-1):
            for ny in range(1, Ny-1):
                CHz[nx, ny] = (Hy[nx, ny] - Hy[nx-1, ny]) / dx - (Hx[nx, ny] - Hx[nx, ny-1]) / dy
        # points at the lower boundary
        # for ny in range(1, Ny):
        #     CHz[0, ny] = (Hy[0, ny] - 0) / dx - (Hx[0, ny] - Hx[0, ny-1]) / dy
        # # points at the left boundary
        # for nx in range(1, Nx):
        #     CHz[nx, 0] = (Hy[nx, 0] - Hy[nx-1, 0]) / dx - (Hx[nx, 0] - 0) / dy
        # # point at the left bottom corner
        # CHz[0, 0] = (Hy[0, 0] - 0) / dx - (Hx[0, 0] - 0) / dy

        # update D field
        global Dz
        Dz += (sp.constants.c*dt)*CHz

        # add source
        Dz[x_src, y_src] += Jz[n]; # source has to have same dimensionality

        # obtaining E field (normalized)
        Ez = np.divide(Dz, eps_zz) # supposed to be element-wise


        return Hx, Hy, Ez



    # wrap up into a function for animation

    fig = plt.figure(figsize=(8, 8))
    imag = plt.imshow(Ez, vmin=-0.03, vmax=0.03)
    plt.show()


    def anim_init():
        imag.set_data(Ez)
        # plt.set_title("time={} sec".format(0))
        return imag


    def animate(n):
        Hx, Hy, Ez = time_step_2D_Dirichlet(n)
        imag.set_data(Ez)  # update the data
        # plt.set_title("time={:.2f} nsec".format(n*dt*1e+9))
        return imag


    ani = animation.FuncAnimation(fig=fig, func=animate, frames=Nt, fargs=(), init_func=anim_init)


    f = r"./2D_test_Dirichlet_BC.mp4"
    writervideo = animation.FFMpegWriter(fps=20)
    ani.save(f, writer=writervideo)






"""

time step for 3D code; will leave it here to unload 3D code



    def time_step(n):

        # inside
        # for nx in range(Nx-1):
        #     for ny in range(Ny-1):
        #         for nz in range(Nz-1):
        #             CEx[nx, ny, nz] = (Ez[nx, ny+1, nz] - Ez[nx, ny, nz]) / dy - (Ey[nx, ny, nz+1] - Ey[nx, ny, nz]) / dz
        #             CEy[nx, ny, nz] = (Ex[nx, ny, nz+1] - Ex[nx, ny, nz]) / dz - (Ez[nx+1, ny, nz] - Ez[nx, ny, nz]) / dx
        #             CEz[nx, ny, nz] = (Ey[nx+1, ny, nz] - Ey[nx, ny, nz]) / dx - (Ex[nx, ny+1, nz] - Ex[nx, ny, nz]) / dy
        # nx == Nx
        # for ny in range(Ny-1):
        #     for nz in range(Nz-1):
        #         CEx[Nx, ny, nz] = (Ez[Nx, ny+1, nz] - Ez[Nx, ny, nz]) / dy - (Ey[Nx, ny, nz+1] - Ey[Nx, ny, nz]) / dz
        #         CEy[Nx, ny, nz] = (Ex[Nx, ny, nz+1] - Ex[Nx, ny, nz]) / dz - (0 - Ez[Nx, ny, nz]) / dx
        #         CEz[Nx, ny, nz] = (0 - Ey[Nx, ny, nz]) / dx - (Ex[Nx, ny+1, nz] - Ex[Nx, ny, nz]) / dy
        # ny == Ny
        # for nx in range(Nx-1):
        #     for nz in range(Nz-1):
        #         CEx[nx, Ny, nz] = (0 - Ez[nx, Ny, nz]) / dy - (Ey[nx, Ny, nz+1] - Ey[nx, Ny, nz]) / dz
        #         CEy[nx, Ny, nz] = (Ex[nx, Ny, nz+1] - Ex[nx, Ny, nz]) / dz - (Ez[nx+1, Ny, nz] - Ez[nx, Ny, nz]) / dx
        #         CEz[nx, Ny, nz] = (Ey[nx+1, Ny, nz] - Ey[nx, Ny, nz]) / dx - (0 - Ex[nx, Ny, nz]) / dy
        # nz == Nz
        # for nx in range(Nx-1):
        #     for ny in range(Ny-1):
        #         CEx[nx, ny, Nz] = (Ez[nx, ny+1, Nz] - Ez[nx, ny, Nz]) / dy - (0 - Ey[nx, ny, Nz]) / dz
        #         CEy[nx, ny, Nz] = (0 - Ex[nx, ny, Nz]) / dz - (Ez[nx+1, ny, Nz] - Ez[nx, ny, Nz]) / dx
        #         CEz[nx, ny, Nz] = (Ey[nx+1, ny, Nz] - Ey[nx, ny, Nz]) / dx - (Ex[nx, ny+1, Nz] - Ex[nx, ny, Nz]) / dy
        # nx == Nx, Ny == ny, nz = 0 ... Nz-1
        # for nz in range(Nz - 1):
        #     CEx[Nx, Ny, nz] = (0 - Ez[Nx, Ny, nz]) / dy - (Ey[Nx, Ny, nz + 1] - Ey[Nx, Ny, nz]) / dz
        #     CEy[Nx, Ny, nz] = (Ex[Nx, Ny, nz + 1] - Ex[Nx, Ny, nz]) / dz - (0 - Ez[Nx, Ny, nz]) / dx
        #     CEz[Nx, Ny, nz] = (0 - Ey[Nx, Ny, nz]) / dx - (0 - Ex[Nx, Ny, nz]) / dy
        # nx == Nx, ny = 0 ... Ny-1, nz == Nz
        # for ny in range(Ny-1):
        #     CEx[Nx, ny, Nz] = (Ez[Nx, ny + 1, Nz] - Ez[Nx, ny, Nz]) / dy - (0 - Ey[Nx, ny, Nz]) / dz
        #     CEy[Nx, ny, Nz] = (0 - Ex[Nx, ny, Nz]) / dz - (0 - Ez[Nx, ny, Nz]) / dx
        #     CEz[Nx, ny, Nz] = (0 - Ey[Nx, ny, Nz]) / dx - (Ex[Nx, ny + 1, Nz] - Ex[Nx, ny, Nz]) / dy
        # nx = 0 ... Nx-1, ny == Ny, nz == Nz
        # for nx in range(Nx-1):
        #     CEx[nx, Ny, Nz] = (0 - Ez[nx, Ny, Nz]) / dy - ( - Ey[nx, Ny, Nz]) / dz
        #     CEy[nx, Ny, Nz] = (0 - Ex[nx, Ny, Nz]) / dz - (Ez[nx + 1, Ny, Nz] - Ez[nx, Ny, Nz]) / dx
        #     CEz[nx, Ny, Nz] = (Ey[nx + 1, Ny, Nz] - Ey[nx, Ny, Nz]) / dx - (0 - Ex[nx, Ny, Nz]) / dy
        # nx == Nx, ny == Ny, nz == Nz
        # CEx[Nx-1, Ny-1, Nz-1] = (0 - Ez[Nx-1, Ny-1, Nz-1]) / dy - (0 - Ey[Nx-1, Ny-1, Nz-1]) / dz
        # CEy[Nx-1, Ny-1, Nz-1] = (0 - Ex[Nx-1, Ny-1, Nz-1]) / dz - (0 - Ez[Nx-1, Ny-1, Nz-1]) / dx
        # CEz[Nx-1, Ny-1, Nz-1] = (0 - Ey[Nx-1, Ny-1, Nz-1]) / dx - (0 - Ex[Nx-1, Ny-1, Nz-1]) / dy


        # calculate the curls of the electric field

        global Ex; global Ey; global Ez

        # inside
        CEx[:Nx, :Ny-1, :Nz-1] = (Ez[:Nx, 1:Ny, :Nz-1] - Ez[:Nx, :Ny-1, :Nz-1]) / dy - (Ey[:Nx, :Ny-1, 1:Nz] - Ey[:Nx, :Ny-1, :Nz-1]) / dz
        CEy[:Nx-1, :Ny, :Nz-1] = (Ex[:Nx-1, :Ny, 1:Nz] - Ex[:Nx-1, :Ny, :Nz-1]) / dz - (Ez[1:Nx, :Ny, :Nz-1] - Ez[:Nx-1, :Ny, :Nz-1]) / dx
        CEz[:Nx-1, :Ny-1, :Nz] = (Ey[1:Nx, :Ny-1, :Nz] - Ey[:Nx-1, :Ny-1, :Nz]) / dx - (Ex[:Nx-1, 1:Ny, :Nz] - Ex[:Nx-1, :Ny-1, :Nz]) / dy


        # nx == Nx
        # CEx[Nx-1, :Ny-1, :Nz-1] = (Ez[Nx-1, 1:Ny, :Nz-1] - Ez[Nx-1, :Ny-1, :Nz-1]) / dy - (Ey[Nx-1, :Ny-1, 1:Nz] - Ey[Nx-1, :Ny-1, :Nz-1]) / dz
        CEy[Nx-1, :Ny, :Nz-1] = (Ex[Nx-1, :Ny, 1:Nz] - Ex[Nx-1, :Ny, :Nz-1]) / dz - (0 - Ez[Nx-1, :Ny, :Nz-1]) / dx
        CEz[Nx-1, :Ny-1, :Nz] = (0 - Ey[Nx-1, :Ny-1, :Nz]) / dx - (Ex[Nx-1, 1:Ny, :Nz] - Ex[Nx-1, :Ny-1, :Nz]) / dy

        # ny == Ny
        CEx[:Nx, Ny-1, :Nz-1] = (0 - Ez[:Nx, Ny-1, :Nz-1]) / dy - (Ey[:Nx, Ny-1, 1:Nz] - Ey[:Nx, Ny-1, :Nz-1]) / dz
        # CEy[:Nx-1, Ny-1, :Nz-1] = (Ex[:Nx-1, Ny-1, :Nz-1] - Ex[:Nx-1, Ny-1, :Nz-1]) / dz - (Ez[1:Nx, Ny-1, :Nz-1] - Ez[:Nx-1, Ny-1, :Nz-1]) / dx
        CEz[:Nx-1, Ny-1, :Nz] = (Ey[1:Nx, Ny-1, :Nz] - Ey[:Nx-1, Ny-1, :Nz]) / dx - (0 - Ex[:Nx-1, Ny-1, :Nz]) / dy

        # nz == Nz
        CEx[:Nx, :Ny-1, Nz-1] = (Ez[:Nx, 1:Ny, Nz-1] - Ez[:Nx, :Ny-1, Nz-1]) / dy - (0 - Ey[:Nx, :Ny-1, Nz-1]) / dz
        CEy[:Nx-1, :Ny, Nz-1] = (0 - Ex[:Nx-1, :Ny, Nz-1]) / dz - (Ez[1:Nx, :Ny, Nz-1] - Ez[:Nx-1, :Ny, Nz-1]) / dx
        # CEz[:Nx-1, :Ny-1, Nz-1] = (Ey[1:Nx, :Ny-1, Nz-1] - Ey[:Nx-1, :Ny-1, Nz-1]) / dx - (Ex[:Nx-1, 1:Ny, Nz-1] - Ex[:Nx-1, :Ny-1, Nz-1]) / dy


        # nx == Nx, Ny == ny, nz = 0 ... Nz-1
        # CEx[Nx-1, Ny-1, :Nz-1] = (0 - Ez[Nx-1, Ny-1, :Nz-1]) / dy - (Ey[Nx-1, Ny-1, 1:Nz] - Ey[Nx-1, Ny-1, :Nz-1]) / dz
        # CEy[Nx-1, Ny-1, :Nz-1] = (Ex[Nx-1, Ny-1, 1:Nz] - Ex[Nx-1, Ny-1, :Nz-1]) / dz - (0 - Ez[Nx-1, Ny-1, :Nz-1]) / dx
        CEz[Nx-1, Ny-1, :Nz] = (0 - Ey[Nx-1, Ny-1, :Nz]) / dx - (0 - Ex[Nx-1, Ny-1, :Nz]) / dy
        # nx == Nx, ny = 0 ... Ny-1, nz == Nz
        # CEx[Nx-1, :Ny-1, Nz-1] = (Ez[Nx-1, 1:Ny, Nz-1] - Ez[Nx-1, :Ny-1, Nz-1]) / dy - (0 - Ey[Nx-1, :Ny-1, Nz-1]) / dz
        CEy[Nx-1, :Ny, Nz-1] = (0 - Ex[Nx-1, :Ny, Nz-1]) / dz - (0 - Ez[Nx-1, :Ny, Nz-1]) / dx
        # CEz[Nx-1, :Ny-1, Nz-1] = (0 - Ey[Nx-1, :Ny-1, Nz-1]) / dx - (Ex[Nx-1, 1:Ny, Nz-1] - Ex[Nx-1, :Ny-1, Nz-1]) / dy
        # nx == 0 ... Nx, ny == Ny nz == Nz
        CEx[:Nx, Ny-1, Nz-1] = (0 - Ez[:Nx, Ny-1, Nz-1]) / dy - (0 - Ey[:Nx, Ny-1, Nz-1]) / dz
        # CEy[:Nx-1, Ny-1, Nz-1] = (0 - Ex[:Nx-1, Ny-1, Nz-1]) / dz - (Ez[1:Nx, Ny-1, Nz-1] - Ez[:Nx-1, Ny-1, Nz-1]) / dx
        # CEz[:Nx-1, Ny-1, Nz-1] = (Ey[1:Nx, Ny-1, Nz-1] - Ey[:Nx-1, Ny-1, Nz-1]) / dx - (0 - Ex[:Nx-1, Ny-1, Nz-1]) / dy
        # nx == Nx, ny == Ny, nz == Nz
        # CEx[Nx-1, Ny-1, Nz-1] = (0 - Ez[Nx-1, Ny-1, Nz-1]) / dy - (0 - Ey[Nx-1, Ny-1, Nz-1]) / dz
        # CEy[Nx-1, Ny-1, Nz-1] = (0 - Ex[Nx-1, Ny-1, Nz-1]) / dz - (0 - Ez[Nx-1, Ny-1, Nz-1]) / dx
        # CEz[Nx-1, Ny-1, Nz-1] = (0 - Ey[Nx-1, Ny-1, Nz-1]) / dx - (0 - Ex[Nx-1, Ny-1, Nz-1]) / dy


        # update magnetic field H

        global Hx; global Hy; global Hz

        Hx += np.multiply(-mHx, CEx)
        Hy += np.multiply(-mHy, CEy)
        Hz += np.multiply(-mHz, CEz)


        # calculate the curls of the magnetic field

        # inside
        CHx[0:Nx, 1:Ny, 1:Nz] = (Hz[0:Nx, 1:Ny, 1:Nz] - Hz[0:Nx, 0:Ny-1, 1:Nz]) / dy - (Hy[0:Nx, 1:Ny, 1:Nz] - Hy[0:Nx, 1:Ny, 0:Nz-1]) / dz
        CHy[1:Nx, 0:Ny, 1:Nz] = (Hx[1:Nx, 0:Ny, 1:Nz] - Hx[1:Nx, 0:Ny, 0:Nz-1]) / dz - (Hz[1:Nx, 0:Ny, 1:Nz] - Hz[0:Nx-1, 0:Ny, 1:Nz]) / dx
        CHz[1:Nx, 1:Ny, 0:Nz] = (Hy[1:Nx, 1:Ny, 0:Nz] - Hy[0:Nx-1, 1:Ny, 0:Nz]) / dx - (Hx[1:Nx, 1:Ny, 0:Nz] - Hx[1:Nx, 0:Ny-1, 0:Nz]) / dy
        # nx == 0
        # CHx[0, 1:Ny, 1:Nz] = (Hz[0, 1:Ny, 1:Nz] - Hz[0, 0:Ny-1, 1:Nz]) / dy - (Hy[0, 1:Ny, 1:Nz] - Hy[0, 1:Ny, 0:Nz-1]) / dz # can be included
        CHy[0, 0:Ny, 1:Nz] = (Hx[0, 0:Ny, 1:Nz] - Hx[0, 0:Ny, 0:Nz-1]) / dz - (Hz[0, 0:Ny, 1:Nz] - 0) / dx
        CHz[0, 1:Ny, 0:Nz] = (Hy[0, 1:Ny, 0:Nz] - 0) / dx - (Hx[0, 1:Ny, 0:Nz] - Hx[0, 0:Ny-1, 0:Nz]) / dy
        # ny == 0
        CHx[0:Nx, 0, 1:Nz] = (Hz[0:Nx, 0, 1:Nz] - 0) / dy - (Hy[0:Nx, 0, 1:Nz] - Hy[0:Nx, 0, 0:Nz-1]) / dz
        # CHy[1:Nx, 0, 1:Nz] = (Hx[1:Nx, 0, 1:Nz] - Hx[1:Nx, 0, 0:Nz-1]) / dz - (Hz[1:Nx, 0, 1:Nz] - Hz[0:Nx-1, 0, 1:Nz]) / dx # can be included into the "inside"
        CHz[1:Nx, 0, 0:Nz] = (Hy[1:Nx, 0, 0:Nz] - Hy[0:Nx-1, 0, 0:Nz]) / dx - (Hx[1:Nx, 0, 0:Nz] - 0) / dy
        # nz == 0
        CHx[0:Nx, 1:Ny, 0] = (Hz[0:Nx, 1:Ny, 0] - Hz[0:Nx, 0:Ny-1, 0]) / dy - (Hy[0:Nx, 1:Ny, 0] - 0) / dz
        CHy[1:Nx, 0:Ny, 0] = (Hx[1:Nx, 0:Ny, 0] - 0) / dz - (Hz[1:Nx, 0:Ny, 0] - Hz[0:Nx-1, 0:Ny, 0]) / dx
        # CHz[1:Nx, 1:Ny, 0] = (Hy[1:Nx, 1:Ny, 0] - Hy[0:Nx-1, 1:Ny, 0]) / dx - (Hx[1:Nx, 1:Ny, 0] - Hx[1:Nx, 0:Ny-1, 0]) / dy # can be included in inside case
        # nx == 0, ny == 0, nz = 1 ... Nz
        # CHx[0, 0, 1:Nz] = (Hz[0, 0, 1:Nz] - 0) / dy - (Hy[0, 0, 1:Nz] - Hy[0, 0, 0:Nz-1]) / dz
        # CHy[0, 0, 1:Nz] = (Hx[0, 0, 1:Nz] - Hx[0, 0, 0:Nz-1]) / dz - (Hz[0, 0, 1:Nz] - 0) / dx
        CHz[0, 0, 0:Nz] = (Hy[0, 0, 0:Nz] - 0) / dx - (Hx[0, 0, 0:Nz] - 0) / dy
        # nx == 0, ny == 1 ... Ny, nz == 0
        # CHx[0, 1:Ny, 0] = (Hz[0, 1:Ny, 0] - Hz[0, 0:Ny-1, 0]) / dy - (Hy[0, 1:Ny, 0] - 0) / dz
        CHy[0, 0:Ny, 0] = (Hx[0, 0:Ny, 0] - 0) / dz - (Hz[0, 0:Ny, 0] - 0) / dx
        # CHz[0, 1:Ny, 0] = (Hy[0, 1:Ny, 0] - 0) / dx - (Hx[0, 1:Ny, 0] - Hx[0, 0:Ny-1, 0]) / dy
        # nx == 1 ... Nx, ny == 0, nz == 0
        CHx[0:Nx, 0, 0] = (Hz[0:Nx, 0, 0] - 0) / dy - (Hy[0:Nx, 0, 0] - 0) / dz
        # CHy[1:Nx, 0, 0] = (Hx[1:Nx, 0, 0] - 0) / dz - (Hz[1:Nx, 0, 0] - Hz[0:Nx-1, 0, 0]) / dx
        # CHz[1:Nx, 0, 0] = (Hy[1:Nx, 0, 0] - Hy[0:Nx-1, 0, 0]) / dx - (Hx[1:Nx, 0, 0] - 0) / dy
        # nx == 0, ny == 0, nz == 0
        # CHx[0, 0, 0] = (Hz[0, 0, 0] - 0) / dy - (Hy[0, 0, 0] - 0) / dz
        # CHy[0, 0, 0] = (Hx[0, 0, 0] - 0) / dz - (Hz[0, 0, 0] - 0) / dx
        # CHz[0, 0, 0] = (Hy[0, 0, 0] - 0) / dx - (Hx[0, 0, 0] - 0) / dy


        # update electric flux density D

        global Dx; global Dy; global Dz
        Dx += (sp.constants.c * dt) * CHx
        Dy += (sp.constants.c * dt) * CHy
        Dz += (sp.constants.c * dt) * CHz

        # add source
        Dz[x_src, y_src, z_src] += Jz[n]  # source has to have same dimensionality


        # update electric field E (normalized, has units of magnetic field H)


        Ex = np.divide(Dx, eps_xx) # element-wise
        Ey = np.divide(Dy, eps_yy)
        Ez = np.divide(Dz, eps_zz)



        return Ex, Ey, Ez, Hx, Hy, Hz






"""








