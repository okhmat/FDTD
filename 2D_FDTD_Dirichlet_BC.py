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


### this code is for Ez mode -- we have magnetic field components in xy-plane: Hx and Hy
# so, we need 3 2-dimensional arrays

if __name__=="__main__":

    Nx = 200  # total number of spatial cells in x direction
    Ny = 200
    Nt = 400  # total num of time steps
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
        Jz[n] = g(T[n], t0, tau)

    X = np.arange(0, (Nx)*dx, dx)
    Y = np.arange(0, (Ny) * dy, dy)


    mu_xx = np.ones((Nx, Nx))
    mu_yy = np.ones((Ny, Ny))
    eps_zz = np.ones((Nx, Nx)) # Nz == Nx
    eta_0 = np.sqrt(sp.constants.mu_0 / sp.constants.epsilon_0)

    mHx = sp.constants.c * dt / mu_xx # coefficients in
    mHy = sp.constants.c * dt / mu_yy

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
        # inner points
        for nx in range(1, Nx):
            for ny in range(1, Ny):
                CHz[nx, ny] = (Hy[nx, ny] - Hy[nx-1, ny]) / dx - (Hx[nx, ny] - Hx[nx, ny-1]) / dy
        # points at the lower boundary
        for ny in range(1, Ny):
            CHz[0, ny] = (Hy[0, ny] - 0) / dx - (Hx[0, ny] - Hx[0, ny-1]) / dy
        # points at the left boundary
        for nx in range(1, Nx):
            CHz[nx, 0] = (Hy[nx, 0] - Hy[nx-1, 0]) / dx - (Hx[nx, 0] - 0) / dy
        # point at the left bottom corner
        CHz[0, 0] = (Hy[0, 0] - 0) / dx - (Hx[0, 0] - 0) / dy

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















