import numba
from numba import jit
import numpy as np
import scipy as sp
from scipy import constants
import matplotlib.pyplot as plt
import matplotlib.animation as animation



plt.rcParams['text.usetex'] = True


def g(t, t0, tau):
    return np.exp(-(t - t0)**2 / tau**2)

# if we use gaussian source we naturally acquire a DC component of the E field. To avoid this we introduce the source
# that averages to zero
def src_no_dc(t, tau):
    return (t / tau)**3 * np.exp(-t / tau) * (4 - t/tau)




if __name__=="__main__":

    # I want to set up the size of the cavity
    L = 6 # meters
    # then consider max frequency I want to resolve (based on the mode of the cavity)
    f = 3e+8
    # from the frequency calculate the wavelength
    lam = sp.constants.c / f
    dx = lam / 25
    dy = dx
    dz = dx
    Nx = int(np.ceil(L / dx))
    Ny = Nx
    Nz = Nx
    omega = 2*np.pi*f
    k = omega / sp.constants.c

    # parameters of the Gaussian source
    tau = .5 / f
    t0 = 6 * tau

    Nt = 100

    dt = dx / np.sqrt(3)/ sp.constants.c # CFL

    print("target frequency (MHz): ", f/1e+6)
    print("size of the cavity (m): ", dx*Nx)
    print("wavelength / 2 (m):", lam/2)

    print("number of spatial cells Nx: ", Nx)
    print("number of spatial cells Ny: ", Ny)
    print("number of spatial cells Nz: ", Nz)



    Ex = np.zeros((Nx+1, Ny+1, Nz+1))
    Ey = np.zeros((Nx+1, Ny+1, Nz+1))
    Ez = np.zeros((Nx+1, Ny+1, Nz+1)) # E(0) .... E(Nz) # for PEC BC

    Dx = np.zeros((Nx+1, Ny+1, Nz+1))
    Dy = np.zeros((Nx+1, Ny+1, Nz+1))
    Dz = np.zeros((Nx+1, Ny+1, Nz+1))

    Hx = np.zeros((Nx, Ny, Nz)) # H(0) .... H(Nz) # for PMC BC
    Hy = np.zeros((Nx, Ny, Nz))
    Hz = np.zeros((Nx, Ny, Nz))

    J = np.zeros(Nt)  # electric current density -- variation in time at the source point

    T = np.arange(0, Nt * dt, dt)
    for n in range(Nt):
        # J[n] = np.cos(omega * T[n])
        # J[n] = 6*src_no_dc(T[n], tau)
        J[n] = 10 * g(t=T[n], t0=t0, tau=tau)
    X = np.arange(0, (Nx) * dx, dx)
    Y = np.arange(0, (Ny) * dy, dy)
    Z = np.arange(0, (Ny) * dz, dz)

    # consider that there are no off-diagonal components of eps and mu
    mu_xx = np.ones((Nx, Nx, Nz))
    mu_yy = np.ones((Ny, Ny, Nz))
    mu_zz = np.ones((Ny, Ny, Nz))
    eps_xx = np.ones((Nx+1, Ny+1, Nz+1)) # Nz == Nx
    eps_yy = np.ones((Nx+1, Ny+1, Nz+1))
    eps_zz = np.ones((Nx+1, Ny+1, Nz+1))
    eta_0 = np.sqrt(sp.constants.mu_0 / sp.constants.epsilon_0)

    mHx = sp.constants.c * dt / mu_xx
    mHy = sp.constants.c * dt / mu_yy
    mHz = sp.constants.c * dt / mu_zz

    CEx = np.zeros((Nx, Ny, Nz))
    CEy = np.zeros((Nx, Ny, Nz))
    CEz = np.zeros((Nx, Ny, Nz))

    CHx = np.zeros((Nx+1, Ny+1, Nz+1))
    CHy = np.zeros((Nx+1, Ny+1, Nz+1))
    CHz = np.zeros((Nx+1, Ny+1, Nz+1))

    x_src = int(Nx / 3)
    y_src = int(Ny / 4)
    z_src = int(Nz / 5)



    # for Fourier transform:
    Nf = 1000 # number of frequencies we are interested in
    FREQ = np.linspace(0, 10*f, Nf)
    # kernel for the FT
    K = np.exp(-2j*np.pi*np.outer(FREQ, T)) # 1 kernel for each frequency

    Ez_fft = np.zeros((Nf, Nx+1), dtype=complex)
    Ezn = np.zeros((Nt, Nx+1))


    # @jit
    def time_step_pec(n): # TODO: cross-check with lecture 15 slide 12

        global Ex
        global Ey
        global Ez
        CEx[0:Nx, 0:Ny, 0:Nz] = (Ez[0:Nx, 1:Ny+1, 0:Nz] - Ez[0:Nx, 0:Ny, 0:Nz]) / dy - (Ey[0:Nx, 0:Ny, 1:Nz+1] - Ey[0:Nx, 0:Ny, 0:Nz]) / dz
        CEy[0:Nx, 0:Ny, 0:Nz] = (Ex[0:Nx, 0:Ny, 1:Nz+1] - Ex[0:Nx, 0:Ny, 0:Nz]) / dz - (Ez[1:Nx+1, 0:Ny, 0:Nz] - Ez[0:Nx, 0:Ny, 0:Nz]) / dx
        CEz[0:Nx, 0:Ny, 0:Nz] = (Ey[1:Nx+1, 0:Ny, 0:Nz] - Ey[0:Nx, 0:Ny, 0:Nz]) / dx - (Ex[0:Nx, 1:Ny+1, 0:Nz] - Ex[0:Nx, 0:Ny, 0:Nz]) / dy

        global Hx
        global Hy
        global Hz
        Hx += np.multiply(-mHx, CEx)
        Hy += np.multiply(-mHy, CEy)
        Hz += np.multiply(-mHz, CEz)

        CHx[0:Nx, 1:Ny, 1:Nz] = (Hz[0:Nx, 1:Ny, 1:Nz] - Hz[0:Nx, 0:Ny-1, 1:Nz]) / dy - (Hy[0:Nx, 1:Ny, 1:Nz] - Hy[0:Nx, 1:Ny, 0:Nz-1]) / dz
        CHy[1:Nx, 0:Ny, 1:Nz] = (Hx[1:Nx, 0:Ny, 1:Nz] - Hx[1:Nx, 0:Ny, 0:Nz-1]) / dz - (Hz[1:Nx, 0:Ny, 1:Nz] - Hz[0:Nx-1, 0:Ny, 1:Nz]) / dx
        CHz[1:Nx, 1:Ny, 0:Nz] = (Hy[1:Nx, 1:Ny, 0:Nz] - Hy[0:Nx-1, 1:Ny, 0:Nz]) / dx - (Hx[1:Nx, 1:Ny, 0:Nz] - Hx[1:Nx, 0:Ny-1, 0:Nz]) / dy

        global Dx
        global Dy
        global Dz
        Dx += (sp.constants.c * dt) * CHx
        Dy += (sp.constants.c * dt) * CHy
        Dz += (sp.constants.c * dt) * CHz

        # Dx[x_src, y_src, z_src] += J[n]
        # Dy[x_src, y_src, z_src] += J[n]
        Dz[x_src, y_src, z_src] += J[n]

        Ex = np.divide(Dx, eps_xx)
        Ey = np.divide(Dy, eps_yy)
        Ez = np.divide(Dz, eps_zz)

        return Ex, Ey, Ez, Hx, Hy, Hz



    # -------------------------------------------------------------
    # lazy FFT
    # -------------------------------------------------------------

    # for nt in range(Nt):
    #     time_step_pec(nt)
    #     # add Ez(x) component to the array
    #     Ezn[nt, :] = Ez[:, y_src, z_src]
    #
    # aux = np.real(np.fft.fft(Ezn[:, 50]))
    #
    # plt.plot(aux)
    # plt.show()
    # plt.plot(np.real(Ezn[:, 50]))
    # plt.show()

    # -------------------------------------------------------------

    # let's visualize a slice first

    fig = plt.figure(figsize=(8, 8))
    imag = plt.imshow(Ez[:, :, z_src], vmin=-0.03, vmax=0.03)
    plt.show()


    def anim_init():
        imag.set_data(Ez[:, :, z_src])
        # plt.set_title("time={} sec".format(0))
        return imag


    def animate(n):
        time_step_pec(n)
        imag.set_data(Ez[:, :, z_src])  # update the data
        # plt.set_title("time={:.2f} nsec".format(n*dt*1e+9))
        return imag


    ani = animation.FuncAnimation(fig=fig, func=animate, frames=Nt, fargs=(), init_func=anim_init)


    f = r"./3D_Dirichlet_BC.mp4"
    writervideo = animation.FFMpegWriter(fps=20)
    ani.save(f, writer=writervideo)
