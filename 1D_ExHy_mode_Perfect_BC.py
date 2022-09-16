import numpy as np
import scipy as sp
from scipy import constants
import matplotlib.pyplot as plt
import matplotlib.animation as animation


plt.rcParams['text.usetex'] = True


# gaussian source - this is to sampled in time for all the time steps
def g(t, t0, tau):
    return np.exp(-(t - t0)**2 / tau**2)


if __name__=="__main__":

    Nz = 600 # total number of spatial cells []
    Nt = 1000 # total num of time steps []
    f = 3e+8  # target frequency we are interested in having in our model, [Hz]
    num_dz_per_wavelen = 10 # rule of thumb []
    tau = 1/2/f # this is the width of the Gaussian pulse that we calculate according to the maximum frequency we want to account for
    t0 = 6*tau # "ease" the source into the computational domain - start at the point

    lam = sp.constants.c / f # wavelength [m]
    dz = lam / num_dz_per_wavelen  # size of a spatial cell [m]
    # the wave propagates one cell in exactly one time step; CFL stab cond satisfied
    dt = dz / 2 / sp.constants.c  # CFL stability condition [s]
    omega = 2 * np.pi * f # angular frequency [Hz]
    k = omega / sp.constants.c # wave number [1/m]

    Ey = np.zeros(Nz) # E(0) .... E(Nz) # for PEC BC
    Hx = np.zeros(Nz) # H(0) .... H(Nz) # for PMC BC
    Mx = np.zeros(Nt) # magnetic current density -- variation in time at the source point
    Jy = np.zeros(Nt) # electric current density -- variation in time at the source point
    T = np.arange(0, Nt*dt, dt)
    for n in range(Nt):
        Jy[n] = g(T[n], t0, tau)
    # for efficiency Jy *= eta_0

    Z = np.arange(0, (Nz)*dz, dz)

    eps_yy = np.ones(Nz)
    mu_xx = np.ones(Nz)
    eta_0 = np.sqrt(sp.constants.mu_0/sp.constants.epsilon_0)

    mEy = sp.constants.c * dt / eps_yy
    mHx = sp.constants.c * dt / mu_xx


    # auxiliary variables for ABC
    h1 = float(0.0)
    h2 = float(0.0)
    h3 = float(0.0)
    e1 = float(0.0)
    e2 = float(0.0)
    e3 = float(0.0)

    def time_step_perfect(n):
        assert(Ey.shape == Hx.shape)
        Hx[0:Nz - 1] += mHx[0:Nz - 1] * (Ey[1:Nz] - Ey[0:Nz - 1]) / dz - mHx[0:Nz - 1] * Mx[n]
        # ABC at the right boundary
        global e3
        global e2
        global e1
        e3 = e2
        e2 = e1
        e1 = Ey[Nz-1]
        Hx[Nz - 1] += mHx[Nz - 1] * (e3 - Ey[Nz - 1]) / dz - mHx[Nz - 1] * Mx[n]

        n_bc = 1 # refractive index at both boundaries should be the same for ABC to work
        # Ey[Nz+1] = Ey_prev[Nz]

        Ey[1:Nz] += mEy[1:Nz] * (Hx[1:Nz] - Hx[0:Nz - 1]) / dz
        # ABC at the left boundary
        global h1
        global h2
        global h3
        h3 = h2
        h2 = h1
        h1 = Hx[0]
        Ey[0] += mEy[0] * (Hx[0] - h3) / dz

        # source
        Ey[int(Nz / 3)] -= mEy[int(Nz / 3)] * eta_0 * Jy[n]

        return Ey, Hx


    # wrap up into a function for animation

    fig, ax = plt.subplots()
    ax.plot([int(Nz/3)*dz, int(Nz/3)*dz], [-50, 50], color='green')
    plt.text(int(Nz/3)*dz+1,-48, 'source \n location', fontsize=8)
    line0, = ax.plot(np.arange(0, (Nz)*dz, dz), Ey, c='b', label=r'$E_y,~[\frac{V}{m}]$')
    line1, = ax.plot(np.arange(0, (Nz)*dz, dz)+dz/2, Hx, c='r', label=r'$\tilde{H}_x = \eta_0 H_x = E_y,~[\frac{V}{m}]$')
    plt.xlabel('distance, m')
    plt.ylabel('amplitude of Ey, V/m')

    plt.ylim([-50, 50])
    plt.legend()
    plt.show()


    def anim_init():
        line0.set_ydata(Ey)
        ax.set_title("time={} sec".format(0))

        return line0, line1,


    def animate(n):
        Ey, Hx= time_step_perfect(n)
        line0.set_ydata(Ey)  # update the data
        line1.set_ydata(Hx)  # update the data
        ax.set_title("time={:.2f} nsec".format(n*dt*1e+9))
        return line0, line1


    ani = animation.FuncAnimation(fig=fig, func=animate, frames=Nt, fargs=(), init_func=anim_init, blit=True)

    f = r"./1D_EyHx_mode_Perfect_BC.mp4"
    writervideo = animation.FFMpegWriter(fps=20)
    ani.save(f, writer=writervideo)

