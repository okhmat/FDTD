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

    Nz = 400 # total number of spatial cells []
    Nt = 100 # total num of time steps []
    f = 3  # target frequency we are interested in having in our model, [Hz]
    num_dz_per_wavelen = 10 # rule of thumb []
    tau = 1/2/f # this is the width of the Gaussian pulse that we calculate according to the maximum frequency we want to account for
    t0 = 6*tau # "ease" the source into the computational domain - start at the point
    eta_0 = np.sqrt(sp.constants.mu_0/sp.constants.epsilon_0)

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
    Jy[:] = g(T[:], t0, tau) * eta_0
    # for efficiency Jy *= eta_0

    Z = np.arange(0, (Nz)*dz, dz)

    # relative permittivities and permeabilities along the grid
    eps_yy = np.ones(Nz)
    mu_xx = np.ones(Nz)
    n_src = 1 # refractive index at the source cell

    mEy = sp.constants.c * dt / eps_yy
    mHx = sp.constants.c * dt / mu_xx

    N_src = int(Nz/3)
    z_src = N_src*dz


    # Total field / Scattered field correction terms
    Ey_src = Jy
    # Hx_src = np.zeros(Nt)
    Hx_src = -np.sqrt(eps_yy[N_src-1]/mu_xx[N_src-1])*g(T[:]+n_src*dz/2/sp.constants.c + 0.5*dt, t0, tau)


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
        Ey[N_src] -= mEy[N_src] * Jy[n]

        # corrections for the Total field / Scattered field source ---- this implementation works, but for lower frequency - there's something going back
        Ey[N_src] -= Hx[N_src-1] * mEy[N_src] / dz / eta_0 # in the total field region # Hx / eta_0 = real value of Hx, without normalization
        Hx[N_src-1] -= Ey[N_src] * mHx[N_src-1] / dz # in the scattered field region

        # approach 2 - doesn't work
        # Hx[N_src-1] -= (mHx[N_src-1]/dz)*Ey_src[n]/eta_0
        # Ey[N_src] -= (mEy[N_src]/dz)*Hx_src[n]


        return Ey, Hx


    # wrap up into a function for animation

    fig, ax = plt.subplots()
    ax.plot([z_src, z_src], [-900, 900], color='green')
    plt.text(z_src+1,-88, 'source \n location', fontsize=8)
    line0, = ax.plot(np.arange(0, (Nz)*dz, dz), Ey, c='b', label=r'$E_y,~[\frac{V}{m}]$')
    line1, = ax.plot(np.arange(0, (Nz)*dz, dz)+dz/2, Hx, c='r', label=r'$\tilde{H}_x = \eta_0 H_x = E_y,~[\frac{V}{m}]$')
    plt.xlabel('distance, m')
    plt.ylabel('amplitude of Ey, V/m')

    plt.ylim([-900, 900])
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

    f = r"./1D_EyHx_mode_TF_SF_src.mp4"
    writervideo = animation.FFMpegWriter(fps=60)
    ani.save(f, writer=writervideo)

