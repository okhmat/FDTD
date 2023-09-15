import numba
from numba import jit
import numpy as np
import scipy as sp
from scipy import constants
import matplotlib.pyplot as plt
import matplotlib.animation as animation
plt.rcParams['text.usetex'] = True


'''
            The workflow of the FDTD
        
        ________________________________________________________________    
        Define device parameters
        ||
        Define FDTD parameters
        ||
        Compute grid parameters
        ||
        Build device on grid
        ||
        Compute time step
        ||
        Compute source values at the time steps
        |
        Compute PML parameters
        |
        Compute update coeffcients
        |
        Initialize fields
        |
        Initialize curl arrays
        |
        Initialize integration arrays
        |_______________________________________________________________
        
        
        
        Then we enter the main loop
        
        Compute curl of E
        |
        Update H integrations
        |
        Update H field
        |
        Compute curl of H field
        |
        Update D integrations 
        |
        Update D
        |
        Inject source
        |
        Update E
        |
        Visualize fields (some callback that would return )
        
        



'''

def g(t, t0, tau):
    return np.exp(-(t - t0)**2 / tau**2)

# if we use gaussian source we naturally acquire a DC component of the E field. To avoid this we introduce the source
# that averages to zero
def src_no_dc(t, tau):
    return (t / tau)**3 * np.exp(-t / tau) * (4 - t/tau)

# class Source:
#     pass
#
# class Device:
#     def __init__(self, epsr):
#         self.x_len = 10
#         self.y_len = 10
#         self.z_len = 10
#         self.epsr = epsr
#
#
#
# class FDTD:
#     def __init__(self, device, time, output):
#         self.T = 1



if __name__=="__main__":


    '''
        Define device parameters
        ||
        Define FDTD parameters
        ||
        Compute grid parameters
        ||
        Build device on grid
        ||
        Compute time step
        ||
    '''

    # I want to set up the size of the cavity
    Lx = 6 # meters
    Ly = 6  # meters
    Lz = 6  # meters
    # highest frequency (shortest wavelength) I want to resolve in my simulation
    f = 3e+8
    # from the frequency calculate the shortest wavelength
    lam = sp.constants.c / f
    # based on the wavelength we compute the resolution of the grid; rule of thumb is 10 spatial cells per wavelength
    cells_per_wavelength_x = 10
    cells_per_wavelength_y = 10
    cells_per_wavelength_z = 10
    dx = lam / cells_per_wavelength_x
    dy = lam / cells_per_wavelength_y
    dz = lam / cells_per_wavelength_z
    # round up number of spatial cells; we will revise spatial resolutions based on these numbers
    Nx = int(np.ceil(Lx / dx))
    Ny = int(np.ceil(Ly / dy))
    Nz = int(np.ceil(Lz / dz))

    # adjust spatial steps according to their number and the physical size of the domain (L)
    dx = Lx / Nx
    dy = Ly / Ny
    dz = Lz / Nz



    omega = 2*np.pi*f
    k = omega / sp.constants.c

    # parameters of the Gaussian source
    tau = .5 / f
    t0 = 6 * tau

    Nt = 300

    # generalized CFL condition
    n_min = 1 # minimum refractive index in the simulation -- related to PML parameters - how?
    # dt <= n_min / c0 * sqrt(1/dx^2 + ...)
    dt = n_min / sp.constants.c / np.sqrt(1/dx**2 + 1/dy**2 + 1/dz**2) / 1.5
    # dt = dx / np.sqrt(3) / sp.constants.c / 1.5 # CFL

    print("target frequency (MHz): ", f/1e+6)
    print("size of the cavity (m): ", dx*Nx)
    print("wavelength / 2 (m):", lam/2)

    print("number of spatial cells Nx: ", Nx)
    print("number of spatial cells Ny: ", Ny)
    print("number of spatial cells Nz: ", Nz)

    print("Number of time steps: ", Nt)


    Ex = np.zeros((Nx, Ny, Nz))
    Ey = np.zeros((Nx, Ny, Nz))
    Ez = np.zeros((Nx, Ny, Nz)) # E(0) .... E(Nz) # for PEC BC
    # E_z_exact = np.zeros((Nx, Ny, Nz))

    Dx = np.zeros((Nx, Ny, Nz))
    Dy = np.zeros((Nx, Ny, Nz))
    Dz = np.zeros((Nx, Ny, Nz))

    Hx = np.zeros((Nx, Ny, Nz)) # H(0) .... H(Nz) # for PMC BC
    Hy = np.zeros((Nx, Ny, Nz))
    Hz = np.zeros((Nx, Ny, Nz))

    J = np.zeros(Nt)  # electric current density -- variation in time at the source point

    T = np.arange(0, Nt * dt, dt)
    for n in range(Nt):
        J[n] = 5*np.sin(2 * np.pi * f * dt * n) # should introduce a single frequency into the domain
        # J[n] = 10*src_no_dc(T[n], tau)
        # J[n] = 10 * g(t=T[n], t0=t0, tau=tau)
    X = np.arange(0, (Nx) * dx, dx)
    Y = np.arange(0, (Ny) * dy, dy)
    Z = np.arange(0, (Ny) * dz, dz)

    # consider that there are no off-diagonal components of eps and mu
    mu_xx = np.ones((Nx, Ny, Nz))
    mu_yy = np.ones((Nx, Ny, Nz))
    mu_zz = np.ones((Nx, Ny, Nz))
    eps_xx = np.ones((Nx, Ny, Nz))
    eps_yy = np.ones((Nx, Ny, Nz))
    eps_zz = np.ones((Nx, Ny, Nz))
    eta_0 = np.sqrt(sp.constants.mu_0 / sp.constants.epsilon_0)

    mHx = sp.constants.c * dt / mu_xx
    mHy = sp.constants.c * dt / mu_yy
    mHz = sp.constants.c * dt / mu_zz

    CEx = np.zeros((Nx, Ny, Nz))
    CEy = np.zeros((Nx, Ny, Nz))
    CEz = np.zeros((Nx, Ny, Nz))

    CHx = np.zeros((Nx, Ny, Nz))
    CHy = np.zeros((Nx, Ny, Nz))
    CHz = np.zeros((Nx, Ny, Nz))

    x_src = int(Nx / 3)
    y_src = int(Ny / 4)
    z_src = int(Nz / 2)



    ''' 
        For Fourier transform:
    Nf = 1000 # number of frequencies we are interested in
    FREQ = np.linspace(0, 10*f, Nf)
    # kernel for the FT
    K = np.exp(-2j*np.pi*np.outer(FREQ, T)) # 1 kernel for each frequency

    Ez_fft = np.zeros((Nf, Nx+1), dtype=complex)
    Ezn = np.zeros((Nt, Nx+1))
    '''

    '''
    here I initialize PML parameters and the fictitious conductivity matrices sigma
    '''

    Nx2 = 2*Nx
    Ny2 = 2*Ny
    Nz2 = 2*Nz

    NPML = [16, 16, 16, 16, 16, 16] # x_low, x_high, y_low, y_high, z_low, z_high

    sigx = np.zeros((Nx2, Ny2, Nz2))
    for nx in range(2*NPML[0]): # low x side
        nx1 = 2*NPML[0] - nx - 1
        sigx[nx1, :, :] = (0.5*sp.constants.epsilon_0/dt) * (((nx+1)/2)/NPML[0])**3
    for nx in range(2*NPML[1]): # high x side
        nx1 = Nx2 - 2*NPML[1] + nx
        sigx[nx1, :, :] = (0.5*sp.constants.epsilon_0/dt) * (((nx+1)/2)/NPML[1])**3

    sigy = np.zeros((Nx2, Ny2, Nz2))
    for ny in range(2*NPML[2]): # low y side
        ny1 = 2*NPML[2] - ny - 1
        sigy[:, ny1, :] = (0.5*sp.constants.epsilon_0/dt) * (((ny+1)/2)/NPML[2])**3
    for ny in range(2*NPML[3]):
        ny1 = Ny2 - 2*NPML[3] + ny
        sigy[:, ny1, :] = (0.5 * sp.constants.epsilon_0 / dt) * (((ny+1) / 2) / NPML[3]) ** 3

    sigz = np.zeros((Nx2, Ny2, Nz2))
    for nz in range(2 * NPML[4]):  # low y side
        nz1 = 2 * NPML[4] - nz - 1
        sigz[:, :, nz1] = (0.5 * sp.constants.epsilon_0 / dt) * (((nz + 1) / 2) / NPML[4]) ** 3
    for nz in range(2 * NPML[5]):
        nz1 = Nz2 - 2 * NPML[5] + nz
        sigz[:, :, nz1] = (0.5 * sp.constants.epsilon_0 / dt) * (((nz + 1) / 2) / NPML[5]) ** 3

    # sigx *= 0
    # sigy *= 0
    # sigz *= 0


    '''
    Then I need to calculate the coefficients mHx for the update equations 
    '''
    sigHx = sigx[0:Nx2:2, 1:Ny2:2, 1:Nz2:2]
    sigHy = sigy[0:Nx2:2, 1:Ny2:2, 1:Nz2:2]
    sigHz = sigz[0:Nx2:2, 1:Ny2:2, 1:Nz2:2]

    mHx0 = (1 / dt) + (sigHy + sigHz) / (2 * sp.constants.epsilon_0) + (0.25 * dt / sp.constants.epsilon_0**2) * np.multiply(sigHy, sigHz)

    mHx1 = (1 / dt) - (sigHy + sigHz) / (2 * sp.constants.epsilon_0) - (0.25 * dt / sp.constants.epsilon_0**2) * np.multiply(sigHy, sigHz)
    mHx1 = np.divide(mHx1, mHx0)

    mHx2 = np.divide(-sp.constants.c, mu_xx)
    mHx2 = np.divide(mHx2, mHx0)

    mHx3 = (-sp.constants.c * dt / sp.constants.epsilon_0) * np.divide(sigHx, mu_xx)
    mHx3 = np.divide(mHx3, mHx0)

    mHx4 = (-dt / sp.constants.epsilon_0**2) * np.multiply(sigHy, sigHz)
    mHx4 = np.divide(mHx4, mHx0)


    '''
        ... and update coefficients mHy for the update equation for Hy 
        for y I shift x and z components on
    '''
    sigHx = sigx[1:Nx2:2, 0:Ny2:2, 1:Nz2:2]
    sigHy = sigy[1:Nx2:2, 0:Ny2:2, 1:Nz2:2]
    sigHz = sigz[1:Nx2:2, 0:Ny2:2, 1:Nz2:2]

    mHy0 = (1 / dt) + (sigHx + sigHz) / (2 * sp.constants.epsilon_0) + (
                0.25 * dt / sp.constants.epsilon_0 ** 2) * np.multiply(sigHx, sigHz)

    mHy1 = (1 / dt) - (sigHx + sigHz) / (2 * sp.constants.epsilon_0) - (
                0.25 * dt / sp.constants.epsilon_0 ** 2) * np.multiply(sigHx, sigHz)
    mHy1 = np.divide(mHy1, mHy0)

    mHy2 = np.divide(-sp.constants.c, mu_yy)
    mHy2 = np.divide(mHy2, mHy0)

    mHy3 = (-sp.constants.c * dt / sp.constants.epsilon_0) * np.divide(sigHy, mu_yy)
    mHy3 = np.divide(mHy3, mHy0)

    mHy4 = (-dt / sp.constants.epsilon_0 ** 2) * np.multiply(sigHx, sigHz)
    mHy4 = np.divide(mHy4, mHy0)


    '''
            ... and update coefficients mHz for the update equation for Hz 
            for z I shift x and y components on
        '''
    sigHx = sigx[1:Nx2:2, 1:Ny2:2, 0:Nz2:2]
    sigHy = sigy[1:Nx2:2, 1:Ny2:2, 0:Nz2:2]
    sigHz = sigz[1:Nx2:2, 1:Ny2:2, 0:Nz2:2]

    mHz0 = (1 / dt) + (sigHx + sigHy) / (2 * sp.constants.epsilon_0) + (
            0.25 * dt / sp.constants.epsilon_0 ** 2) * np.multiply(sigHx, sigHy)

    mHz1 = (1 / dt) - (sigHx + sigHy) / (2 * sp.constants.epsilon_0) - (
            0.25 * dt / sp.constants.epsilon_0 ** 2) * np.multiply(sigHx, sigHy)
    mHz1 = np.divide(mHz1, mHz0)

    mHz2 = np.divide(-sp.constants.c, mu_zz)
    mHz2 = np.divide(mHz2, mHz0)

    mHz3 = (-sp.constants.c * dt / sp.constants.epsilon_0) * np.divide(sigHz, mu_zz)
    mHz3 = np.divide(mHz3, mHz0)

    mHz4 = (-dt / sp.constants.epsilon_0 ** 2) * np.multiply(sigHx, sigHy)
    mHz4 = np.divide(mHz4, mHz0)



    '''
    then same update coefficients for the Dx field
    '''
    sigDx = sigx[0:Nx2:2, 0:Ny2:2, 0:Nz2:2]
    sigDy = sigy[0:Nx2:2, 0:Ny2:2, 0:Nz2:2]
    sigDz = sigz[0:Nx2:2, 0:Ny2:2, 0:Nz2:2]


    mDx0 = (1 / dt) + (sigDy + sigDz) / (2 * sp.constants.epsilon_0) + (
            0.25 * dt / sp.constants.epsilon_0 ** 2) * np.multiply(sigDy, sigDz)

    mDx1 = (1 / dt) - (sigDy + sigDz) / (2 * sp.constants.epsilon_0) - (
                0.25 * dt / sp.constants.epsilon_0 ** 2) * np.multiply(sigDy, sigDz)
    mDx1 = np.divide(mDx1, mDx0)

    mDx2 = np.divide(sp.constants.c, mDx0)

    mDx3 = (sp.constants.c * dt / sp.constants.epsilon_0) * np.divide(sigDx, mDx0)

    mDx4 = (-dt / sp.constants.epsilon_0 ** 2) * np.multiply(sigDy, sigDz)
    mDx4 = np.divide(mDx4, mDx0)


    '''
        then same update coefficients for the Dy field
    '''
    # sigDx/y/z do not change so there's no need to update them

    mDy0 = (1 / dt) + (sigDx + sigDz) / (2 * sp.constants.epsilon_0) + (
            0.25 * dt / sp.constants.epsilon_0 ** 2) * np.multiply(sigDx, sigDz)

    mDy1 = (1 / dt) - (sigDx + sigDz) / (2 * sp.constants.epsilon_0) - (
            0.25 * dt / sp.constants.epsilon_0 ** 2) * np.multiply(sigDx, sigDz)
    mDy1 = np.divide(mDy1, mDy0)

    mDy2 = np.divide(sp.constants.c, mDy0)

    mDy3 = (sp.constants.c * dt / sp.constants.epsilon_0) * np.divide(sigDy, mDy0)

    mDy4 = (-dt / sp.constants.epsilon_0 ** 2) * np.multiply(sigDx, sigDz) # many similar multiplications, compute them once
    mDy4 = np.divide(mDy4, mDy0)



    '''
            then same update coefficients for the Dz field
        '''
    # sigDx/y/z do not change so there's no need to update them

    mDz0 = (1 / dt) + (sigDx + sigDy) / (2 * sp.constants.epsilon_0) + (
            0.25 * dt / sp.constants.epsilon_0 ** 2) * np.multiply(sigDx, sigDy)

    mDz1 = (1 / dt) - (sigDx + sigDy) / (2 * sp.constants.epsilon_0) - (
            0.25 * dt / sp.constants.epsilon_0 ** 2) * np.multiply(sigDx, sigDy)
    mDz1 = np.divide(mDz1, mDz0)

    mDz2 = np.divide(sp.constants.c, mDz0)

    mDz3 = (sp.constants.c * dt / sp.constants.epsilon_0) * np.divide(sigDz, mDz0)

    mDz4 = (-dt / sp.constants.epsilon_0 ** 2) * np.multiply(sigDx, sigDy)  # many similar multiplications, compute them once
    mDz4 = np.divide(mDz4, mDz0)


    '''
    Now the update coefficients for the E fields
    '''
    mEx1 = np.divide(1.0, eps_xx)
    mEy1 = np.divide(1.0, eps_yy)
    mEz1 = np.divide(1.0, eps_zz)



    # create the arrays for the integral terms H-update equations
    ICEx = np.zeros((Nx, Ny, Nz))
    ICEy = np.zeros((Nx, Ny, Nz))
    ICEz = np.zeros((Nx, Ny, Nz))

    IHx = np.zeros((Nx, Ny, Nz))
    IHy = np.zeros((Nx, Ny, Nz))
    IHz = np.zeros((Nx, Ny, Nz))


    # E-update equations
    ICHx = np.zeros((Nx, Ny, Nz))
    ICHy = np.zeros((Nx, Ny, Nz))
    ICHz = np.zeros((Nx, Ny, Nz))

    IDx = np.zeros((Nx, Ny, Nz))
    IDy = np.zeros((Nx, Ny, Nz))
    IDz = np.zeros((Nx, Ny, Nz))





    # @jit
    def time_step_pec(n): # TODO: cross-check with lecture 15 slide 12

        ''' Coding sequence
            1. Compute curl of E (BC are embedded here) x
            2. Update H integrations
            3. Update H field
            4. Compute curl of H
            5. Update D integrations
            6. Update D
            7. Inject source
            8. Update E
        '''


        # 1. Compute curl of E (PEC are already embedded here)
        global Ex
        global Ey
        global Ez
        CEx[:, 0:Ny-1, 0:Nz-1] = (Ez[:, 1:Ny, 0:Nz-1] - Ez[:, 0:Ny-1, 0:Nz-1]) / dy - (Ey[:, 0:Ny-1, 1:Nz] - Ey[:, 0:Ny-1, 0:Nz-1]) / dz
        CEy[0:Nx-1, :, 0:Nz-1] = (Ex[0:Nx-1, :, 1:Nz] - Ex[0:Nx-1, :, 0:Nz-1]) / dz - (Ez[1:Nx, :, 0:Nz-1] - Ez[0:Nx-1, :, 0:Nz-1]) / dx
        CEz[0:Nx-1, 0:Ny-1, :] = (Ey[1:Nx, 0:Ny-1, :] - Ey[0:Nx-1, 0:Ny-1, :]) / dx - (Ex[0:Nx-1, 1:Ny, :] - Ex[0:Nx-1, 0:Ny-1, :]) / dy



        # 2. Update H integration and curl E integration
        global ICEx
        global ICEy
        global ICEz
        global IHx
        global IHy
        global IHz
        global Hx
        global Hy
        global Hz
            # update curl integration
        ICEx += CEx
        ICEy += CEy
        ICEz += CEz
            # update H field integration
        IHx += Hx
        IHy += Hy
        IHz += Hz


        # 3. Update H field
        Hx = np.multiply(mHx1, Hx) + np.multiply(mHx2, CEx) + np.multiply(mHx3, ICEx) + np.multiply(mHx4, IHx)
        Hy = np.multiply(mHy1, Hy) + np.multiply(mHy2, CEy) + np.multiply(mHy3, ICEy) + np.multiply(mHy4, IHy)
        Hz = np.multiply(mHz1, Hz) + np.multiply(mHz2, CEz) + np.multiply(mHz3, ICEz) + np.multiply(mHz4, IHz)


        # 4. Compute curl of H TODO: are there any BC embedded
        CHx[0:Nx, 1:Ny, 1:Nz] = (Hz[0:Nx, 1:Ny, 1:Nz] - Hz[0:Nx, 0:Ny-1, 1:Nz]) / dy \
                                - (Hy[0:Nx, 1:Ny, 1:Nz] - Hy[0:Nx, 1:Ny, 0:Nz-1]) / dz
        CHy[1:Nx, 0:Ny, 1:Nz] = (Hx[1:Nx, 0:Ny, 1:Nz] - Hx[1:Nx, 0:Ny, 0:Nz-1]) / dz \
                                - (Hz[1:Nx, 0:Ny, 1:Nz] - Hz[0:Nx-1, 0:Ny, 1:Nz]) / dx
        CHz[1:Nx, 1:Ny, 0:Nz] = (Hy[1:Nx, 1:Ny, 0:Nz] - Hy[0:Nx-1, 1:Ny, 0:Nz]) / dx \
                                - (Hx[1:Nx, 1:Ny, 0:Nz] - Hx[1:Nx, 0:Ny-1, 0:Nz]) / dy


        # 5. Update D integrations
        global ICHx
        global ICHy
        global ICHz
        global IDx
        global IDy
        global IDz
        global Dx
        global Dy
        global Dz
            # update curl of H integration
        ICHx += CHx
        ICHy += CHy
        ICHz += CHz
            # update D integration
        IDx += Dx
        IDy += Dy
        IDz += Dz

        # 6. Update D field
        Dx[:Nx-1, :Ny-1, :Nz-1] = np.multiply(mDx1[:Nx-1, :Ny-1, :Nz-1], Dx[:Nx-1, :Ny-1, :Nz-1]) \
                                  + np.multiply(mDx2, CHx)[:Nx-1, :Ny-1, :Nz-1] \
                                  + np.multiply(mDx3, ICHx)[:Nx-1, :Ny-1, :Nz-1] \
                                  + np.multiply(mDx4, IDx)[:Nx-1, :Ny-1, :Nz-1]
        Dy[:Nx-1, :Ny-1, :Nz-1] = np.multiply(mDy1[:Nx-1, :Ny-1, :Nz-1], Dy[:Nx-1, :Ny-1, :Nz-1]) \
                                  + np.multiply(mDy2, CHy)[:Nx-1, :Ny-1, :Nz-1] \
                                  + np.multiply(mDy3, ICHy)[:Nx-1, :Ny-1, :Nz-1] \
                                  + np.multiply(mDy4, IDy)[:Nx-1, :Ny-1, :Nz-1]
        Dz[:Nx-1, :Ny-1, :Nz-1] = np.multiply(mDz1[:Nx-1, :Ny-1, :Nz-1], Dz[:Nx-1, :Ny-1, :Nz-1]) \
                                  + np.multiply(mDz2, CHz)[:Nx-1, :Ny-1, :Nz-1] \
                                  + np.multiply(mDz3, ICHz)[:Nx-1, :Ny-1, :Nz-1] \
                                  + np.multiply(mDz4, IDz)[:Nx-1, :Ny-1, :Nz-1]


        # 7. Inject source (z-oriented point current source)
        Dz[x_src, y_src, z_src] += J[n]


        # 8. Update E
        Ex = np.multiply(mEx1, Dx)
        Ey = np.multiply(mEy1, Dy)
        Ez = np.multiply(mEz1, Dz)



        # sample the analytical solution
        # global E_z_exact
        # R = np.zeros((Nx, Ny, Nz))
        # for i in range(Nx):
        #     for j in range(Ny):
        #         for k in range(Nz):
        #             R[i, j, k] = np.linalg.norm(np.asarray((X[i]-x_src, Y[j]-y_src, Z[k]-z_src)))
        # print("r.shape = ", R.shape)
        # E_z_exact[:, :, :] = -(sp.constants.mu_0 / 4 / np.pi) * 5 * dz * omega * np.cos(omega*(n*dt - R[:, :, :]/sp.constants.c)) / R[:, :, :]
        # print(E_z_exact.shape)
        # exit(1)


        return Ex, Ey, Ez, Hx, Hy, Hz # , E_z_exact


    fig = plt.figure(figsize=(8, 8))
    imag = plt.imshow(Ez[:, :, z_src], vmin=-0.03, vmax=0.03)
    # plt.colorbar(np.linalg.norm(Ez[:, :, z_src]))
    # plt.show()


    def anim_init():
        imag.set_data(Ez[:, :, z_src])
        # plt.colorbar(np.linalg.norm(Ez[:, :, z_src]))
        # plt.set_title("time={} sec".format(0))
        return imag


    def animate(n):
        time_step_pec(n)
        imag.set_data(Ez[:, :, z_src])  # update the data
        # plt.set_title("time={:.2f} nsec".format(n*dt*1e+9))
        return imag


    ani = animation.FuncAnimation(fig=fig, func=animate, frames=Nt, fargs=(), init_func=anim_init)


    f = r"./3D_FDTD_UPML_1.mp4"
    writervideo = animation.FFMpegWriter(fps=20)
    ani.save(f, writer=writervideo)
