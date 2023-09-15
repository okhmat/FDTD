import numpy as np
import scipy as sp
from scipy import constants
import matplotlib.pyplot as plt
import matplotlib.animation as animation


'''
Sequence of implementation of the Ez mode

1. Define device parameters
2. Define FDTD parameters
3. Compute grid parameters
4. Build device on grid
5. Compute time step
6. Compute source
7. Compute PML parameters
8. Compute update coefficients
9. Initialize fields
10. Initialize arrays for curls
11. Initialize integration arrays (PML)
    Main loop: 
    1. Compute curl of E (BC are embedded here)
    2. Update H integrations
    3. Update H field
    4. Compute curl of H
    5. Update D integrations
    6. Update Dz
    7. Inject source
    8. Update Ez
    9. Visualize fields (optional)
    



'''

plt.rcParams['text.usetex'] = True


# gaussian source - this is to be sampled in time for all the time steps
# in 2D case it's the amplitude of the Ez out of the xy plane
def g(t, t0, tau):
    return np.exp(-(t - t0)**2 / tau**2)

def src_no_dc(t, t0, tau):
    return ((t) / tau)**3 * np.exp(-(t) / tau) * (4 - (t)/tau)


if __name__=="__main__":

    Nx = 100  # total number of spatial cells in x direction
    Ny = 100
    Nt = 600  # total num of time steps
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

    # mHx = sp.constants.c * dt / mu_xx -- new update coefficients for the PML
    # mHy = sp.constants.c * dt / mu_yy
    # mHz = sp.constants.c * dt / mu_zz

    CEx = np.zeros((Nx, Ny))
    CEy = np.zeros((Nx, Ny))
    CHz = np.zeros((Nx, Ny))

    x_src = int(Nx / 3)
    y_src = int(Ny / 4)


    # # calculate PML parameters; will consider only
    # sigma_max = sp.constants.epsilon_0 / 2 / dt
    # def sigma_pml(coord, length):
    #     return sigma_max * (coord / length)**3
    #
    # # create 2 arrays that ar Nx * Ny for sigma_pml_x and sigma_pml_y
    # sigma_pml_x = np.zeros((Nx, Ny))
    # sigma_pml_y = np.zeros((Nx, Ny))

    '''
    To calculate the PML parameters we will use 2x grid concept.
    This concept originates from the structure of Yee grid
    This is to be used ONLY to simplify the material matrices composition, 
    do not use it in simulation itself
    '''
    Nx2 = 2*Nx
    Ny2 = 2*Ny

    NPML = [10, 10, 16, 16] # size of PML on 1x grid

    sigx = np.zeros((Nx2, Ny2))
    for nx in range(2*NPML[0]): # low x side
        nx1 = 2*NPML[0] - nx - 1
        sigx[nx1, :] = (0.5*sp.constants.epsilon_0/dt) * (((nx+1)/2)/NPML[0])**3
    for nx in range(2*NPML[1]): # high x side
        nx1 = Nx2 - 2*NPML[1] + nx
        sigx[nx1, :] = (0.5*sp.constants.epsilon_0/dt) * (((nx+1)/2)/NPML[1])**3

    sigy = np.zeros((Nx2, Ny2))
    for ny in range(2*NPML[2]): # low y side
        ny1 = 2*NPML[2] - ny - 1
        sigy[:, ny1] = (0.5*sp.constants.epsilon_0/dt) * (((ny+1)/2)/NPML[2])**3
    for ny in range(2*NPML[3]):
        ny1 = Ny2 - 2*NPML[3] + ny
        sigy[:, ny1] = (0.5 * sp.constants.epsilon_0 / dt) * (((ny+1) / 2) / NPML[3]) ** 3

    # print(sigx[:, 0])
    # print(sigy[0, :])


    '''
    # Nx_lo and Nx_hi are the number of cells -- thickness of the PML layer
    Nx_hi = 5 # number of cells in the PML layer
    Nx_lo = 3
    Ny_hi = 10  # number of cells in the PML layer
    Ny_lo = 10

    Lx_hi = dx * Nx_hi
    Lx_lo = dx * Nx_lo
    Ly_hi = dy * Ny_hi
    Ly_lo = dy * Ny_lo

    # fill-in the low boundary
    for nx in range(Nx_lo): # 0, 1, 2, 3, 4
        sigma_pml_x[Nx_lo-nx-1, :] = sigma_pml(nx, Nx_lo) # 4, 3, 2, 1, 0 (nx=4: nx/Nx_lo=4/5)
    for nx in range(Nx_hi):
        sigma_pml_x[Nx-Nx_hi+nx, :] = sigma_pml(nx, Nx_hi) #

    # fill-in the high boundary
    for ny in range(Ny_lo):
        sigma_pml_y[:, Ny_lo-ny-1] = sigma_pml(ny, Ny_lo)
    for ny in range(Ny_hi):
        sigma_pml_y[:, Ny-Ny_hi+ny] = sigma_pml(ny, Ny_hi)
    '''

    # calculate the update coefficients for the update equations
    sigHx = sigx[0:Nx2:2, 1:Ny2:2] # auxiliary array
    sigHy = sigy[0:Nx2:2, 1:Ny2:2] # auxiliary array
    mHx0 = (1/dt) + sigHy/(2*sp.constants.epsilon_0)
    mHx1 = np.divide((1/dt) - sigHy/(2*sp.constants.epsilon_0), mHx0)
    mHx2 = - np.divide(np.divide(sp.constants.c, mu_xx), mHx0)
    mHx3 = - np.divide(np.divide((sp.constants.c*dt/sp.constants.epsilon_0) * sigHx, mu_xx), mHx0)

    sigHx = sigx[1:Nx2:2, 0:Ny2:2] # aux
    sigHy = sigy[1:Nx2:2, 0:Ny2:2] # aux
    mHy0 = (1/dt) + sigHx/(2*sp.constants.epsilon_0)
    mHy1 = np.divide((1/dt) - sigHx/(2*sp.constants.epsilon_0), mHy0)
    mHy2 = - np.divide(np.divide(sp.constants.c, mu_yy), mHy0)
    mHy3 = - np.divide(np.divide((sp.constants.c*dt/sp.constants.epsilon_0) * sigHy, mu_yy), mHy0)

    sigDx = sigx[0:Nx2:2, 0:Ny2:2]
    sigDy = sigy[0:Nx2:2, 0:Ny2:2]
    mDz0 = 1/dt + (0.5/sp.constants.epsilon_0)*(sigDx + sigDy) + (0.25*dt/sp.constants.epsilon_0**2) \
            * np.multiply(sigDx, sigDy)
    mDz1 = 1/dt - (0.5/sp.constants.epsilon_0)*(sigDx + sigDy) - (0.25*dt/sp.constants.epsilon_0**2) \
            * np.multiply(sigDx, sigDy)
    mDz1 = np.divide(mDz1, mDz0)
    mDz2 = np.divide(sp.constants.c, mDz0)
    mDz4 = - (dt/sp.constants.epsilon_0**2) * np.multiply(sigDx, sigDy)
    mDz4 = np.divide(mDz4, mDz0)



    # arrays for the integrals -- Ez mode
    ICEx = np.zeros((Nx, Ny))
    ICEy = np.zeros((Nx, Ny))
    IDz = np.zeros((Nx, Ny))

    # this is the Ez mode
    def time_step_2D_PML(n):

        '''
        1. Compute curl of E (BC are embedded here) x
        2. Update H integrations
        3. Update H field
        4. Compute curl of H
        5. Update D integrations
        6. Update Dz
        7. Inject source
        8. Update Ez
        9. Visualize fields (optional)
        '''


        # 1. Compute curl of E
        global Ez
        CEx[:, :Ny-1] = (Ez[:, 1:Ny] - Ez[:, :Ny-1]) / dy
        CEx[:, Ny-1] = (0 - Ez[:, Ny-1]) / dy # PEC BC
        CEy[:Nx-1, :] = -(Ez[1:Nx, :] - Ez[:Nx-1, :]) / dx
        CEy[Nx-1, :] = -(0 - Ez[Nx-1, :]) / dx # PEC BC


        # 2. Update H integrations
        # for Ez mode we only need Hx and Hy -- thus update the integration terms for them only. E field that those
        # integrals depend on are not changed during the updating of the Hx and Hy, thus we can calculate both integral
        # terms ahead of updating both of the H field components
        global ICEx
        global ICEy
        ICEx += CEx
        ICEy += CEy


        # 3. Update H field
        global Hx
        Hx = np.multiply(mHx1, Hx) + np.multiply(mHx2, CEx) + np.multiply(mHx3, ICEx)
        global Hy
        Hy = np.multiply(mHy1, Hy) + np.multiply(mHy2, CEy) + np.multiply(mHy3, ICEy)


        # 4. Compute curl of H
        # CHz[1:Nx-1, 1:Ny-1] = (Hy[1:Nx-1, 1:Ny-1] - Hy[0:Nx, 1:Ny-1]) / dx - (Hx[1:Nx-1, 1:Ny-1] - Hx[1:Nx-1, 0:Ny]) / dy
        for nx in range(1, Nx-1):
            for ny in range(1, Ny-1):
                CHz[nx, ny] = (Hy[nx, ny] - Hy[nx - 1, ny]) / dx - (Hx[nx, ny] - Hx[nx, ny - 1]) / dy
        # # points at the lower boundary
        # for ny in range(1, Ny):
        #     CHz[0, ny] = (Hy[0, ny] - 0) / dx - (Hx[0, ny] - Hx[0, ny-1]) / dy
        # # points at the left boundary
        # for nx in range(1, Nx):
        #     CHz[nx, 0] = (Hy[nx, 0] - Hy[nx-1, 0]) / dx - (Hx[nx, 0] - 0) / dy
        # # point at the left bottom corner
        # CHz[0, 0] = (Hy[0, 0] - 0) / dx - (Hx[0, 0] - 0) / dy


        # 5. Update D integrations
        global Dz
        global IDz
        IDz += Dz # it is normalized quantity, with tilde


        # 6. Update Dz
        Dz = np.multiply(mDz1, Dz) + np.multiply(mDz2, CHz) + np.multiply(mDz4, IDz)


        # 7. Inject source
        Dz[x_src, y_src] += Jz[n]


        # Update Ez
        Ez = np.divide(Dz, eps_zz) # again, this is normalized quantity, with tilde


        return Hx, Hy, Ez



    # animation

    fig = plt.figure(figsize=(8, 8))
    imag = plt.imshow(Ez, vmin=-0.03, vmax=0.03)
    plt.show()


    def anim_init():
        imag.set_data(Ez)
        # plt.set_title("time={} sec".format(0))
        return imag


    def animate(n):
        Hx, Hy, Ez = time_step_2D_PML(n)
        imag.set_data(Ez)  # update the data
        # plt.set_title("time={:.2f} nsec".format(n*dt*1e+9))
        return imag


    ani = animation.FuncAnimation(fig=fig, func=animate, frames=Nt, fargs=(), init_func=anim_init)


    f = r"./2D_PML.mp4"
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








