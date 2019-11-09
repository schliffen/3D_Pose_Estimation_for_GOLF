#
#  ---- This script is for plotting statistics that is saved during run ----
#
import numpy as np
import scipy as sc
import pickle as pk
from numpy.polynomial import legendre as leg
import matplotlib.pyplot as plt
# import naginterfaces

# step one smoothing the points
sourcedir = '/home/ali/CLionProjects/PoseEstimation/full_flow/results/'

# ----------------------------------------this part is just for test -----------------------

#
#
#
# np.random.seed(4484)
# nsamples=1000
# noutliers=50
# xs=np.linspace(-1,1,nsamples)
# ys=xs*np.pi
# f=np.cos(ys) + np.sin(ys)*np.sin(ys)*np.sin(ys)+np.cos(ys)*np.cos(ys)*np.cos(ys*ys)
# errs=np.random.normal(0.0,0.4,(nsamples))
#
#
# if noutliers>0:
#     outliers=np.floor(np.random.rand(noutliers)*nsamples).astype(np.int)
#     errs[outliers]=errs[outliers]*20.0
# deg=20
# V=leg.legvander(xs,deg)
# coeffs=np.linalg.lstsq(V,f+errs,rcond=None)[0]
# g=leg.legval(xs,coeffs)



#Find least-1-norm solution to Ax=b using linear programming

# we can use scipy.linalg.solve
# from naginterfaces.library.opt import lp_solve
# from naginterfaces.library.opt import nlp1_init
# from naginterfaces.library.opt import lp_option_string


class smoothing_prediction():
    def __init__(self, finit, fend):
        self.deg = 25
        self.finit = finit
        self.fend = fend

    def approx_polynomial(self, data, xs):
        V = leg.legvander(xs, self.deg)
        # Do the fit -> this part should be implemented #todo: complete the implementation of this part
        coeffs = np.linalg.lstsq(V, data, rcond=None)[0]
        #Evaluate the fit for plotting purposes
        g = leg.legval(xs, coeffs)
        return g
    def optimize_pose(self, poses):
        #
        # reading x y z separately
        #
        t_axis = sc.linspace(self.finit, self.fend, self.fend-self.finit + 1)
        #Number of samples of our data d
        nsamples = len(t_axis)-1
        #The independent variable sampled at regular intervals
        xs = np.linspace(-1, 1, nsamples)
        #The Legendre Vandermonde matrix
        # pose_coordinates
        pose_coords = []
        for j in range(3):
            xyz_axis = [poses[i][j] for i in range(nsamples)]
            pose_coords.append( self.approx_polynomial(xyz_axis, xs) )
        return pose_coords










    def lst1norm_nag(self, A,b):
        (m,n)=A.shape
        nvars=m+n
        ncons=2*m

        tcons=[k for k in range(m+n,2*m+n)]
        cons=np.zeros((ncons,nvars))
        cons[0:m,0:m]=-np.identity(m)
        cons[0:m,m:m+n]=A
        cons[m:2*m,0:m]=-np.identity(m)
        cons[m:2*m,m:m+n]=-A

        c=np.zeros(nvars)
        c[0:m]=1.0

        ub=np.zeros(nvars+ncons)
        lb=np.zeros(nvars+ncons)

        for i in range(0,m):
            lb[i]=0.0
            ub[i]=1.00E+20
        for i in range(m,m+n):
            lb[i]=-1.00E+20
            ub[i]=1.00E+20
        for i in range(m+n,m+n+m):
            ub[i]=b[i-m-n]
            lb[i]=-1.00E+20
        for i in range(m+n+m,m+n+m+m):
            ub[i]=-b[i-m-n-m]
            lb[i]=-1.00E+20

        istate = np.zeros(nvars+ncons).astype(np.int)

        x0=np.zeros(nvars)
        # comm = nlp1_init('lp_solve')
        # lp_option_string("Cold Start",comm)
        # istate,x,itera,obj,ax,clamda)=lp_solve(ncons,cons,lb,ub,c,istate,x0,comm)
        return x0[m:2*m]





#
# testing post process for implementing over the program
#

if __name__ == '__main__':
    #
    # construction of coeff matrix A and rhs b
    #
    # loading DATA
    #
    with open(sourcedir + 'golf_04.mp4_best_pos.pickle', 'rb') as f:
        headps = np.array (pk.load( f ))
    # with open(sourcedir + 'hands_pos.pickle', 'rb') as f:
    #     handsps = np.array(pk.load( f ))
    # frame count
    t_axis = np.linspace(0, headps.shape[-1], headps.shape[-1] + 1)

    x_axis_lh = headps[0, 13, 0, :]
    y_axis_lh = headps[0, 13, 1, :]
    z_axis_lh = headps[0, 13, 2, :]
    # x_axis_hd = [headps[:,0][i][0] for i in range(t_axis.shape[0])]
    # y_axis_hd = [headps[:,0][i][1] for i in range(t_axis.shape[0])]
    # z_axis_hd = [headps[:,0][i][2] for i in range(t_axis.shape[0])]

    # plotting before fitting
    plt.figure(0)
    plt.plot(t_axis[1:], z_axis_lh, 'b o')


    # plotting after fitting
    # Use 20th order polynomial fit
    deg = 4
    #Number of samples of our data d
    nsamples = len(t_axis)-1
    #The independent variable sampled at regular intervals
    xs=np.linspace(-1, 1, nsamples)
    #The Legendre Vandermonde matrix
    V=leg.legvander(xs, deg)
    #Generate some data to fit
    ys=xs*np.pi
    # f=np.cos(ys) + np.sin(ys)*np.sin(ys)*np.sin(ys)+np.cos(ys)*np.cos(ys)*np.cos(ys*ys)
    #Do the fit
    coeffs=np.linalg.lstsq(V, z_axis_lh, rcond=None)[0]
    #Evaluate the fit for plotting purposes
    g=leg.legval(xs, coeffs)
    # computing the derivative
    g_prm = (g[1:] - g[0:-1])/.1
    # rescaling
    plt.figure(0)
    plt.plot(t_axis[1:], g, 'r--')
    plt.plot(t_axis[1:-1], g_prm, ' s')
    plt.title('head x position')
    # this is soft motion of head (x axis)
    # plotting hands position
    x_axis_hn = [handsps[:,0][i][0] for i in range(t_axis.shape[0])]
    y_axis_hn = [handsps[:,0][i][1] for i in range(t_axis.shape[0])]
    z_axis_hn = [handsps[:,0][i][2] for i in range(t_axis.shape[0])]

    # plotting before fitting
    plt.figure(1)
    plt.plot(t_axis, z_axis_hn, 'b o')


    # plotting after fitting
    # Use 20th order polynomial fit
    deg = 10
    #Number of samples of our data d
    nsamples = len(t_axis)
    #The independent variable sampled at regular intervals
    xs=np.linspace(-1, 1, nsamples)
    #The Legendre Vandermonde matrix
    V=leg.legvander(xs, deg)
    #Generate some data to fit
    ys=xs*np.pi
    # f=np.cos(ys) + np.sin(ys)*np.sin(ys)*np.sin(ys)+np.cos(ys)*np.cos(ys)*np.cos(ys*ys)
    #Do the fit
    coeffs=np.linalg.lstsq(V, z_axis_hn, rcond=None)[0]
    #Evaluate the fit for plotting purposes
    g=leg.legval(xs, coeffs)
    # rescaling
    plt.figure(1)
    plt.plot(t_axis, g, 'r--')
    plt.title('hand y position')
    # this is soft motion of head (x axis)
    print('plotting finished')

    # adding the smothed form to the visualization


