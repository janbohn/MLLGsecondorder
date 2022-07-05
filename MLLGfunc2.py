from __future__ import division
from dolfin import *
import numpy as np
import bempp.api
from scipy.special import comb 
from scipy.sparse.linalg.interface import aslinearoperator
from scipy.sparse.linalg import gmres
from bempp.api.external.fenics import FenicsOperator
from maxwellrt0 import *

from bempp.api.operators.boundary import maxwell
import timeit
#import sys

#sys.setrecursionlimit(100000)

if not has_linear_algebra_backend("PETSc"):
    print("DOLFIN has not been configured with PETSc. Exiting.")
    exit()
if not has_slepc():
    print("DOLFIN has not been configured with SLEPc. Exiting.")
    exit()

def MLLGfunc(T, N, tau, h, m, rho, L, tolgmres, eps, mu, sig, alpha, Ce,k,meshh):
    theta=0.5
    alstab=1
    r=1
    print('h is ', h, ' N is', N,' T is', T,' tAU is', tau,' p is', k)

    #precomputation for the BDF scheme
    gamma = []
    delta = []

    tmp=0
    #print(k)
    for i in range(1,k+1):
        tmp+= 1.0/i;
    delta.append(tmp)

    for i in range(1,k+1):
        gamma.append(comb(k,i)*(-1)**(i-1))
        
        tmp=0
        for j in range(i,k+1):
            tmp+=comb(j,i)*(-1)**float(i)/float(j)
        delta.append(tmp)
    
    print(delta)
    print(gamma)

    start = timeit.default_timer()
    #meshh = UnitCubeMesh(h, h, h)

    # approximaion spaces 
    # m   = Lagrange1
    # E,H = N1curl
    # gamma_TE, gamma_TH = trace_space = RT
    # phi ~ gamma_TH = BC
    # psi ~ -gamma_TE = RWG

    Pr3 = VectorElement('Lagrange', meshh.ufl_cell(), 1, dim=3);
    V3 = FunctionSpace(meshh, Pr3)
    Pr = FiniteElement('P', meshh.ufl_cell(), 1);
    V = FunctionSpace(meshh, Pr)
    element = MixedElement([Pr3, Pr]);
    VV = FunctionSpace(meshh, element)

    Xr = FiniteElement("N1curl", meshh.ufl_cell(), 1)
    X = FunctionSpace(meshh, Xr)
    fenics_space = X

    XXr = MixedElement([Xr, Xr]);
    XX = FunctionSpace(meshh, XXr)

    trace_space, trace_matrix = nc1_tangential_trace(fenics_space)  # trace space and restriction matrix trace space ist RWG (zuvor RT)

    bc_space = bempp.api.function_space(trace_space.grid, "BC", 0)  # domain spaces
    rwg_space = bempp.api.function_space(trace_space.grid, "RWG", 0)
    snc_space = bempp.api.function_space(trace_space.grid, "SNC", 0)  # dual to range spaces
    rbc_space = bempp.api.function_space(trace_space.grid, "RBC", 0)
    #brt_space = bempp.api.function_space(trace_space.grid, "RWG", 0) #attention, wurde zu RWG gemacht..

    nBEM = trace_space.global_dof_count  # DOFs
    nFEM = fenics_space.dim()
    nMAG = V3.dim()
    nLAM = V.dim()

    def dlt(z):
        # BDF1
        #print(k)
        #return 
        if k==1:
            z=1.0 - z
        # BDF2
        if k==2: 
            z=1.0-z+0.5*(1.0-z)**2
        return z

    # Initial Data and Input functions

    Ej = project(Expression(['0.00', '0.0', '0.0'], degree=1), X)
    Hj = project(Expression(['0.0', '0.0', '0.0'], degree=1), X)
    mag = interpolate(Expression(['1.00', '0.0', '0.0'], degree=1), V3)
    J=project(Expression(['100.00','000.0','00.0'],degree=1),X)
    mj = mag
    mj = mj / sqrt(dot(mj, mj))
    mj = project(mj, V3, solver_type='cg')


    mjko = np.zeros(nMAG)  # Coefficient vector magnetization
    vjko = np.zeros(nMAG)  # Coefficient vector time derivative magnetization
    vlamjko = np.zeros(nMAG + nLAM)  # Coefficient vector time derivative magnetization and lagrange function
    mjko[:] = mj.vector()[:]

    # Coefficient vector [E,H,phi,psi]
    phiko = np.zeros(nBEM)
    psiko = np.zeros(nBEM)

    # EH =  Function(XX)
    Ulitt = TrialFunction(fenics_space)
    phi = TestFunction(V3)
    # interior operators and mass matrices
    # LLG Part

    rhsLLGH = FenicsOperator((+inner(phi, Ulitt)) * dx)

    # Maxwell part
    Uli = TrialFunction(fenics_space)
    UliT = TestFunction(fenics_space)
    M0 = FenicsOperator((inner(Uli, UliT)) * dx)
    Mwv = assmatrix(M0.weak_form()) 
    from scipy.linalg import inv
    Minv= inv(Mwv)
    D = FenicsOperator(0.5 * inner(Uli, curl(UliT)) * dx + 0.5 * inner(curl(Uli), UliT) * dx)
    D=assmatrix(D.weak_form())
 
    trace_op = aslinearoperator(trace_matrix)  # trace operator
    # exterior mass matrices
    massbdfo = bempp.api.operators.boundary.sparse.identity(rwg_space, rwg_space, rbc_space)  # das hier falls oben all_BC bzw für fakeop
    massbd = bempp.api.operators.boundary.sparse.identity(rwg_space, bc_space, snc_space)
    #dd=assmatrix(massbd.weak_form())
    C0T= 0.5*np.matmul(assmatrix(massbd.weak_form()),assmatrix(trace_op))
    C1T=1/mu*C0T
    C0= np.transpose(C0T)
    C1=1/mu*C0
    # calderon operator 
    calderon2 = maxwell.multitrace_operator(trace_space.grid, 1j * sqrt(mu * eps) * dlt(
        0) / tau, space_type="all_rwg")  # bc, precision='single', assembler="dense_evaluator"
    # maxwell.multitrace_operator(trace_space.grid, 1j * sqrt(mu * eps) * dlt(
    #    0) / tau)
    cald =calderon2.weak_form()# fakeop (trace_space.grid, 1j * sqrt(mu * eps) * dlt(0) / tau ,eps,mu,massbdfo,nBEM) #
    caldfac = 1.0 / mu #* (dlt(0)) ** (-m)

    # Definition coupled 4x4 matrix 
    from bempp.api.assembly.blocked_operator import BlockedDiscreteOperator
    blocke1 = np.ndarray([2, 2], dtype=np.object) 

    blocke1[0, 0] = 1.0 / mu * np.sqrt(mu / eps) * caldfac*assmatrix(cald[0, 1])  +tau/2/eps*np.matmul(np.matmul(C0T,Minv),C0)   # Calderon* \partial_t^m phiko =

    blocke1[2-2, 3-2] = -caldfac*cald[0, 0]

    blocke1[3-2, 2-2] = caldfac*cald[1, 1]

    blocke1[3-2, 3-2] = -mu * np.sqrt(eps / mu) * caldfac*assmatrix(cald[1, 0]) + tau*2*alstab/mu*np.matmul(np.matmul(C1T,Minv),C1)

    Lhhs =  assmatrix(BlockedDiscreteOperator(np.array(blocke1))) #bempp.api.as_matrix()

    stop = timeit.default_timer()

    start2 = timeit.default_timer()

    # Definition of Convolution Quadrature weights 
    storblock = np.ndarray([2, 2], dtype=np.object);  # dummy variable
    wei = np.ndarray([int(L)], dtype=np.object);  # dummy array of B(zeta_l)(zeta_l)**(-m)
    CQweights = np.ndarray([int(N + 1)], dtype=np.object);  # array of the weights CQweights[n]~B_n


    for ell in range(0,int(L)): # CF Lubich 1993 On the multistep time discretization of linearinitial-boundary value problemsand their boundary integral equations, Formula (3.10)
        if int(ell/10)==ell/10.0:
            print(ell)
        calderon=maxwell.multitrace_operator(trace_space.grid,1j*sqrt(mu*eps)* dlt( rho*np.exp(2.0*np.pi*1j*ell/L))/tau,space_type="all_rwg")
        cald = calderon.weak_form()#fakeop (trace_space.grid, 1j * sqrt(mu * eps) * dlt(rho * np.exp(2.0 * np.pi * 1j * ell / L)) / tau,eps,mu,massbdfo,nBEM)#(dlt(rho*np.exp(2.0*np.pi*1j*ell/L))/tau)**(-m)*1.0/mu* wurde unten eingebaut
        storblock[0,0]=1.0/mu* np.sqrt(mu/eps)* cald[0,1]
        storblock[0,1]=-cald[0,0]
        storblock[1,0]=cald[1,1]
        storblock[1,1]=-mu* np.sqrt(eps/mu)*cald[1,0]
        wei[ell]=  assmatrix( BlockedDiscreteOperator(np.array(storblock)))# bempp.api.as_matrix()
        wei[ell]= (dlt(rho*np.exp(2.0*np.pi*1j*ell/L))/tau)**(-m)*1.0/mu*wei[ell]
    stop2 = timeit.default_timer()
    print('Time for initial data and LHS: ', stop - start, ' Time for Calderon evaluation: ', stop2 - start2)
    for n in range(0, int(N + 1)):
        CQweights[n] = wei[0]  # Fourier Transform
        for ell in range(1, int(L)):
            CQweights[n] = CQweights[n] + np.exp(-2.0 * np.pi * 1j * n * ell / L) * wei[ell]

        CQweights[n] = rho ** (-n) / L * CQweights[n]

    randkoeff = 1j * np.zeros([int(N + 1), 2 * nBEM])  # storage variable for boundary coefficients es ist rk[j]=phi(t_j-0.5) bzw rk[j+1]=phi(t_j+1/2)
    dtmpsiko = 1j * np.zeros([int(N + 1), 2 * nBEM])
    
    mkoeff = np.zeros([int(N + 1), nMAG])
    Ekoeff = np.zeros([int(N + 1), nFEM])
    Hkoeff = np.zeros([int(N + 1), nFEM])
    psikoeff= 1j * np.zeros([int(N + 1),  nBEM])
    mvec=np.zeros([int(N + 1), nMAG])
    
    Hjko=Hj.vector()[:]
    Ejko=Ej.vector()[:]
    Hrko=np.zeros([int(N + 1)])
    psiko=np.zeros(nBEM)
    Ekoeff[0, :] = np.real(Ejko)
    Hkoeff[0, :] = np.real(Hjko)
    mvec[0,:]=mjko
    psikoeff[0,:]=psiko
    randkoeff[0, :] = np.zeros([1,2*nBEM])
    vjko[:]=np.zeros([1,nMAG])
    solu=np.zeros(2*nBEM)
    stop = timeit.default_timer()
    
    
    ## initial value magnetization
    
    if k>1:
        [mvec_tmp ,Ekoeff_tmp, Hkoeff_tmp, phi_tmp, psi_tmp] = MLLGfunc((k-1)*tau, 1.0/N*int(T*tau**(float(-1)/(k-1))), 1/T*N*tau**(float(k)/(k-1)), h, m, rho, L/N, tolgmres, eps, mu, sig, alpha, Ce,k-1,meshh)
        #bdfllg_func(r,k-1,N,alpha,(k-1)*tau,tau**(float(k)/(k-1)),minit,mesh,VV,V3,V);
        #mvec_tmp=bdfllg_func(r,k-1,N,alpha,tau,tau,minit,mesh,VV,V3,V);
        #print(mvec_tmp)
    for i in range(1,k):
        mvec[i,:]=mvec_tmp[int(T*i*tau**(float(-1)/(k-1))/N),:]
        Hjko[:]=Hkoeff_tmp[int(T*i*tau**(float(-1)/(k-1))/N),:]
        Ejko[:]=Ekoeff_tmp[int(T*i*tau**(float(-1)/(k-1))/N),:]
        psiko[:]=psi_tmp[int(T*i*tau**(float(-1)/(k-1))/N),:]
        solu[:]= np.concatenate([phi_tmp[int(T*i*tau**(float(-1)/(k-1))/N),:], psiko[:]/2])
        Ekoeff[i, :] = np.real(Ejko)
        Hkoeff[i, :] = np.real(Hjko)
        psikoeff[i,:]=psiko[:]
        randkoeff[i, :] = solu[:]
        vjko[:]=(mvec[1,:]-mvec[0,:])/tau;
    #vjko[:]=(mvec_tmp[int(i*tau**(float(-1)/(k-1))/N),:]-mvec_tmp[int(i*tau**(float(-1)/(k-1))/N)-1,:])/(N*tau**(float(k)/(k-1)))
    for j in range(k-1, int(N)):  # time stepping: update from timestep t_j to j+1
 
        #### Maxwell update
        
        Hj12ko= Hjko+tau/2/mu*Minv.dot(D.dot(Ejko))-tau/2.0*C1.dot(psiko)- tau*0.5 * Minv.dot(rhsLLGH.weak_form().transpose().dot(  vjko))
        Erko=Ejko+tau/eps*Minv.dot(D.dot(Hj12ko))+ ((j+1.0/2.0)*tau)**4* tau/eps* J.vector()[:]   
        
        #### Boundary solve
        
        dtmpsiko[:, :] = randkoeff[:, :]  # compute \partial_t^m phi,  \partial_t^m psi,
        for ss in range(0, m):
            for r in range(0, j + 1):
                dtmpsiko[j + 1 - r, :] = (dtmpsiko[j + 1 - r, :] - dtmpsiko[j + 1 - r - 1, :]) / tau
        boundary_rhs = 1j * np.zeros(2 * nBEM)

        for kk in range(1, j + 2):  # Convolution, start bei 1, da psiko(0)=0
            dd = CQweights[j + 1 - kk].dot(dtmpsiko[kk, :])
            boundary_rhs[:] += dd[:]


        # Right hand side
        Rhhs = - 1.0 * boundary_rhs+ np.concatenate([C0T.dot(Ejko/2+Erko/2),C1T.dot(Hj12ko)+2*alstab*tau/mu*np.matmul(np.matmul(C1T,Minv),C1).dot(psiko)])

        # Solution of Lhs=Rhs with gmres
        nrm_rhs = np.linalg.norm(Rhhs)  # norm of r.h.s. vector
        it_count = 0

        def count_iterations(x):
            nonlocal it_count
            it_count += 1
        solu, info = gmres(Lhhs, Rhhs, tol=tolgmres, callback=count_iterations, x0=solu)
        if (info > 0):
            print("Failed to converge after " + str(info) + " iterations")
        else:
            print("Solved system " + str(j) + " in " + str(it_count) + " iterations. Size of RHS " + str(nrm_rhs))
        phi12ko = solu[:nBEM]
        psi12ko = solu[nBEM:]
        psiko=2*psi12ko-psiko
            
        #### Maxwell update    
        Ejko =  Erko-tau/eps*Minv.dot(C0.dot(phi12ko))
        Hrko=Hj12ko+tau/2/mu*Minv.dot(D.dot(Ejko))-tau/2/mu*C1.dot(psiko)

        
        #### LLG update ########
        
        mhat = interpolate(Expression(['0','0','0'],degree=r),V3)
        mr = interpolate(Expression(['0','0','0'],degree=r),V3)
        for i in range(0,k):
            mhat.vector()[:]=mhat.vector()[:] + float(gamma[i])*mvec[j+1-i-1,:]
            mr.vector()[:]=mr.vector()[:] - float(delta[i+1])*mvec[j+1-(i+1),:]
            
        mr.vector()[:]=mr.vector()[:]/float(delta[0])
        #mr=project(mr,V3)

        mhat=mhat/sqrt(dot(mhat,mhat))
        #for i in range(0,k):
        #    mhat=mhat + float(gamma[i])*mvec[j-i-1]
        #    mr=mr - float(delta[i+1])*mvec[j-(i+1)]
        #    
        #mr=mr/float(delta[0])
        #mr=project(mr,V3)
        #mhat=mhat/sqrt(dot(mhat,mhat))
        # define variational problem
        
        (v,lam) = TrialFunctions(VV)
        (phi,muu) = TestFunctions(VV)
       
        # build external field
        Hh = interpolate(Expression(['0','0','0'],degree=r),X)
        Hh.vector()[:]=Hrko
        # reducing quadrature degree
        #dx = dx(metadata={'quadrature_degree': 5})
        
        # define LLG form
        dxr= dx(metadata={'quadrature_degree': 5})
        lhs = ((alpha*inner(v,phi)+inner(cross(mhat,v),phi)+tau/float(delta[0])*inner(nabla_grad(v),nabla_grad(phi)))*dxr
            + inner( dot(phi,mhat),lam)*dxr   + inner(dot(v,mhat),muu)*dxr+ tau* 0.5 *inner(v,phi)*dxr)

        rhs = (-inner(nabla_grad(mr),nabla_grad(phi))+inner(Hh, phi))*dxr 
        
        # compute solution
        vlam = Function(VV)
        solve(lhs == rhs, vlam)#,solver_parameters={"linear_solver": "gmres"},form_compiler_parameters={"optimize": True})
        
        # update magnetization
        (v,lam) = vlam.split(deepcopy=True)
        vjko[:] = v.vector()[:]
        
        mvec[j+1,:]= mr.vector()[:] + tau/float(delta[0]) * v.vector()[:];
        
        Hjko =  Hrko -tau* 0.5 * Minv.dot(rhsLLGH.weak_form().transpose().dot(  vjko))
        
        
        
        Ekoeff[j+1, :] = np.real(Ejko)
        Hkoeff[j+1, :] = np.real(Hjko)
        psikoeff[j+1,:]=psiko
        randkoeff[j+1, :] = solu[:]
        
    return (mvec, Ekoeff, Hkoeff, randkoeff[:, :nBEM], psikoeff[:, :]);


def fakeop(gridd,k,eps,mu,massbd,nBEM):
    from bempp.api.assembly.blocked_operator import BlockedDiscreteOperator
    s= -1j*k/np.sqrt(eps*mu)
    #rwg_space = bempp.api.function_space(gridd, "RWG", 0)
    #snc_space = bempp.api.function_space(gridd, "SNC", 0)
    #rbc_space = bempp.api.function_space( gridd, "RBC", 0)
    #nBEM=rbc_space.global_dof_count
    storblock = np.ndarray([2, 2], dtype=np.object);
    #massbd = bempp.api.operators.boundary.sparse.identity(rwg_space, rwg_space, rbc_space)
    factor=s +s**2*np.exp(-s)+1/s+s**(0.5)+ 1#np.cos(10*s)
    storblock[0, 0] =  factor*massbd.weak_form()
    storblock[0, 1] =  factor*np.zeros((nBEM,nBEM))
    storblock[1, 0] =  factor*np.zeros((nBEM,nBEM))
    storblock[1, 1] =  factor*massbd.weak_form()
    
    est=BlockedDiscreteOperator(np.array(storblock)) 
    return est
    
    
def assmatrix(operator):
    from numpy import eye
    cols = operator.shape[1]
    return operator @ eye(cols)
    
    