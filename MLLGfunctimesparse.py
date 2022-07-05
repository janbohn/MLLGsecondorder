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
from scipy.linalg import inv
import timeit

if not has_linear_algebra_backend("PETSc"):
    print("DOLFIN has not been configured with PETSc. Exiting.")
    exit()
if not has_slepc():
    print("DOLFIN has not been configured with SLEPc. Exiting.")
    exit()

def MLLGfunc(T,rm, N, tau, h,pt,ps, m, rho,alstab, L, tolgmres,matpar,V3,VV,fenics_space,trace_space,trace_matrix):
    print('h is ', h, ' N is', N,' T is', T,' tau is', tau,' pt is', pt,'ich bin sparse')
    [eps, mu, sig, alpha, Ce]=matpar
    #### precomputation for the BDF scheme
    gamma = []
    delta = []

    tmp=0
    for i in range(1,pt+1):
        tmp+= 1.0/i;
    delta.append(tmp)

    for i in range(1,pt+1):
        gamma.append(comb(pt,i)*(-1)**(i-1))
        
        tmp=0
        for j in range(i,pt+1):
            tmp+=comb(j,i)*(-1)**float(i)/float(j)
        delta.append(tmp)
    
    def dlt(z):
        # BDF1
        if pt==1:
            z=1.0 - z
        # BDF2
        if pt==2: 
            z=1.0-z+0.5*(1.0-z)**2
        return z


    start = timeit.default_timer()

    #### approximaion spaces 
    # m   = Lagrange1
    # E,H = N1curl
    # gamma_TE, gamma_TH = trace_space = RWG
    # phi ~ mu gamma_TH = RWG
    # psi ~ -gamma_TE = RWG

    bc_space = bempp.api.function_space(trace_space.grid, "BC", 0)  # domain space
    snc_space = bempp.api.function_space(trace_space.grid, "SNC", 0)  # dual to range spaces

    nBEM = trace_space.global_dof_count  # DOFs
    nFEM = fenics_space.dim()
    nMAG = V3.dim()
    
    #### interior operators and mass matrices
    # LLG Part
    Ulitt = TrialFunction(fenics_space)
    phi = TestFunction(V3)
    rhsLLGH = FenicsOperator((+inner(phi, Ulitt)) * dx)

    # Maxwell part
    Uli = TrialFunction(fenics_space)
    UliT = TestFunction(fenics_space)
    M0 = FenicsOperator((inner(Uli, UliT)) * dx)
    Mwv = M0.weak_form()
    D = FenicsOperator(-0.5 * inner(Uli, curl(UliT)) * dx - 0.5 * inner(curl(Uli), UliT) * dx)
    D= D.weak_form()
 
    trace_op = aslinearoperator(trace_matrix)  # trace operator
    
    # exterior part
    #massbdfo = bempp.api.operators.boundary.sparse.identity(rwg_space, rwg_space, rbc_space)  # das hier falls oben all_BC bzw für fakeop
    massbd = bempp.api.operators.boundary.sparse.identity(trace_space, bc_space, snc_space).weak_form()
    C1T= -0.5* massbd *trace_op # unsicher wegen minus!!!  
    C0T=1/mu*C1T
    C1= -0.5*trace_op.adjoint() * massbd.transpose()
    C0=1/mu*C1
    # calderon operator 
    calderon2 = maxwell.multitrace_operator(trace_space.grid, 1j * sqrt(mu * eps) * dlt(
        0) / tau, space_type="all_rwg")  # bc, precision='single', assembler="dense_evaluator"
    cald =calderon2.weak_form()# fakeop (trace_space.grid, 1j * sqrt(mu * eps) * dlt(0) / tau ,eps,mu,massbdfo,nBEM) #
    caldfac = 1.0 / mu #* (dlt(0)) ** (-m)
    # Definition of 2x2 matrix 
    from bempp.api.assembly.blocked_operator import BlockedDiscreteOperator
    blocke1 = np.ndarray([4, 4], dtype=np.object)
    
    blocke1[0, 0] = eps * Mwv
    blocke1[1, 0] = np.zeros((nFEM,nFEM))   
    blocke1[0, 1] = np.zeros((nFEM,nFEM)) 
    blocke1[1, 1] = mu * Mwv

    
    blocke1[0, 2] = tau/2*C0
    blocke1[1, 2] = np.zeros((nFEM, nBEM))
    blocke1[0, 3] = np.zeros((nFEM, nBEM))
    blocke1[1, 3] = 2*alstab*tau*C1

    blocke1[2, 0] = - C0T
    blocke1[3, 0] = np.zeros((nBEM, nFEM))
    blocke1[2, 1] = np.zeros((nBEM, nFEM))
    blocke1[3, 1] = -C1T
    
    
    blocke1[2, 2] = 1.0 / mu * np.sqrt(mu / eps) * caldfac*cald[0, 1]  #+tau/2/eps*np.matmul(np.matmul(C0T,Minv),C0)   
    blocke1[2, 3] = -caldfac*cald[0, 0]
    blocke1[3, 2] = caldfac*cald[1, 1]
    blocke1[3, 3] = -mu * np.sqrt(eps / mu) * caldfac*cald[1, 0] #+ tau*2*alstab/mu*np.matmul(np.matmul(C1T,Minv),C1)
    
    Lhhs =  BlockedDiscreteOperator(np.array(blocke1))
    stop = timeit.default_timer()

    # Definition of Convolution Quadrature weights 
    start2 = timeit.default_timer()
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
        wei[ell]=   BlockedDiscreteOperator(np.array(storblock)) 
        wei[ell]= (dlt(rho*np.exp(2.0*np.pi*1j*ell/L))/tau)**(-m)*1.0/mu*wei[ell]
    for n in range(0, int(N + 1)):
        CQweights[n] = wei[0]  # Fourier Transform
        for ell in range(1, int(L)):
            CQweights[n] = CQweights[n] + np.exp(-2.0 * np.pi * 1j * n * ell / L) * wei[ell]
        CQweights[n] = rho ** (-n) / L * CQweights[n]
    stop2 = timeit.default_timer()
    
    #### Initial and starting data
    start3= timeit.default_timer()
    #return storage and dtmpsiko
    mvec=np.zeros([int(N + 1), nMAG])
    Ekoeff = np.zeros([int(N + 1), nFEM])
    Hkoeff = np.zeros([int(N + 1), nFEM])
    psikoeff= 1j * np.zeros([int(N + 1),  nBEM])
    randkoeff = 1j * np.zeros([int(N + 1), 2 * nBEM]) # storage variable for boundary per rk[j]=phi(t_j-0.5) bzw rk[j+1]=phi(t_j+1/2)
    dtmpsiko = 1j * np.zeros([int(N + 1), 2 * nBEM])
    
    # Initial Data and Input functions
    [mj,Ej,Hj,J] = initialdata(rm,ps,V3,fenics_space) 
    
    # actual time step values
    vjko=np.zeros([nMAG])
    Ejko=Ej.vector()[:]
    Hjko=Hj.vector()[:]
    psiko = np.zeros(nBEM)
    solu=np.zeros(2*nFEM+2*nBEM)

    # Initial data stored 
    mvec[0,:]=mj.vector()[:]
    Ekoeff[0, :] = np.real(Ejko)
    Hkoeff[0, :] = np.real(Hjko)
    psikoeff[0,:]= psiko
    randkoeff[0, :] = solu[2*nFEM:]
    
    # starting values t=tau
    if pt>1:
        [mvec_tmp ,Ekoeff_tmp, Hkoeff_tmp, phi_tmp, psi_tmp] = startingdata(T,rm, N, tau, h,pt,ps, m, rho,alstab, L, tolgmres,matpar,V3,VV,fenics_space,trace_space,trace_matrix)
        # For computation/gmres initial data
        vjko[:]=(mvec_tmp[:]-mvec[0,:])/tau
        Ejko[:]=np.real(Ekoeff_tmp[:])        
        Hjko[:]=np.real(Hkoeff_tmp[:])
        psiko[:]=np.real(psi_tmp[:])
        solu[:]= np.concatenate([Ejko,Hjko,np.real(phi_tmp[:]), psiko[:]/2])
        
        #storage
        mvec[1,:]=mvec_tmp[:]
        Ekoeff[1, :] = Ejko
        Hkoeff[1, :] = Hjko
        psikoeff[1,:]=psiko[:]
        randkoeff[1, :] = solu[2*nFEM:]
        
    stop3 = timeit.default_timer()
    print('Time for LHS: ', stop - start, ' for Calderon evaluation: ', stop2 - start2,' for Inidat: ', stop3 - start3)
    
    for j in range(pt-1, int(N)):  #### time stepping: update from timestep t_j to j+1
 
        #### Maxwell update
        
        Hj12ko,infoo= gmres(Mwv, Mwv.dot(Hjko)+tau/2/mu*D.dot(Ejko)-tau/2.0*C1.dot(psiko)- tau*0.5 *rhsLLGH.weak_form().transpose().dot(vjko),tol=tolgmres,x0=Hjko)
        if (infoo > 0):
            print("Failed to converge after " + str(infoo) + " iterations")
        Erko, infoo=gmres(Mwv, Mwv.dot(Ejko)-tau/eps* D.dot(Hj12ko) + ((j+1.0/2.0)*tau)**4* tau/eps* Mwv.dot(J.vector()[:]),tol=tolgmres,x0=Ejko)   
        if (infoo > 0):
            print("Failed to converge after " + str(infoo) + " iterations")
        #### Boundary solve
        
        # Convolution
        dtmpsiko[:, :] = randkoeff[:, :]  # compute \partial_t^m phi,  \partial_t^m psi,
        for ss in range(0, m):
            for r in range(0, j + 1):
                dtmpsiko[j + 1 - r, :] = (dtmpsiko[j + 1 - r, :] - dtmpsiko[j + 1 - r - 1, :]) / tau
        boundary_rhs = 1j * np.zeros(2 * nBEM)

        for kk in range(1, j + 2):  # Convolution, start bei 1, da psiko(0)=0
            dd = CQweights[j + 1 - kk].dot(dtmpsiko[kk, :])
            boundary_rhs[:] += dd[:]

        # Right hand side
        Rhhs = np.concatenate([eps*Mwv.dot(Erko+Ejko)/2, 2*alstab*tau*C1.dot(psiko),-boundary_rhs[:nBEM],-boundary_rhs[nBEM:]+C1T.dot(Hj12ko)])
#-boundary_rhs+np.concatenate([C0T.dot(Ejko/2+Erko/2),C1T.dot(Hj12ko)+2*alstab*tau/mu*C1T.dot(Minv.dot(C1.dot(psiko)))])

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
        phi12ko = solu[2*nFEM:2*nFEM+nBEM]
        psi12ko = solu[2*nFEM+nBEM:]
        psiko=2*psi12ko-psiko
            
        #### Maxwell update    
        
        Ejko,infoo =gmres(Mwv,  Mwv.dot(Erko)-tau/eps*C0.dot(phi12ko), tol=tolgmres,x0=Ejko)
        if (infoo > 0):
            print("Failed to converge after " + str(infoo) + " iterations")
        Hrko, infoo= gmres(Mwv, Mwv.dot(Hj12ko)+tau/2/mu*D.dot(Ejko)-tau/2/mu*C1.dot(psiko),tol=tolgmres,x0=Hjko )
        if (infoo > 0):
            print("Failed to converge after " + str(infoo) + " iterations")
        #### LLG update ########
        
        mhat = interpolate(Expression(['0','0','0'],degree=ps),V3)
        mr = interpolate(Expression(['0','0','0'],degree=ps),V3)
        for i in range(0,pt):
            mhat.vector()[:]=mhat.vector()[:] + float(gamma[i])*mvec[j+1-i-1,:]
            mr.vector()[:]=mr.vector()[:] - float(delta[i+1])*mvec[j+1-(i+1),:]
            
        mr.vector()[:]=mr.vector()[:]/float(delta[0])
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
        Hh= interpolate(Expression(['0','0','0'],degree=ps),fenics_space)
        Hh.vector()[:]=np.real(Hrko)
        
        # define LLG form
        dxr= dx(metadata={'quadrature_degree': 5})
        lhs = ((alpha*inner(v,phi)+inner(cross(mhat,v),phi)+tau/float(delta[0])*inner(nabla_grad(v),nabla_grad(phi)))*dxr
            + inner( dot(phi,mhat),lam)*dxr   + inner(dot(v,mhat),muu)*dxr+ tau* 0.5 *inner(v,phi)*dxr)

        rhs = (-inner(nabla_grad(mr),nabla_grad(phi))+inner(Hh, phi))*dxr 
        
        # compute solution
        vlam = Function(VV)
        solve(lhs == rhs, vlam)#,solver_parameters={"linear_solver": "gmres"},form_compiler_parameters={"optimize": True}
        (v,lam) = vlam.split(deepcopy=True)
        vjko[:] = v.vector()[:]
        
        # update magnetization and H
        mvec[j+1,:]= mr.vector()[:] + tau/float(delta[0]) * vjko[:];
        
        Hjko, infoo = gmres(Mwv, Mwv.dot(Hrko) -tau* 0.5 *  rhsLLGH.weak_form().transpose().dot( vjko[:]),tol=tolgmres, x0=Hjko)
        if (infoo > 0):
            print("Failed to converge after " + str(infoo) + " iterations")
        
        ### storage
        Ekoeff[j+1, :] = np.real(Ejko)
        Hkoeff[j+1, :] = np.real(Hjko)
        psikoeff[j+1,:]=psiko
        randkoeff[j+1, :] = solu[2*nFEM:]
        
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
    

def initialdata(rm,ps,V3,X):
    if rm>0:
        Ej=project(Expression(['0.00', '0.0', '0.0'], degree=ps), X)
        Hj = project(Expression(['0.0', '0.0', '0.0'], degree=ps), X)
        mag = interpolate(Expression(['1.00', '0.0', '0.0'], degree=ps), V3)

        J=project(Expression(['100.00','000.0','00.0'],degree=ps),X)
        mj = mag
        mj = mj / sqrt(dot(mj, mj))
        mj = project(mj, V3, solver_type='cg')
    else:
        Ej=project(Expression(['0.00', '0.0', '0.0'], degree=ps), X)
        Hj = project(Expression(['0.0', '0.0', '0.0'], degree=ps), X)
        mj = interpolate(Expression(['1.00', '0.0', '0.0'], degree=ps), V3)
        J=project(Expression(['100.00','000.0','00.0'],degree=ps),X)    
    return [mj,Ej,Hj,J]

def startingdata(T,rm, N, tau, h,k,ps, m, rho,alstab, L, tolgmres,matpar,V3,VV,fenics_space,trace_space,trace_matrix):
    if rm>0:
        [mvec_tmp ,Ekoeff_tmp, Hkoeff_tmp, phi_tmp, psi_tmp]=MLLGfunc((k-1)*tau,rm, 1.0/N*int(T*tau**(float(-1)/(k-1))), 1/T*N*tau**(float(k)/(k-1)), h,k-1,ps, m, rho,alstab, L/N, tolgmres,matpar,V3,VV,fenics_space,trace_space,trace_matrix)
    #( m, rho, LN, tolgmres, eps, mu, sig, alpha, Ce,k,meshh)
            #bdfllg_func(r,k-1,N,alpha,(k-1)*tau,tau**(float(k)/(k-1)),minit,mesh,VV,V3,V);
        #mvec_tmp=bdfllg_func(r,k-1,N,alpha,tau,tau,minit,mesh,VV,V3,V);
        #print(mvec_tmp)
        #int(T*i*tau**(float(-1)/(k-1))/N)
        #vjko[:]=(mvec_tmp[int(i*tau**(float(-1)/(k-1))/N),:]-mvec_tmp[int(i*tau**(float(-1)/(k-1))/N)-1,:])/(N*tau**(float(k)/(k-1)))
        return [mvec_tmp[1,:] ,Ekoeff_tmp[1,:], Hkoeff_tmp[1,:], phi_tmp[1,:], psi_tmp[1,:]]
    else:
        Ej=project(Expression(['0.00', '0.0', '0.0'], degree=ps), fenics_space)
        Hj = project(Expression(['0.0', '0.0', '0.0'], degree=ps), fenics_space)
        mj = interpolate(Expression(['1.00', '0.0', '0.0'], degree=ps), V3)
        return [mj.vector()[:],Ej.vector()[:],Hj.vector()[:],np.zeros(trace_space.global_dof_count),np.zeros(trace_space.global_dof_count)]
    
    