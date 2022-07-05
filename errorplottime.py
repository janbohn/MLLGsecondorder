from __future__ import division
from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
from maxwellrt0 import *
from MLLGfunctimesparseCA import *
#set_log_level(31)

href = 4


N0 = 1.0
mult=np.asarray([1.0,2.0,4.0,8.0,16.0,32.0,64.0])#]),128.0 #time refinement
refmult = 2.0

pt= 2
ps= 1

T=0.25

onof=1.0 # 1.0 -- coupled, 0.0 -- uncoupled

normm='L2' #'H1'#
normE= 'Hcurl'# 'L2'#
normH= 'Hcurl'#'L2'#


# Time integration and CQ approximation parameters # tolerance gmres and for CQ
Nref = np.amax(mult) * refmult * N0
Nvec = N0 * mult
tauref = float(T) / Nref
tau = float(T) / Nvec
tolgmres = 10 ** (-8)  
m = 0 #maybe at some places factor depending on m too much, in 2ndorder ist es richtig
astab=1.0
rhoref = tolgmres ** (0.5 * (Nref ** (-1)))
rho = tolgmres ** (0.5 * np.power(Nvec, -1))
Lref = 1 * Nref
L = 1 * Nvec

#Physikalische Parameter
eps = 1*1.1
mu = 1*1.2
sig = 1.3
alpha = 1.4
Ce = 1.5
matpar=[eps, mu, sig, alpha, Ce]
    
#Mesh and Spaces    
meshhref =UnitCubeMesh(int(href), int(href), int(href))#BoxMesh(Point(0,0,0),Point(1,2/href,2/href),int(href),2,2) #BoxMesh(Point(0,0,0),Point(href,href,1.0 ),int(href),int(href),1) #

Pr3 = VectorElement('Lagrange', meshhref.ufl_cell(), ps, dim=3);
V3 = FunctionSpace(meshhref, Pr3)

Pr = FiniteElement('P', meshhref.ufl_cell(), ps);
element = MixedElement([Pr3, Pr]);
VV = FunctionSpace(meshhref, element)

Xr = FiniteElement("N1curl", meshhref.ufl_cell(), ps)
X = FunctionSpace(meshhref, Xr)

trace_space, trace_matrix = nc1_tangential_trace(X)
trace_space.grid.plot()

nMAG = V3.dim()
nFEM = X.dim()
nBEM = trace_space.global_dof_count

#Exact solution // reference solution
if refmult>0:
    dd = Function(V3)
    dd2= Function(X)
    [mref, Eref, Href, phiref, psiref] = MLLGfunc(T,refmult,onof, Nref, tauref, href,pt,ps, m, rhoref,astab, Lref, tolgmres,matpar,V3,VV,X,trace_space,trace_matrix)
#(T, Nref, tauref, int(href), m, rhoref, Lref, tolgmres, eps, mu, sig,alpha, Ce,pt,meshhref)
                                              
else:   
    maprfkt = Function(V3)
    Eaprfkt = Function(X)
    Haprfkt = Function(X)
    mex=0
    Eex=0
    Hex=0
    
# Error values initialization    
err = np.zeros([len(mult)])
errmaxm = np.zeros([len(mult)])
errmaxE = np.zeros([len(mult)])
errmaxH = np.zeros([len(mult)])
errmaxPhi = np.zeros([len(mult)])
errmaxPsi = np.zeros([len(mult)])
    
#Precomputations for Timeintegration
# not so necessary?
# initial data ? 
# starting data von reference solution//exact solution ?


for i in range(0, len(mult)):
    errim=np.zeros(int(Nvec[i]+1))
    erriE=np.zeros(int(Nvec[i]+1))
    erriH=np.zeros(int(Nvec[i]+1))
    erriphi=np.zeros(int(Nvec[i]+1))
    erripsi=np.zeros(int(Nvec[i]+1))
    
    [mapr, Eapr, Hapr, phiapr, psiapr] = MLLGfunc(T,refmult,onof, Nvec[i], tau[i], href,pt,ps, m, rho[i], astab, L[i], tolgmres,matpar,V3,VV,X,trace_space,trace_matrix)
    #(T, Nvec[i], tau[i], int(href), m, rho[i], L[i], tolgmres, eps,mu, sig, alpha, Ce,pt,meshhref)
                                                  
    for j in range(0, int(Nvec[i]) + 1):
        if refmult>0: 
            #error Magnetiation
            dd.vector()[:] = mref[int(j * np.amax(mult) * refmult / mult[i]), :]-mapr[j, :]
            errim[j]=norm(dd, normm)
            errmaxm[i] = np.amax([errmaxm[i], errim[j]])
            #error E and H
            dd2.vector()[:]= Eref[int(j * np.amax(mult) * refmult / mult[i]), :]-Eapr[j, :]
            erriE[j]=norm(dd2,normE)
            errmaxE[i] = np.amax([errmaxE[i], erriE[j] ])
            dd2.vector()[:]= Href[int(j * np.amax(mult) * refmult / mult[i]), :]-Hapr[j, :]
            erriH[j]=norm(dd2,normH)
            errmaxH[i] = np.amax([errmaxH[i], erriH[j]])
            #error Phi and Psi
            if (j>0) and (j<int(Nvec[i])+1):
                ddPhi = bempp.api.GridFunction(trace_space, coefficients=phiref[int((j-0.5) * np.amax(mult) * refmult / mult[i]),:]/2+phiref[int((j-0.5) * np.amax(mult) * refmult / mult[i]+1),:]/2-phiapr[j,:])
                erriphi[j]=ddPhi.l2_norm()
            else: 
                erriphi[j]=0.0
            errmaxPhi[i] = np.amax([errmaxPhi[i], erriphi[j] ])
            ddPsi = bempp.api.GridFunction(trace_space, coefficients=psiref[int(j * np.amax(mult) * refmult / mult[i]),:]-psiapr[j,:])
            erripsi[j]=ddPsi.l2_norm()        
            errmaxPsi[i]= np.amax([errmaxPsi[i], erripsi[j]])
        else:
            maprfkt.vector()[:] = mapr[j, :]
            Eaprfkt.vector()[:] = Eapr[j, :]
            Haprfkt.vector()[:] = Hapr[j, :]
            Phiapr = bempp.api.GridFunction(trace_space, coefficients=phiapr[j,:])
            Psiapr = bempp.api.GridFunction(trace_space, coefficients=psiapr[j,:])
            #error Magnetization 
            errim[j]=errornorm(Expression(['1.00', '0.0', '0.0'], degree=ps),maprfkt, normm)
            errmaxm[i] = np.amax([errmaxm[i], errim[j]])
            #error E and H
            erriE[j]=errornorm(Expression(['0.0', '0.0', '0.0'], degree=ps),Eaprfkt,normE)
            errmaxE[i] = np.amax([errmaxE[i], erriE[j] ])
            erriH[j]=errornorm(Expression(['0.0', '0.0', '0.0'], degree=ps),Haprfkt,normH)
            errmaxH[i] = np.amax([errmaxH[i], erriH[j]])
            #error Phi and Psi
            erriphi[j]=Phiapr.l2_norm()#errornorm(,Phiapr)
            errmaxPhi[i] = np.amax([errmaxPhi[i], erriphi[j] ]) 
            erripsi[j]=Psiapr.l2_norm()#errornorm(,Psiapr)      
            errmaxPsi[i]= np.amax([errmaxPsi[i], erripsi[j]])
    #plot error over time
    plt.plot(errim)
    plt.show()
    plt.plot(erriE)
    plt.plot(erriH)
    plt.show()
    plt.plot(erripsi)# corresponds E
    plt.plot(erriphi)
    plt.show()

print('Results for m')
for i in range(0, len(mult)):
    dd = ' '
    if (i > 0):
        dd = dd + '  EOCtau:' + str(np.log(errmaxm[i] / errmaxm[i - 1]) / np.log(tau[i - 1] / tau[i]))
    print(tau[i], nMAG, meshhref.hmax(), errmaxm[i], dd)
print('Results for E')
for i in range(0, len(mult)):
    dd = ' '
    if (i > 0):
        dd = dd + '  EOCtau:' + str(np.log(errmaxE[i] / errmaxE[i - 1]) / np.log(tau[i - 1] / tau[i]))
    print(tau[i], nFEM, meshhref.hmax(), errmaxE[i], dd)
print('Results for H')
for i in range(0, len(mult)):
    dd = ' '
    if (i > 0):
        dd = dd + '  EOCtau:' + str(np.log(errmaxH[i] / errmaxH[i - 1]) / np.log(tau[i - 1] / tau[i]))
    print(tau[i], nFEM, meshhref.hmax(), errmaxH[i], dd)
print('Results for phi')
for i in range(0, len(mult)):
    dd = ' '
    if (i > 0):
        dd = dd + '  EOCtau:' + str(np.log(errmaxPhi[i] / errmaxPhi[i - 1]) / np.log(tau[i - 1] / tau[i]))
    print(tau[i], nBEM, meshhref.hmax(), errmaxPhi[i], dd)
print('Results for psi')
for i in range(0, len(mult)):
    dd = ' '
    if (i > 0):
        dd = dd + '  EOCtau:' + str(np.log(errmaxPsi[i] / errmaxPsi[i - 1]) / np.log(tau[i - 1] / tau[i]))
    print(tau[i], nBEM, meshhref.hmax(), errmaxPsi[i], dd)

if 1: #direct plot
    err=errmaxm;
    x = tau
    od1=err[0]* 1.2 *tau**2 /tau[0]**2
    plt.loglog(x, err, marker='o')
    plt.loglog(x, od1, ls='--')
    plt.xlabel('Time step size')
    plt.ylabel('Error Magnetization in '+ str(normm))
    plt.legend((r'$err(\tau)$', r'$O(\tau^2)$'), loc='lower right')
    plt.tight_layout()
    plt.savefig('books_read.png')
    plt.show()
    err=errmaxE;
    x = tau
    od1=err[0]* 1.2 *tau**2 /tau[0]**2
    plt.loglog(x, err, marker='o')
    plt.loglog(x, od1, ls='--')
    plt.xlabel('Time step size')
    plt.ylabel('Error E in '+ str(normE))
    plt.legend((r'$err(\tau)$', r'$O(\tau^2)$'), loc='lower right')
    plt.tight_layout()
    plt.savefig('books_read.png')
    plt.show()
    err = errmaxH;
    x = tau
    od1 = err[0] * 1.2 * tau**2 / tau[0]**2
    plt.loglog(x, err, marker='o')
    plt.loglog(x, od1, ls='--')
    plt.xlabel('Time step size')
    plt.ylabel('Error H in '+ str(normH))
    plt.legend((r'$err(\tau)$', r'$O(\tau^2)$'), loc='lower right')
    plt.tight_layout()
    plt.savefig('books_read.png')
    plt.show()
    err = errmaxPhi;
    x = tau
    od1 = err[0] * 1.2 * tau**2 / tau[0]**2
    plt.loglog(x, err, marker='o')
    plt.loglog(x, od1, ls='--')
    plt.xlabel('Time step size')
    plt.ylabel('Error Phi---H')
    plt.legend((r'$err(\tau)$', r'$O(\tau^2)$'), loc='lower right')
    plt.tight_layout()
    plt.savefig('books_read.png')
    plt.show()
    err = errmaxPsi;
    x = tau
    od1 = err[0] * 1.2 * tau**2 / tau[0]**2
    plt.loglog(x, err, marker='o')
    plt.loglog(x, od1, ls='--')
    plt.xlabel('Time step size')
    plt.ylabel('Error Psi---E')
    plt.legend((r'$err(\tau)$', r'$O(\tau^2)$'), loc='lower right')
    plt.tight_layout()
    plt.savefig('books_read.png')
    plt.show()
