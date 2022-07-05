from __future__ import division
from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
from maxwellrt0 import *
from MLLGfuncspacesparseCA import *
#set_log_level(31)

Nref = 4.0

h0 = 1
hmult=np.asarray([2,3,4])#,6,8,12,16,10#,8,16,20,12#6,,12,16
hrefadd = 4.0

pt= 2
ps= 1

T=0.5

onof=1.0 # 1.0 -- coupled, 0.0 -- uncoupled

normm='L2' #'H1'#
normE= 'Hcurl'#'L2'# 
normH= 'Hcurl'#'L2'#

# Time integration and Space and CQ approximation parameters # tolerance gmres and for CQ
href = np.amax(hmult)  * h0+hrefadd
hvec = h0 * hmult
tolgmres = 10 ** (-8)  # tolerance gmres
m = 0  # CQ approximation parameters
astab=1.0
tauref = float(T) / Nref
rhoref = tolgmres ** (0.5 * (Nref ** (-1)))
Lref = 1 * Nref

#Physikalische Parameter
eps = 1.1  
mu = 1.2  
sig = 1.3  
alpha = 1.4
Ce = 1.5
matpar=[eps, mu, sig, alpha, Ce]




meshh = np.ndarray([len(hmult)], dtype=np.object)
X = np.ndarray([len(hmult)], dtype=np.object)
V3 = np.ndarray([len(hmult)], dtype=np.object)
VV = np.ndarray([len(hmult)], dtype=np.object)
trace_space = np.ndarray([len(hmult)], dtype=np.object)
trace_matrix = np.ndarray([len(hmult)], dtype=np.object)
hlg= np.zeros(len(hmult))
nMAG = np.zeros(len(hmult))
nFEM = np.zeros(len(hmult))
nBEM = np.zeros(len(hmult))
for k in range(0, len(hmult)):
    meshh[k] =BoxMesh(Point(0,0,0),Point(1,1,0.08),int(hvec[k]),int(hvec[k]),2) # BoxMesh(Point(0,0,0),Point(1,1,2/href),int(hvec[k]),int(hvec[k]),2) #UnitCubeMesh(int(hvec[k]), int(hvec[k]), int(hvec[k]))# UnitCubeMesh(int(hvec[k]), int(hvec[k]), int(hvec[k]))
    Pr3 = VectorElement('Lagrange', meshh[k].ufl_cell(), ps, dim=3);
    V3[k] = FunctionSpace(meshh[k], Pr3 )  
    Pr = FiniteElement('P', meshh[k].ufl_cell(), ps);
    VV[k] = FunctionSpace(meshh[k], MixedElement([Pr3, Pr]))
    Xr = FiniteElement("N1curl", meshh[k].ufl_cell(), ps)
    X[k] = FunctionSpace(meshh[k], Xr)
    trace_space[k], trace_matrix[k] = nc1_tangential_trace(X[k])
    #trace_space[k].grid.plot()
    hlg[k]= meshh[k].hmax()
    nMAG[k] = V3[k].dim()
    nFEM[k] = X[k].dim()
    nBEM[k] = trace_space[k].global_dof_count
    
meshhref = BoxMesh(Point(0,0,0),Point(1,1,0.08),int(href),int(href),2)
#BoxMesh(Point(0,0,0),Point(1,1,2/href),int(href),int(href),2)#UnitCubeMesh(int(href), int(href), int(href))# 
Pr3 = VectorElement('Lagrange', meshhref.ufl_cell(), ps, dim=3);
V3ref = FunctionSpace(meshhref, Pr3)
Pr = FiniteElement('P', meshhref.ufl_cell(), ps);
VVref = FunctionSpace(meshhref, MixedElement([Pr3, Pr]))
Xr = FiniteElement("N1curl", meshhref.ufl_cell(), ps)
Xref = FunctionSpace(meshhref, Xr)
trace_spaceref, trace_matrixref = nc1_tangential_trace(Xref)
#trace_spaceref.grid.plot()
#Exact solution // reference solution
if hrefadd>0:
    dd = Function(V3ref)
    dd2= Function(Xref)
    mreffkt = Function(V3ref)
    Ereffkt = Function(Xref)
    Hreffkt = Function(Xref)
    exSolutionCode = ("0","0","0")
    dtexSolutionCode = ("0","0","0")
    [mref, Eref, Href, phiref, psiref] = MLLGfunc(T,hrefadd,onof, Nref, tauref, int(href),pt,ps, m, rhoref,astab, Lref, tolgmres,matpar,V3ref,VVref,Xref,trace_spaceref,trace_matrixref,exSolutionCode,dtexSolutionCode)
else:   
    #maprfkt = Function(V3ref)
    #Eaprfkt = Function(Xref)
    #Haprfkt = Function(Xref)
    mex=0
    Eex=0
    Hex=0
    exSolutionCode = ("-1*(pow(x[0],3)-0.75*x[0])*sin(multF*M_PI*t/tend)","sqrt(1-1*pow(pow(x[0],3)-0.75*x[0],2))","-1*(pow(x[0],3)-0.75*x[0])*cos(multF*M_PI*t/tend)")
    dtexSolutionCode = ("-1*(pow(x[0],3)-0.75*x[0])*cos(multF*M_PI*t/tend)*multF*M_PI/tend","0","1*(pow(x[0],3)-0.75*x[0])*sin(multF*M_PI*t/tend)*multF*M_PI/tend")
    '''exSolutionCode = ("(pow(x[0],2)+pow(x[1],2))>0.25-1e-14 ? 0 : multF*exp(-(tend+0.001)/(tend-t+0.001)/(0.25-(pow(x[0],2)+pow(x[1],2))))*x[0]",
            "(pow(x[0],2)+pow(x[1],2))>0.25-1e-14 ? 0 : multF*exp(-(tend+0.001)/(tend-t+0.001)/(0.25-(pow(x[0],2)+pow(x[1],2))))*x[1]",
            "(pow(x[0],2)+pow(x[1],2))>0.25-1e-14 ? 1 : sqrt(1-pow(multF,2)*exp(-2*(tend+0.001)/(tend-t+0.001)/(0.25-(pow(x[0],2)+pow(x[1],2))))*(pow(x[0],2)+pow(x[1],2)))")
 
    dtexSolutionCode = ("(pow(x[0],2)+pow(x[1],2))>0.25-1e-14 ? 0 : -(tend+0.001)/pow(tend-t+0.001,2)/(0.25-(pow(x[0],2)+pow(x[1],2)))*multF*exp(-(tend+0.001)/(tend-t+0.001)/(0.25-(pow(x[0],2)+pow(x[1],2))))*x[0]",
            "(pow(x[0],2)+pow(x[1],2))>0.25-1e-14 ? 0 : -(tend+0.001)/pow(tend-t+0.001,2)/(0.25-(pow(x[0],2)+pow(x[1],2)))*multF*exp(-(tend+0.001)/(tend-t+0.001)/(0.25-(pow(x[0],2)+pow(x[1],2))))*x[1]",
            "(pow(x[0],2)+pow(x[1],2))>0.25-1e-14 ? 0 : (tend+0.001)/pow(tend-t+0.001,2)/(0.25-(pow(x[0],2)+pow(x[1],2)))*pow(multF,2)*exp(-2*(tend+0.001)/(tend-t+0.001)/(0.25-(pow(x[0],2)+pow(x[1],2))))*(pow(x[0],2)+pow(x[1],2))/sqrt(1-pow(multF,2)*exp(-2*(tend+0.001)/(tend-t+0.001)/(0.25-(pow(x[0],2)+pow(x[1],2))))*(pow(x[0],2)+pow(x[1],2)))")'''
    
# Error values initialization    
err = np.zeros([len(hmult)])
errmaxm = np.zeros([len(hmult)])
errmaxE = np.zeros([len(hmult)])
errmaxH = np.zeros([len(hmult)])
errmaxPhi = np.zeros([len(hmult)])
errmaxPsi = np.zeros([len(hmult)])




mult=np.asarray([1])
Nvec=np.asarray([Nref])
refmult=1
i=0
errim=np.zeros(int(Nvec[i]+1))
erriE=np.zeros(int(Nvec[i]+1))
erriH=np.zeros(int(Nvec[i]+1))
erriphi=np.zeros(int(Nvec[i]+1))
erripsi=np.zeros(int(Nvec[i]+1))

for k in range(0, len(hmult)):    
    [mapr, Eapr, Hapr, phiapr, psiapr] = MLLGfunc(T,hrefadd,onof, Nref, tauref, int(hvec[k]),pt,ps, m, rhoref, astab, Lref, tolgmres,matpar,V3[k],VV[k],X[k],trace_space[k],trace_matrix[k],exSolutionCode,dtexSolutionCode) 
    maprfkt = Function(V3[k])
    Eaprfkt = Function(X[k])
    Haprfkt = Function(X[k])
    #MLLGfuncsparse(T, Nvec[i], tau[i], int(hvec[k]), m, rho[i], L[i], tolgmres, eps,mu, sig, theta, alpha, Ce)
    for j in range(0, int(Nvec[i]) + 1):
        if hrefadd>0: 
            #error Magnetiation
            maprfkt.vector()[:]= mapr[j, :]
            hilfe= interpolate(maprfkt,V3ref)
            dd.vector()[:] = mref[int(j * np.amax(mult) * refmult / mult[i]), :]-hilfe.vector()[:]
            errim[j]=norm(dd, normm)
            errmaxm[k] = np.amax([errmaxm[k], errim[j]])
            #error E and H
            Eaprfkt.vector()[:]=Eapr[j,:]
            hilfe2=interpolate(Eaprfkt,Xref)
            dd2.vector()[:]= Eref[int(j * np.amax(mult) * refmult / mult[i]), :]-hilfe2.vector()[:]
            erriE[j]=norm(dd2,normE)
            errmaxE[k] = np.amax([errmaxE[k], erriE[j] ])
            Haprfkt.vector()[:]=Hapr[j,:]
            hilfe2=interpolate(Haprfkt,Xref)
            dd2.vector()[:]= Href[int(j * np.amax(mult) * refmult / mult[i]), :]-hilfe2.vector()[:]
            erriH[j]=norm(dd2,normH)
            errmaxH[k] = np.amax([errmaxH[k], erriH[j]])
            #error Phi and Psi
            #ddPhi = bempp.api.GridFunction(trace_space[k], coefficients=phiref[int(j * np.amax(mult) * refmult / mult[i]),:]-phiapr[j,:])
            ddPhi = bempp.api.GridFunction(trace_space[k], coefficients=phiapr[j,:])
            erriphi[j]=ddPhi.l2_norm()
            errmaxPhi[k] = np.amax([errmaxPhi[k], erriphi[j] ])
            #ddPsi = bempp.api.GridFunction(trace_space[k], coefficients=psiref[int(j * np.amax(mult) * refmult / mult[i]),:]-psiapr[j,:])
            ddPsi = bempp.api.GridFunction(trace_space[k], coefficients=psiapr[j,:])
            erripsi[j]=ddPsi.l2_norm()        
            errmaxPsi[k]= np.amax([errmaxPsi[k], erripsi[j]])
        else:
            class MyExpressionE(UserExpression):
                def eval(self, value, x):
                    s1=sin(np.pi*x[0])
                    s2=sin(np.pi*x[1])
                    s3=sin(np.pi*x[2])
                    s1s=s1**2
                    s2s=s2**2
                    s3s=s3**2
                    value[0] = s1s*s2s*s3s    * (j*tauref)**(2)
                    value[1] = 0
                    value[2] = 0 
                def value_shape(self):
                    return (3,)
            class MyExpressionH(UserExpression):
                def eval(self, value, x):
                    s1=sin(np.pi*x[0])
                    s2=sin(np.pi*x[1])
                    s3=sin(np.pi*x[2])
                    s1s=s1**2
                    s2s=s2**2
                    s3s=s3**2
                    value[0] = 0
                    value[1] =-2*np.pi*s1s*s2s*s3*cos(np.pi*x[2])  *(j*tauref)**(3)/3 /matpar[1]
                    value[2] = 2*np.pi*s1s*s2*cos(np.pi*x[1])*s3s  *(j*tauref)**(3)/3 /matpar[1]
                def value_shape(self):
                    return (3,)    
            maprfkt.vector()[:] = mapr[j, :]
            Eaprfkt.vector()[:] = Eapr[j, :]
            Haprfkt.vector()[:] = Hapr[j, :] 
            Phiapr = bempp.api.GridFunction(trace_space[k], coefficients=phiapr[j,:])
            Psiapr = bempp.api.GridFunction(trace_space[k], coefficients=psiapr[j,:])
            #error Magnetization 
            errim[j]=errornorm(Expression(exSolutionCode,degree=ps,t=j*tauref,tend=10*T,multF=3),maprfkt, normm)
            errmaxm[k] = np.amax([errmaxm[k], errim[j]])
            #error E and H
            erriE[j]=errornorm(MyExpressionE(degree=ps),Eaprfkt,normE)
            errmaxE[k] = np.amax([errmaxE[k], erriE[j] ])
            erriH[j]=errornorm(MyExpressionH(degree=ps),Haprfkt,normH)
            errmaxH[k] = np.amax([errmaxH[k], erriH[j]])
            #error Phi and Psi
            erriphi[j]=Phiapr.l2_norm()#errornorm(,Phiapr)
            errmaxPhi[k] = np.amax([errmaxPhi[k], erriphi[j] ]) 
            erripsi[j]=Psiapr.l2_norm()#errornorm(,Psiapr)      
            errmaxPsi[k]= np.amax([errmaxPsi[k], erripsi[j]])
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
    for k in range(0, len(hmult)):
        dd = ' '
        if (k > 0):
            dd = dd + 'EOCh:' + str( np.log(errmaxm[ k] / errmaxm[ k - 1]) / np.log(hlg[k] / hlg[k - 1]))
        if (i > 0):
            dd = dd + '  EOCtau:' + str(np.log(errmax[i, k] / errmax[i - 1, k]) / np.log(tau[i - 1] / tau[i]))
        # print(tau[i], nMAG[k], meshh[k].hmax(), err[ i, k])  # 'EOCtau:',  np.log(errmax[i,k]/errmax[i-1,k])/np.log(tau[k]/tau[k-1]) , 'EOCh:', 3.0* np.log(errmax[i,k]/errmax[i,k-1])/np.log(nMAG[k]/nMAG[k-1]))
        print(tauref, nMAG[k], meshh[k].hmax(), errmaxm[ k], dd)
print('Results for E')
for i in range(0, len(mult)):
    for k in range(0, len(hmult)):
        dd = ' '
        if (k > 0):
            dd = dd + 'EOCh:' + str( np.log(errmaxE[ k] / errmaxE[ k - 1]) / np.log(hlg[k] / hlg[k - 1]))
        if (i > 0):
            dd = dd + '  EOCtau:' + str(np.log(errmaxE[i, k] / errmaxE[i - 1, k]) / np.log(tau[i - 1] / tau[i]))
        # print(tau[i], nFEM[k], meshh[k].hmax(), errE[i, k])  # 'EOCtau:',  np.log(errmax[i,k]/errmax[i-1,k])/np.log(tau[k]/tau[k-1]) , 'EOCh:', 3.0* np.log(errmax[i,k]/errmax[i,k-1])/np.log(nMAG[k]/nMAG[k-1]))
        print(tauref, nFEM[k], meshh[k].hmax(), errmaxE[ k], dd)
print('Results for H')
for i in range(0, len(mult)):
    for k in range(0, len(hmult)):
        dd = ' '
        if (k > 0):
            dd = dd + 'EOCh:' + str( np.log(errmaxH[ k] / errmaxH[k - 1]) / np.log(hlg[k] / hlg[k - 1]))

        if (i > 0):
            dd = dd + '  EOCtau:' + str(np.log(errmaxH[i, k] / errmaxH[i - 1, k]) / np.log(tau[i - 1] / tau[i]))
        # print(tau[i], nFEM[k], meshh[k].hmax(), errH[i, k])  # 'EOCtau:',  np.log(errmax[i,k]/errmax[i-1,k])/np.log(tau[k]/tau[k-1]) , 'EOCh:', 3.0* np.log(errmax[i,k]/errmax[i,k-1])/np.log(nMAG[k]/nMAG[k-1]))
        print(tauref, nFEM[k], meshh[k].hmax(), errmaxH[ k], dd)
print('Results for phi')
for i in range(0, len(mult)):
    for k in range(0, len(hmult)):
        dd = ' '
        if (k > 0):
            dd = dd + 'EOCh:' + str(
                3.0 * np.log(errmaxPhi[ k] / errmaxPhi[ k - 1]) / np.log((nBEM[k]) / (nBEM[k - 1])))#

        if (i > 0):
            dd = dd + '  EOCtau:' + str(np.log(errmaxPhi[i, k] / errmaxPhi[i - 1, k]) / np.log(tau[i - 1] / tau[i]))
        #print(tauref, nBEM[k], meshh[k].hmax(), errPhi[ k])  # 'EOCtau:',  np.log(errmax[i,k]/errmax[i-1,k])/np.log(tau[k]/tau[k-1]) , 'EOCh:', 3.0* np.log(errmax[i,k]/errmax[i,k-1])/np.log(nMAG[k]/nMAG[k-1]))
        print(tauref, nBEM[k], meshh[k].hmax(), errmaxPhi[ k], dd)
        # print (tau[i],nMAG[k],test[i,k])
print('Results for psi')
for i in range(0, len(mult)):
    for k in range(0, len(hmult)):
        dd = ' '
        if (k > 0):
           dd = dd + 'EOCh:' + str(
               3.0 * np.log(errmaxPsi[ k] / errmaxPsi[ k - 1]) / np.log((nBEM[k]) / (nBEM[k - 1])))

        if (i > 0):
            dd = dd + '  EOCtau:' + str(np.log(errmaxPsi[i, k] / errmaxPsi[i - 1, k]) / np.log(tau[i - 1] / tau[i]))
        # print(tau[i], nBEM[k], meshh[k].hmax(), errPhi[i, k])  # 'EOCtau:',  np.log(errmax[i,k]/errmax[i-1,k])/np.log(tau[k]/tau[k-1]) , 'EOCh:', 3.0* np.log(errmax[i,k]/errmax[i,k-1])/np.log(nMAG[k]/nMAG[k-1]))
        print(tauref, nBEM[k], meshh[k].hmax(), errmaxPsi[ k], dd)
        #print (tau[i],nMAG[k],test[i,k])
#
if 1:
    err = errmaxm.T;
    x = hlg  # 0.25 * np.asarray([0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125])
    od1 = err[0] * 1.2 * hlg / hlg[0]  # *np.asarray([1.0,0.5,0.25,0.125,0.0625,0.03125,0.015625])
    plt.loglog(x, err, marker='o')
    plt.loglog(x, od1, ls='--')
    plt.xlabel('mesh size')
    plt.ylabel('Error m in '+ str(normm))
    plt.legend((r'$err(h)$', r'$O(h)$'), loc='lower right')
    plt.tight_layout()
    #plt.savefig('books_read.png')
    plt.show()
    err = errmaxE.T;
    x = hlg  # 0.25 * np.asarray([0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125])
    od1 = err[0] * 1.2 * hlg / hlg[0]  # *np.asarray([1.0,0.5,0.25,0.125,0.0625,0.03125,0.015625])
    plt.loglog(x, err, marker='o')
    plt.loglog(x, od1, ls='--')
    plt.xlabel('mesh size')
    plt.ylabel('Error E in '+ str(normE))
    plt.title(' T= '+ str(T)+' and tau= '+str(tauref))
    plt.legend((r'$err(h)$', r'$O(h)$'), loc='lower right')
    plt.tight_layout()
    #plt.savefig('books_read.png')
    plt.show()
    err = errmaxH.T;
    x = hlg   # 0.25 * np.asarray([0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125])
    od1 = err[0] * 1.2 * hlg / hlg[0]   # *np.asarray([1.0,0.5,0.25,0.125,0.0625,0.03125,0.015625])
    plt.loglog(x, err, marker='o')
    plt.loglog(x, od1, ls='--')
    plt.xlabel('mesh size')
    plt.ylabel('Error H in '+ str(normH) )
    plt.title(' T= '+ str(T)+' and tau= '+str(tauref))
    plt.legend((r'$err(h)$', r'$O(h)$'), loc='lower right')
    plt.tight_layout()
    plt.show()
    err = errmaxPhi;
    x = hlg  # 0.25 * np.asarray([0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125])
    od1 = err[0] * 1.2 * hlg / hlg[0]  # *np.asarray([1.0,0.5,0.25,0.125,0.0625,0.03125,0.015625])
    plt.loglog(x, err, marker='o')
    plt.loglog(x, od1, ls='--')
    plt.xlabel('mesh size')
    plt.ylabel('Error Phi -- H')
    plt.title(' T= '+ str(T)+' and tau= '+str(tauref))
    plt.legend((r'$err(h)$', r'$O(h)$'), loc='lower right')
    plt.tight_layout()
    plt.show()
    err = errmaxPsi;
    x = hlg  # 0.25 * np.asarray([0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125])
    od1 = err[0] * 1.2 * hlg / hlg[0]  # *np.asarray([1.0,0.5,0.25,0.125,0.0625,0.03125,0.015625])
    plt.loglog(x, err, marker='o')
    plt.loglog(x, od1, ls='--')
    plt.xlabel('mesh size')
    plt.ylabel('Error Psi -- E')
    plt.title(' T= '+ str(T)+' and tau= '+str(tauref))
    plt.legend((r'$err(h)$', r'$O(h)$'), loc='lower right')
    plt.tight_layout()
    plt.show()

'''for j in range(0,int(Nref/refmult)):
    print j*tauref

        if np.remainder(float(j+1)/refmult*mult[i],np.amax(mult)) == 0: 
            print(float(j+1)/refmult*tau[i],mult[i])
            #print(tau[i])

            dd=project(m0[i]-mref,V3)
            err[i]= err[i]+tau[i]* norm(dd,'l2')**2 
            errmax[i]= np.amax([errmax[i],norm(dd,'l2')])
            test[i]=test[i]+1
for i in range(0,len(mult)):
        err[i]=sqrt(err[i])
        print (tau[i],err[i])
        print (tau[i],errmax[i])
        print (tau[i],test[i])

'''
