""" PROGRAMA QUE CALCULA LA RESPUESTA DINAMICA DE UNA VIGA EN VOLADIZO
(EMPOTRADA EN X=0) Y SOMETIDA A UNA FUERZA PUNTUAL QUE VARIA CON EL TIEMPO
EN X=L, EL METODO DE SOLUCION ES ELEMENTOS FINITOS EN EL DOMINIO DEL 
ESPACIO-TIEMPO Y EMPLEANDO EL METODO BETA DE NEWMARK.

SE REALIZA UNA COMPARACION CON LA SOLUCION ANALITICA EN EL DOMINIO DEL
ESPACIO-FRECUENCIA

Elaborado por: Juan Camilo Molina Villegas
Fecha: 2016-10-20
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

plt.close("all")

###############################################
### PARTE 1: SOLUCION POR ELEMENTOS FINITOS ###
###############################################

#   1.0 PROPIEDADES DEL MODELO

#   1.1 Propiedades mecanicas, inerciales y geometricas

EI=1000000
m=1000
L=10

#	1.2 Discretizacion con elementos finitos

NEle=4

#   1.3 Discretizacion espacial (numero de observadores)

Nx=100

#   1.4 Discretizacion temporal

Nt=2**11
Dt=0.02

#   1.5 Definicion de la fuerza externa (pulso de Ricker)

ts=4
tp=1.5

#   2.0 CALCULOS PRELIMINARES

#	2.1 Longitud de los elementos

LE=L/NEle

#   2.1 Discretizacion espacial

x=np.linspace(0,L,Nx)
xNodos=np.linspace(0,L,NEle+1)

#   2.2 Discretizacion temporal

t=np.linspace(0,Nt-1,Nt)*Dt

#   2.3 Ubicacion de los observadores (elemento y xAux)

xAux=np.zeros([Nx])
UbiObs=np.zeros([Nx],dtype=int)

EleIni=0

for ix in range(Nx):
    for iEle in np.linspace(EleIni,NEle,NEle-EleIni+1,dtype=int):
        if x[ix]<=(iEle+1)*LE:
            EleIni=iEle
            UbiObs[ix]=iEle         
            xAux[ix]=x[ix]-iEle*LE
            break

#   2.4 Calculo de la fuerza externa (pulso de Ricker) en el dominio del tiempo

a=np.pi*(t-ts)/tp
FueTie=(a**2-0.5)*np.exp(-(a**2))


#   3.0 FORMULACION POR ELEMENTOS FINITOS

kEle=np.zeros([4,4],dtype=float)
mEle=np.zeros([4,4],dtype=float)
K=np.zeros([2*(NEle+1),2*(NEle+1)],dtype=float)
M=np.zeros([2*(NEle+1),2*(NEle+1)],dtype=float)

#   3.1 Matriz de rigidez y de masa de un elemento tipico

kEle[0,0]=12*EI/LE**3
kEle[2,2]=kEle[0,0]
kEle[0,1]=6*EI/LE**2
kEle[1,0]=kEle[0,1]
kEle[2,3]=-kEle[0,1]
kEle[3,2]=-kEle[0,1]
kEle[0,2]=-12*EI/LE**3
kEle[2,0]=kEle[0,2]
kEle[0,3]=6*EI/LE**2
kEle[3,0]=kEle[0,3]
kEle[1,2]=-kEle[0,3]
kEle[2,1]=-kEle[0,3]
kEle[1,1]=4*EI/LE
kEle[3,3]=kEle[1,1]
kEle[1,3]=2*EI/LE
kEle[3,1]=kEle[1,3]

mEle[0,0]=156/420*m*LE
mEle[2,2]=mEle[0,0]
mEle[0,1]=22/420*m*LE**2
mEle[1,0]=mEle[0,1]
mEle[2,3]=-mEle[0,1]
mEle[3,2]=-mEle[0,1]
mEle[0,2]=54/420*m*LE
mEle[2,0]=mEle[0,2]
mEle[0,3]=-13/420*m*LE**2
mEle[3,0]=mEle[0,3]
mEle[1,2]=-mEle[0,3]
mEle[2,1]=-mEle[0,3]
mEle[1,1]=4/420*m*LE**3
mEle[3,3]=mEle[1,1]
mEle[1,3]=-3/420*m*LE**3
mEle[3,1]=mEle[1,3]

#   3.2 Ensamblaje de la matriz de rigidez  y de masa de todo el sistema

for Ele in range(NEle):
    K[2*Ele+0:2*Ele+2,2*Ele+0:2*Ele+2]=K[2*Ele+0:2*Ele+2,2*Ele+0:2*Ele+2]+kEle[0:2,0:2]
    K[2*Ele+0:2*Ele+2,2*Ele+2:2*Ele+4]=K[2*Ele+0:2*Ele+2,2*Ele+2:2*Ele+4]+kEle[0:2,2:4]
    K[2*Ele+2:2*Ele+4,2*Ele+0:2*Ele+2]=K[2*Ele+2:2*Ele+4,2*Ele+0:2*Ele+2]+kEle[2:4,0:2]
    K[2*Ele+2:2*Ele+4,2*Ele+2:2*Ele+4]=K[2*Ele+2:2*Ele+4,2*Ele+2:2*Ele+4]+kEle[2:4,2:4]

    M[2*Ele+0:2*Ele+2,2*Ele+0:2*Ele+2]=M[2*Ele+0:2*Ele+2,2*Ele+0:2*Ele+2]+mEle[0:2,0:2]
    M[2*Ele+0:2*Ele+2,2*Ele+2:2*Ele+4]=M[2*Ele+0:2*Ele+2,2*Ele+2:2*Ele+4]+mEle[0:2,2:4]
    M[2*Ele+2:2*Ele+4,2*Ele+0:2*Ele+2]=M[2*Ele+2:2*Ele+4,2*Ele+0:2*Ele+2]+mEle[2:4,0:2]
    M[2*Ele+2:2*Ele+4,2*Ele+2:2*Ele+4]=M[2*Ele+2:2*Ele+4,2*Ele+2:2*Ele+4]+mEle[2:4,2:4]


# 3.3 Beta de Newmark

uNodos=np.zeros([2*(NEle+1),Nt])
vNodos=np.zeros([2*(NEle+1),Nt])
aNodos=np.zeros([2*(NEle+1),Nt])

Alpha=1/2
Beta=1/4
Gamma=2*Beta

a1=Alpha*Dt
a2=(1-Alpha)*Dt
a3=1/(Beta*(Dt**2))
a4=a3*Dt
a5=1/Gamma-1

K1=K[2:,2:]+a3*M[2:,2:]
F=np.zeros([2*NEle,1])
A=np.zeros([2*NEle,1])


for i in range(Nt-2):
    F[2*NEle-2,0]=FueTie[i+1]
    A[:,0]=a3*uNodos[2:,i]+a4*vNodos[2:,i]+a5*aNodos[2:,i]
    F1=F+np.dot(M[2:,2:],A)
    uNodos[2:,i+1]=np.squeeze(np.dot(np.linalg.inv(K1),F1))
    aNodos[2:,i+1]=a3*(uNodos[2:,i+1]-uNodos[2:,i])-a4*vNodos[2:,i]-a5*aNodos[2:,i]
    vNodos[2:,i+1]=vNodos[2:,i]+a2*aNodos[2:,i]+a1*aNodos[2:,i+1]


#   3.4 Calculo del desplazamiento en los puntos no nodales

vTieFEM=np.zeros([Nx,Nt],dtype=float)

for ix in range(Nx):
    Psi1=1-3*(xAux[ix]/LE)**2+2*(xAux[ix]/LE)**3
    Psi2=(xAux[ix]/LE)*((1-xAux[ix]/LE)**2)*LE
    Psi3=3*(xAux[ix]/LE)**2-2*(xAux[ix]/LE)**3    
    Psi4=(xAux[ix]/LE)*((xAux[ix]/LE)**2-xAux[ix]/LE)*LE    
    vTieFEM[ix,:]=Psi1*uNodos[2*UbiObs[ix],:]+Psi2*uNodos[2*UbiObs[ix]+1,:]+Psi3*uNodos[2*UbiObs[ix]+2,:]+Psi4*uNodos[2*UbiObs[ix]+3,:]    


##################################################################
### PARTE 2: SOLUCION ANALITICA EN EL DOMINIO DE LA FRECUENCIA ###
##################################################################

#   4.0 PRELIMINARES

#   4.1 Vector de frecuencias

f=np.zeros(Nt)

fMax=1/(2*Dt)
Df=fMax/(Nt/2)

f[0:int(Nt/2+1)]=np.linspace(0,int(Nt/2),int(Nt/2+1),dtype=int)*Df
f[int(Nt/2+1):]=-f[int(Nt/2-1):0:-1]

f[0]=f[1]/20

OmeIma=1.5*np.pi/np.max(t)
Ome=2*np.pi*f-1j*OmeIma

#   4.2 Calculo del pulso de Ricker en el dominio de la frecuencia

wp=2*np.pi/tp
b=Ome/wp

FueOme=-tp/np.sqrt(np.pi)*(b**2)*np.exp(-(b**2))*np.exp(-1j*Ome*ts)

#   4.3 Figura espectro de Fourier de la fuerza externa

plt.figure(1,figsize=(6,4))
plt.plot(f[0:int(Nt/2+1)],np.abs(FueOme[0:int(Nt/2+1)]),color='k')
plt.xlabel(r'$f$ [Hz]',fontsize=16)
plt.ylabel(r'$|F(\omega)|$',fontsize=16)


#   5.0 RESPUESTA ANALITICA EN EL DOMINIO DE LA FRECUENCIA

vOmeAna=np.zeros([Nx,Nt],dtype=complex)

for iOme in range(int(Nt/2+1)):
    Lambda=(m*Ome[iOme]**2/EI)**0.25*L
    s=np.sin(Lambda)
    c=np.cos(Lambda)
    sh=np.sinh(Lambda)
    ch=np.cosh(Lambda)

    vOmeAna[:,iOme]=(L/Lambda)**3*((c+ch)*(np.sin(Lambda*x/L)-np.sinh(Lambda*x/L))-(s+sh)*(np.cos(Lambda*x/L)-np.cosh(Lambda*x/L)))/(2*(1+c*ch))*FueOme[iOme]/EI
    
    if iOme>0:
        vOmeAna[:,Nt-iOme]=np.conj(vOmeAna[:,iOme]) 


#   6.0 RESPUESTA ANALITICA EN EL DOMINIO DEL TIEMPO

vTieAna=np.zeros([Nx,Nt],dtype=float)

for ix in range(Nx):
    vTieAna[ix,:]=np.real(sp.ifft(vOmeAna[ix,:])/Dt)
    vTieAna[ix,:]=vTieAna[ix,:]*np.exp(OmeIma*t)

##########################################
### PARTE 3: COMPARACION DE RESULTADOS ###
##########################################


#   7.0 COMPARACION DE LOS DESPLAZAMIENTOS EN EL EXTREMO DERECHO DE LA VIGA

plt.figure(2)
plt.plot(t,vTieAna[Nx-1,:],color='k',linewidth=2,label='Fourier')
plt.plot(t,uNodos[2*NEle,:],color='r',linestyle='--',linewidth=2,label='E.F.')
plt.legend()
plt.xlabel(r'$t$ [s]',fontsize=16)
plt.ylabel(r'$v(L,t)$',fontsize=16)
plt.grid("on")

plt.savefig('Comparacion.pdf')

#   8.0 ANIMACION CON LA COMPARACION DE LOS CAMPOS DE DESPLAZAMIENTO

tMaxFig=20
DitAni=1

vMax=np.max(np.max(np.abs(vTieAna)))
NtAni=np.ceil(tMaxFig/(DitAni*Dt))
itAni=np.linspace(0,DitAni*(NtAni-1),NtAni,dtype=int)

fig=plt.figure(3,figsize=(10,4)) 

for it in itAni:
    plt.clf()
    plt.plot(x,vTieAna[:,it],color='r',linestyle='--',label='Analitico',linewidth=2)
    plt.plot(x,vTieFEM[:,it],'k',label='FEM',linewidth=2)
    plt.plot(xNodos,uNodos[0:2*NEle+1:2,it],'ob',label='Nodos')
    plt.legend(loc='upper left')    
    plt.title('t='+str(t[it])+' s',fontsize=20)
    plt.ylim([-vMax,vMax])
    plt.pause(0.000005)
    
plt.show()