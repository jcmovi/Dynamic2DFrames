""" PROGRAMA QUE CALCULA LA RESPUESTA DINAMICA DE UNA VIGA EN VOLADIZO
(EMPOTRADA EN X=0) Y SOMETIDA A UNA FUERZA PUNTUAL QUE VARIA CON EL TIEMPO
EN X=L, EL METODO DE SOLUCION ES ELEMENTOS FINITOS EN EL DOMINIO DE LA 
FRECUENCIA EMPLEANDO LA MATRIZ DE RIGIDEZ Y FUNCIONES DE FORMA ANALITICAS.

SE REALIZA UNA COMPARACION CON LA SOLUCION ANALITICA

Elaborado por: Juan Camilo Molina Villegas
Fecha: 18/10/2016
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

plt.close("all")

#   1.0 PROPIEDADES DEL MODELO

#   1.1 Propiedades mecanicas, inerciales y geometricas

EI=10000000
m=1000
L=10

#	1.2 Discretizacion con elementos finitos

NEle=4

#   1.3 Discretizacion espacial

Nx=100

#   1.4 Discretizacion temporal

Nt=2**10
Dt=0.05

#   1.5 Amplitud maxima de la fuerza externa

ts=3
tp=1.5

#   2.0 CALCULOS PRELIMINARES

#	2.1 Longitud de los elementos

LE=L/NEle

#   2.1 Discretizacion espacial

x=np.linspace(0,L,Nx)
xNodos=np.linspace(0,L,NEle+1)


#   2.2 Discretizacion temporal

t=np.linspace(0,Nt-1,Nt)*Dt

#   2.3 Vector de frecuencias

f=np.zeros(Nt)

fMax=1/(2*Dt)
Df=fMax/(Nt/2)

f[0:int(Nt/2+1)]=np.linspace(0,int(Nt/2),int(Nt/2+1),dtype=int)*Df
f[int(Nt/2+1):]=-f[int(Nt/2-1):0:-1]

f[0]=f[1]/20

OmeIma=1.5*np.pi/np.max(t)
Ome=2*np.pi*f-1j*OmeIma

#   2.4 Ubicacion de los observadores (elemento y xAux)

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

#   2.5 Calculo del pulso de Ricker en el dominio del tiempo

a=np.pi*(t-ts)/tp
FueTie=(a**2-0.5)*np.exp(-(a**2))

#   2.6 Calculo del pulso de Ricker en el dominio de la frecuencia

wp=2*np.pi/tp
b=Ome/wp

FueOme=-tp/np.sqrt(np.pi)*(b**2)*np.exp(-(b**2))*np.exp(-1j*Ome*ts)

#   2.7 Figura espectro de Fourier de la fuerza externa

plt.figure(1,figsize=(6,4))
plt.plot(f[0:int(Nt/2+1)],np.abs(FueOme[0:int(Nt/2+1)]),color='k')
plt.xlabel(r'$f$ [Hz]',fontsize=16)
plt.ylabel(r'$|F(\omega)|$',fontsize=16)


#   3.0 FORMULACION POR ELEMENTOS FINITOS

kEle=np.zeros([4,4],dtype=complex)	#Matriz de rigidez de un elemento tipico
FueNodalesOme=np.zeros([2*(NEle+1)],dtype=complex)                #Vector de fuerzas generalizadas en los nodos
DesNodalesOme=np.zeros([2*(NEle+1),Nt],dtype=complex)
vOmeFEM=np.zeros([Nx,Nt],dtype=complex)      #Vector con la respuesta en el dominio de la frecuencia de los Nx puntos


for iOme in range(int(Nt/2+1)):
    #   3.1 Variables auxiliares

    Lambda=((m*Ome[iOme]**2/EI)**0.25)*LE
    s=np.sin(Lambda)
    c=np.cos(Lambda)
    sh=np.sinh(Lambda)
    ch=np.cosh(Lambda)


    #   3.2 Ensamblaje de la matriz de rigidez    
    
    kEle[0,0]=(s*ch+c*sh)/(1-c*ch)*(Lambda/LE)**3*EI
    kEle[2,2]=kEle[0,0]
    kEle[0,1]=(s*sh)/(1-c*ch)*(Lambda/LE)**2*EI
    kEle[1,0]=kEle[0,1]
    kEle[2,3]=-kEle[0,1]
    kEle[3,2]=-kEle[0,1]
    kEle[0,2]=-(s+sh)/(1-c*ch)*(Lambda/LE)**3*EI
    kEle[2,0]=kEle[0,2]
    kEle[0,3]=(ch-c)/(1-c*ch)*(Lambda/LE)**2*EI
    kEle[3,0]=kEle[0,3]
    kEle[1,2]=-kEle[0,3]
    kEle[2,1]=-kEle[0,3]
    kEle[1,1]=(s*ch-c*sh)/(1-c*ch)*(Lambda/LE)*EI
    kEle[3,3]=kEle[1,1]
    kEle[1,3]=(sh-s)/(1-c*ch)*(Lambda/LE)*EI
    kEle[3,1]=kEle[1,3]
    
    kSis=np.zeros([2*(NEle+1),2*(NEle+1)],dtype=complex)    #Matriz de rigidez de todo el sistema    
    
    for Ele in range(NEle):
        kSis[2*Ele+0:2*Ele+2,2*Ele+0:2*Ele+2]=kSis[2*Ele+0:2*Ele+2,2*Ele+0:2*Ele+2]+kEle[0:2,0:2]
        kSis[2*Ele+0:2*Ele+2,2*Ele+2:2*Ele+4]=kSis[2*Ele+0:2*Ele+2,2*Ele+2:2*Ele+4]+kEle[0:2,2:4]
        kSis[2*Ele+2:2*Ele+4,2*Ele+0:2*Ele+2]=kSis[2*Ele+2:2*Ele+4,2*Ele+0:2*Ele+2]+kEle[2:4,0:2]
        kSis[2*Ele+2:2*Ele+4,2*Ele+2:2*Ele+4]=kSis[2*Ele+2:2*Ele+4,2*Ele+2:2*Ele+4]+kEle[2:4,2:4]

    #   3.3 Vector de fuerzas equivalentes

    FueNodalesOme[2*NEle]=FueOme[iOme]
    
    #   3.4 Calculo del vector de desplazamiento nodales
    
    DesNodalesOme[2:,iOme]=np.linalg.solve(kSis[2:,2:],FueNodalesOme[2:])    
    
    #   3.5 Vector de desplazamiemtos en puntos no nodales    
    
    for ix in range(Nx):
        Psi1=((-c*sh-s*ch)*np.sin(Lambda*xAux[ix]/LE)+(1+s*sh-c*ch)*np.cos(Lambda*xAux[ix]/LE)+(c*sh+s*ch)*np.sinh(Lambda*xAux[ix]/LE)+(1-s*sh-c*ch)*np.cosh(Lambda*xAux[ix]/LE))/(2*(1-c*ch))
        Psi2=(LE/Lambda)*((1-s*sh-c*ch)*np.sin(Lambda*xAux[ix]/LE)+(s*ch-c*sh)*np.cos(Lambda*xAux[ix]/LE)+(1+s*sh-c*ch)*np.sinh(Lambda*xAux[ix]/LE)+(c*sh-s*ch)*np.cosh(Lambda*xAux[ix]/LE))/(2*(1-c*ch))
        Psi3=((sh+s)*np.sin(Lambda*xAux[ix]/LE)+(c-ch)*np.cos(Lambda*xAux[ix]/LE)+(-sh-s)*np.sinh(Lambda*xAux[ix]/LE)+(ch-c)*np.cosh(Lambda*xAux[ix]/LE))/(2*(1-c*ch))
        Psi4=(LE/Lambda)*((c-ch)*np.sin(Lambda*xAux[ix]/LE)+(sh-s)*np.cos(Lambda*xAux[ix]/LE)+(ch-c)*np.sinh(Lambda*xAux[ix]/LE)+(s-sh)*np.cosh(Lambda*xAux[ix]/LE))/(2*(1-c*ch))
        vOmeFEM[ix,iOme]=Psi1*DesNodalesOme[2*UbiObs[ix],iOme]+Psi2*DesNodalesOme[2*UbiObs[ix]+1,iOme]+Psi3*DesNodalesOme[2*UbiObs[ix]+2,iOme]+Psi4*DesNodalesOme[2*UbiObs[ix]+3,iOme]                
        

    if iOme>0:
        DesNodalesOme[:,Nt-iOme]=np.conj(DesNodalesOme[:,iOme]) 
        vOmeFEM[:,Nt-iOme]=np.conj(vOmeFEM[:,iOme])

#   4.0 CALCULO DE LOS DESPLAZAMIENTOS NODALES EN EL DOMIINO DEL TIEMPO

#   4.1 Desplazamiento de los nodos

DesNodalesTie=np.zeros([2*(NEle+1),Nt],dtype=float)

for iDes in range(2*(NEle+1)):
    DesNodalesTie[iDes,:]=np.real(sp.ifft(DesNodalesOme[iDes,:])/Dt)
    DesNodalesTie[iDes,:]=DesNodalesTie[iDes,:]*np.exp(OmeIma*t)

#   4.2 Desplazamiento en los Nx puntos de la viga

vTieFEM=np.zeros([Nx,Nt],dtype=complex)

for ix in range(Nx):
    vTieFEM[ix,:]=np.real(sp.ifft(vOmeFEM[ix,:])/Dt)
    vTieFEM[ix,:]=vTieFEM[ix,:]*np.exp(OmeIma*t)


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


#   6.0 Respuesta analitica en el dominio del tiempo

vTieAna=np.zeros([Nx,Nt],dtype=float)

for ix in range(Nx):
    vTieAna[ix,:]=np.real(sp.ifft(vOmeAna[ix,:])/Dt)
    vTieAna[ix,:]=vTieAna[ix,:]*np.exp(OmeIma*t)


#   7.0 ANIMACION (NO ES VIDEO)

tMaxFig=20
DitAni=1

vMax=np.max(np.max(np.abs(vTieAna)))
NtAni=np.ceil(tMaxFig/(DitAni*Dt))
itAni=np.linspace(0,DitAni*(NtAni-1),NtAni,dtype=int)

fig=plt.figure(2,figsize=(10,4)) 

for it in itAni:
    plt.clf()
    plt.plot(x,vTieFEM[:,it],'k',label='FEM',linewidth=2)
    plt.plot(x,vTieAna[:,it],color='r',linestyle='--',label='Analitico',linewidth=2)
    plt.plot(xNodos,DesNodalesTie[0:2*NEle+1:2,it],'ob',label='Nodos')
    plt.legend(loc='upper left')    
    plt.title('t='+str(t[it])+' s',fontsize=20)
    plt.ylim([-vMax,vMax])
    plt.pause(0.000005)
    
plt.show()
