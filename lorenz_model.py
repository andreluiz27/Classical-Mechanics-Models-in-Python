#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from scipy.integrate import ode
from scipy import signal, misc
from mpl_toolkits.mplot3d import Axes3D

def deriv(t,xyz,sigma,R,b): 
    x,y,z, = xyz
    return [sigma*(y-x),-x*z+R*x-y,x*y-b*z]


R_conjunto=[1,7,15,20,35,50]
x0=0
y0=1
z0=0
iniciais=[x0,y0,z0]
def integrador(func_derivadas,cond_iniciais,sigma,R,b,t_max):
    t0=0
    dt=0.00025 #Assim como proposto
    r=ode(func_derivadas)
    r.set_initial_value(cond_iniciais,t0)
    r.set_f_params(sigma,R,b)
    t2=[t0]

    xyz_t=[cond_iniciais] #lista para guardar os estados
    while r.successful() and t2[-1]<t_max:
        new_t =r.t+ dt
        t2.append(new_t)
        new_xyz=r.integrate(new_t)
        xyz_t.append(new_xyz)

    xyz_t = np.array(xyz_t) # Converte para array para indexação
    t2 = np.array(t2) # Converte para array para indexação
    
    xs = xyz_t[:,0]
    ys = xyz_t[:,1]  
    zs = xyz_t[:,2]
    ts=t2[:]
    
    
    pos_t=xs,ys,zs,ts
    return pos_t

for i in R_conjunto:
   
    v=integrador(deriv,iniciais,10,i,8/3,40)
    x_data=v[0]
    y_data=v[1]
    z_data=v[2]
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(x_data, y_data,z_data, label="R= "+str(i)+"")
    #ax.plot(xs, ys, zs, label='Simulada')
    ax.legend()
    plt.show()

count=1
#Printando os gráficos em X(t)
for i in R_conjunto:
    V=integrador(deriv,iniciais,10,i,8/3,40)
    x_t=V[0]
    tempos=V[3]
    plt.plot(tempos,x_t,label="R="+str(i)+"")
    plt.xlabel("t",fontsize=20)
    plt.ylabel("X(t)",fontsize=20)
    plt.legend()
    plt.show()
    count=count+1

#Printando os gráficos em Y(t)    
for i in R_conjunto:
    V=integrador(deriv,iniciais,10,i,8/3,40)
    y_t=V[1]
    tempos=V[3]
    plt.plot(tempos,y_t,label="R="+str(i)+"")
    plt.xlabel("t",fontsize=20)
    plt.ylabel("Y(t)",fontsize=20)
    plt.legend()
    plt.show()
    
#Printando os gráficos em Z(t)    
for i in R_conjunto:
    V=integrador(deriv,iniciais,10,i,8/3,40)
    z_t=V[2]
    tempos=V[3]
    plt.plot(tempos,z_t,label="R="+str(i)+"")
    plt.xlabel("t",fontsize=20)
    plt.ylabel("Z(t)",fontsize=20)
    plt.legend()
    plt.show()

    
# X(t),Y(t),Z(t) em R<=20 apresentam um ruído de transiência seguida por uma oscilação que tende a diminuir sua amplitude até chegar em um valor de estabilidade. Para os gráficos com R>= 35 vemos um comportamento confuso, mostrando oscilações com picos. Podemos concluir que para algum valor entre R=20 e R=35 ocorre uma mudança entre o regime de estabilidade para o regime caótico, Giordano comenta que tal valor está em torno de 24. 

pertub=10e-6
iniciais_pertub=[0,1+pertub,0]

for i in R_conjunto:
    v=integrador(deriv,iniciais,10,i,8/3,40)
    v_pertub=integrador(deriv,iniciais_pertub,10,i,8/3,40)
    
    x_data=v[0]
    y_data=v[1]
    z_data=v[2]
   
    x_pertub_data=v_pertub[0]
    y_pertub_data=v_pertub[1]
    z_pertub_data=v_pertub[2]
   
    a=x_data-x_pertub_data
    b=y_data-y_pertub_data
    c=z_data-z_pertub_data
    tempo=v[3]
 
    fig=plt.figure()    
    ax=fig.gca(projection='3d')
    ax.plot(x_data, y_data,z_data, label="R="+str(i)+""  )
    ax.plot(x_pertub_data, y_pertub_data, z_pertub_data, label="Pertubação inicial "+str(pertub)+" R="+str(i)+"")
    ax.legend()
    plt.show()
    plt.show()

# Vemos que os gráficos deixam de se sobrepor no gráfico com R=35 em diante, concordando com os gráficos de X(t),Y(t),Z(t). Concordando também com o valor dito na literatura (R próximo de 24). Vemos também que mesmo com pequenas pertubações (10e-6) nas condições iniciais os gráficos começam apresentar trajetórioas diferentes, fenômeno muito comum em mapas logísticos. Podemos concluir que o sistema é sensível as condições iniciais para R>24 . 

def f(x,a,b):  # Descreve uma função linear para o curve_fit
    return np.exp(b*x)

from scipy.optimize import curve_fit

def log_grafico(R,pertubacao,t_inferior,t_superior):
    
    pertub=pertubacao
    iniciais_pertub=[0,1+pertub,0]
    v=integrador(deriv,iniciais,10,R,8/3,40)
    v_pertub=integrador(deriv,iniciais_pertub,10,R,8/3,40)
    
    x_data=v[0]
    y_data=v[1]
    z_data=v[2]

    x_pertub_data=v_pertub[0]
    y_pertub_data=v_pertub[1]
    z_pertub_data=v_pertub[2]

    a=x_data-x_pertub_data
    b=y_data-y_pertub_data
    c=z_data-z_pertub_data

    tempo=v[3] 
    dist=np.sqrt((a**2+b**2+c**2))

    dist2=[]
    tempo2=[]

    for k in range(len(tempo)):
        if(tempo[k]>t_inferior and tempo[k]<t_superior ):
            dist2.append(dist[k])
            tempo2.append(tempo[k])

        
    ax1=plt.semilogy(tempo2,dist2,label="R= "+str(R)+"")
    plt.legend()
    plt.show()
    return None

def lyap(R,pertubacao,t_inferior,t_superior):
    
    pertub=pertubacao
    iniciais_pertub=[0,1+pertub,0]
    v=integrador(deriv,iniciais,10,R,8/3,40)
    v_pertub=integrador(deriv,iniciais_pertub,10,R,8/3,40)
    
    x_data=v[0]
    y_data=v[1]
    z_data=v[2]

    x_pertub_data=v_pertub[0]
    y_pertub_data=v_pertub[1]
    z_pertub_data=v_pertub[2]

    a=x_data-x_pertub_data
    b=y_data-y_pertub_data
    c=z_data-z_pertub_data

    tempo=v[3] 
    
    dist=np.sqrt((a**2+b**2+c**2))

    dist2=[]
    tempo2=[]

    for k in range(len(tempo)):
        if(tempo[k]>t_inferior and tempo[k]<t_superior ):
            dist2.append(dist[k])
            tempo2.append(tempo[k])


    popt,pcov=curve_fit(f,tempo2,dist2)
    
    return popt[1]

log_grafico(1,10e-5,0,40)  #R=1
print(lyap(1,10e-5,0.5,2.5))

log_grafico(7,10e-4,0,40)  #R=7
print(lyap(7,10e-4,1,8))


log_grafico(15,10e-2,0,40) #R=15 
print(lyap(15,10e-2,5,30))

log_grafico(20,10e-3,0,40) #R=20
print(lyap(20,10e-3,5,40))

log_grafico(35,10e-12,0,40) #R=35
print(lyap(35,10e-12,5,10))

log_grafico(50,10e-6,0,40) #R=50
print(lyap(50,10e-6,1,4))

# Acima se encontra os gráficos das distâncias das trajetórias. Vemos que a distância apresenta por uma certa faixa de tempo um comportamento exponencial. Vemos que como discutido na literatura, o sinal do coeficiente de Lyapunov quando atinge R>R_caótico 
from scipy.signal import argrelmax

#Aqui encontro o valor do índice cujo o tempo é maior que 30.

indice_t_inferior=30/0.00025

vetor_maximos1=[]
vetor_maximos2=[]
vetor_maximos3=[]
count=0
for i in range(1,120):
    xmaxs=[]
    ymaxs=[]
    zmaxs=[]
    v=integrador(deriv,iniciais,10,i,8/3,80)
    
    xs=np.array(v[0])
    x_indices_max_aux = argrelmax(xs)
    x_indices_max=x_indices_max_aux[0]
    
    ys=np.array(v[1])
    y_indices_max_aux = argrelmax(ys)
    y_indices_max=y_indices_max_aux[0]
    
    zs=np.array(v[2])
    z_indices_max_aux = argrelmax(zs)
    z_indices_max=z_indices_max_aux[0]
    
    for j in x_indices_max:               #condição que só pegar o máximo se corresponder a t>30
        if j > indice_t_inferior:
            xmaxs.append(xs[j])
  
    for k in y_indices_max:
        if k > indice_t_inferior:
            ymaxs.append(ys[j])
  
    for l in z_indices_max:
        if l > indice_t_inferior:
            zmaxs.append(zs[j])
  
    
    
    vetor_maximos1.append(xmaxs)
    vetor_maximos2.append(ymaxs)
    vetor_maximos3.append(zmaxs)
    count=count+1
    print(count) #Contador apenas pra aocmpanhar as iterações

for i in range(1,119):
    tam=len(vetor_maximos1[i])
    plt.scatter(np.linspace(i,i,tam),vetor_maximos1[i],s=20)
plt.xlabel("R",fontsize=20)
plt.ylabel("X_maxs",fontsize=20)
plt.show
    

for i in range(1,119):
    tam=len(vetor_maximos2[i])
    plt.scatter(np.linspace(i,i,tam),vetor_maximos2[i],s=20)

plt.xlabel("R",fontsize=20)
plt.ylabel("Y_maxs",fontsize=20)
plt.show

for i in range(1,119):
    tam=len(vetor_maximos3[i])
    plt.scatter(np.linspace(i,i,tam),vetor_maximos3[i],s=20)

plt.xlabel("R",fontsize=20)
plt.ylabel("X_maxs",fontsize=20)
plt.show


# Vemos que nos gráficos ocorre uma bifurcação, como esperado.
