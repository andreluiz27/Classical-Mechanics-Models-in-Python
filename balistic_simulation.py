#!/usr/bin/env python
# coding: utf-8

# # Fatores adicionais na trajetória de um projétil
# O estudo da trajetória de um projétil em um campo gravitacional uniforme é uma das primeiras atividades desenvolvidas em cursos de física. Entretanto, nas soluções normalmente apresentadas o modelo recebe diversas simplificações para permitir sua solução analítica.
# ### O caso simples

# Vamos começar com o caso simples em que o campo gravitacional é uniforme, o projétil é pontual e não há nenhum tipo de força externa ou dissipação de energia.
# 
# Consideramos um sistema positivo de coordenadas $(x, y, z)$ orientado de tal forma que o eixo $z$ está na vertical, com valores crescentes para cima, o plano $z=0$ coincide com o solo e o eixo de $x$ é orientado na direção horizontal do lançamento do projétil. O projétil é lançado inicialmente ($t=0$) da posição $(0, 0, h)$ com velocidade inicial $(v_{x,0}, 0, v_{z,0})$. Queremos então calcular a trajetória do projétil $(x(t), y(t), z(t))$ até que ele retorne para o solo.
# 
# Se o projétil tem massa $m=1$ e o campo gravitacional apresenta uma aceleração uniforme $-g$ na direção do eixo $z$, as equações de movimento são expressas de forma simple:
# 
# \begin{eqnarray}
# \frac{d^2x}{dt^2} & = & 0\\
# \frac{d^2y}{dt^2} & = & 0\\
# \frac{d^2z}{dt^2} & = & -g
# \end{eqnarray}
# 
# Como essas equações diferenciais são lineares, considerando as condições iniciais dadas, o cálculo da trajetória fica simples:
# 
# \begin{eqnarray}
# x(t) & = & v_{x,0} t \\
# y(t) & = & 0\\
# z(t) & = & h + v_{z,0} t - \frac{1}{2} g t^2
# \end{eqnarray}
# 
# Isto é: o projétil permanece no plano $y=0$, se desloca uniformemente na direção de $x$ crescente e é uniformemente acelerado em $z$. Essas equações são válidas até que o projétil atinja o solo. Isso ocorre quando $z=0$, e portanto podemos encontrar o maior valor de tempo de interesse $T$ resolvendo a equação:
# 
# $$ h + v_{z,0} T - \frac{1}{2} g T^2 = 0$$
# 
# que resulta na solução:
# 
# $$ T = \frac{v_{z,0}+\sqrt{v_{z,0}^2+2 h g}}{g}$$
# 
# Com essas informações, podemos plotar as componentes da trajetória. Primeiro importamos os módulos necessários e fazemos com que os gráficos fiquem embutidos.


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')




# Função que calcula uma trajetória para o caso simples acima.
# Recebe os parâmetros:
#   h (altura do lançamento)
#   v0 (vetor da velocidade inicial)
#   g (aceleração da gravidade)
#   nintervalos (número de intervalos de tempo até atingir o solo)
# Retorna:
#   um array com os instantes de tempo
#   um array 2D em que cada linha tem o vetor de posição em um dos instantes de tempo
def trajetoria_simples(h, v0, g, nintervalos):
    # Extrai as componentes de velocidade
    vx0, vy0, vz0 = v0
    
    # Intervalo de tempo a usar
    T = (vz0 + np.sqrt(vz0**2 + 2 * g * h))/g
    t = np.linspace(0, T, nintervalos + 1)
    
    # Trajetória
    trajetoria = np.zeros((t.size, 3))
    trajetoria[:, 0] = vx0 * t
    trajetoria[:, 1] = np.zeros_like(t)
    trajetoria[:, 2] = h + vz0 * t - 0.5 * g * t**2
    
    return t, trajetoria

# Parâmetros:
h = 0.00001
v_x0 = 20.0
v_y0 = 0
v_z0 =np.sqrt(700**2-v_x0**2)
g = 9.81

# Calcula uma trajetória
t, trajetoria = trajetoria_simples(h, (v_x0, v_y0, v_z0), g, 2)
# Extraimos as componentes  x, y e z para facilitar nos gráficos
x = trajetoria[:, 0]
y = trajetoria[:, 1]
z = trajetoria[:, 2]

# Plota x(t), y(t) e z(t)
fig, axarray = plt.subplots(3, 1, sharex=True)
fig.set_size_inches(9, 9)
axarray[0].plot(t, x)
axarray[0].set_ylabel('x')
axarray[1].plot(t, y)
axarray[1].set_ylabel('y')
axarray[2].plot(t, z)
axarray[2].set_ylabel('z')
axarray[2].set_xlabel('t')
plt.show()

# Mais interessante é plotar a trajetória no espaço tridimensional. Para isso, usamos o módulo `mplot3d`.
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot(x, y, z)
plt.show()

# Neste caso, como a trajetória é bidimensional (restrita ao plano $y=0$) faz mais sentido plotar apenas o plano $x, z$.

plt.plot(x, z)
plt.xlabel('x')
plt.ylabel('z')
plt.show()

# O próximo passo é simularmos o sistema representado pelas equações diferenciais para comparar o resultado com o analítico apresentado acima. Para isso, como sempre, precisamos converter o sistema de equações em um sistema de equações de primeiro grau, usando o truque tradicional de incluir variáveis para as velocidades. Assim, além das posições $x, y, z$, teremos também as velocidades $u, v, w$, respectivamente, e o conjunto de equações fica:
# 
# \begin{eqnarray}
# \frac{dx}{dt} & = & u \\
# \frac{dy}{dt} & = & v \\
# \frac{dz}{dt} & = & w \\
# \frac{du}{dt} & = & 0 \\
# \frac{dv}{dt} & = & 0 \\
# \frac{dw}{dt} & = & -g \\
# \end{eqnarray}
# 
# E agora as condições iniciais são dadas por $(0, 0, h, v_{x,0}, 0, v_{z,0})$.
# 
# Com isso podemos usar o `odeint` do SciPy para resolver numericamente o sistema de equações.

from scipy.integrate import odeint

# Função que calcula as derivadas para o estado xyzuvw no instante t
def deriv_ideal(xyzuvw, t, g):
    x, y, z, u, v, w = xyzuvw
    return [u, v, w, 0, 0, -g]


# Usamos os mesmos parâmetros e o mesmo t anteriores
iniciais = [0, 0, h, v_x0, v_y0, v_z0]
xyzuvw_t = odeint(deriv_ideal, iniciais, t, args=(g,))
# Agora extraimos as componentes x, y e z (para faciliar o gráfico)
xs = xyzuvw_t[:, 0]
ys = xyzuvw_t[:, 1]
zs = xyzuvw_t[:, 2]

# Agora podemos plotar as duas trajetórias simultaneamente para verificar se há diferença.
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot(x, y, z, label='Analítica')
ax.plot(xs, ys, zs, label='Simulada')
ax.legend()
plt.show()


# Ou no plano $(x, z)$:

plt.plot(x, z, label='Analitica')
plt.plot(xs, zs, label='Simulada')
plt.xlabel('x')
plt.ylabel('z')
plt.legend()

plt.show()
# Vemos que as duas trajetórias estão superpostas, como esperado.
# 
# Infelizmente, no código acima fizemos uma trapaça: Usamos o intervalo de tempo calculado considerando a solução analítica, **que não estaria disponível se precisássemos realizar simulações!**
# 
# Duas formas de lidar com esse problema são:
# - Simular por um tempo suficientemente grande, depois descartar os valores em que $z<0$.
# - Simular um passo por vez e terminar quando chegamos em $z\le 0$.
# 
# A primeira solução tem a vantagem de porder usar `odeint` e é viável quando temos razão para acreditar que sabemos uma boa aproximação inicial para $T$. Se não temos uma aproximação para $T$, então devemos usar a segunda solução, que infelizmente não pode fazer uso de `odeint`, mas precisamos usar um objeto da classe `ode` e realizar a integração passo a passo, como exemplificado no código abaixo.

# Função de derivadas para a classe ode.
# Note a inversão da ordem dos parâmetros t e xyzuvw em relação à outra derivada!
def deriv_ideal_2(t, xyzuvw, g): 
    x, y, z, u, v, w = xyzuvw
    return [u, v, w, 0, 0, -g]


from scipy.integrate import ode

# Especificamos o tempo inicial e o intervalo entre instantes sucessivos
t0 = 0
Δt = 0.01

# Criamos um objeto da classe ode associado à função correta de derivadas
r = ode(deriv_ideal_2)

# Ajustamos os dados do sistema simulado nesse objeto
r.set_initial_value(iniciais, t0) # Indica as condições inicias e instante inicial
r.set_f_params(g) # Passa parâmetro adicinonal da função de derivadas

# Agora fazemos a simulação
t2 = [t0] # Cria uma lista para guardar os instantes de tempo
xyzuvw_t2 = [iniciais] # Cria uma lista para guardar estados
last_z = h # Posição z onde o projétil está atualmente
while r.successful() and last_z > 0: # r.successful() verifica que a integração deu certo
    new_t = r.t + Δt # Calcula o proximo instante (r.t é o último calculado)
    t2.append(new_t) # Adiciona novo instante de tempo na lista
    new_xyzuvw = r.integrate(new_t) # Calcula novo estado
    xyzuvw_t2.append(new_xyzuvw) # Adiciona na lista de estados
    last_z = new_xyzuvw[2] # Verifica o valor de z atual
    
# Extraimos os componentes x, y e z
xyzuvw_t2 = np.array(xyzuvw_t2) # Converte para array para indexação
xs2 = xyzuvw_t2[:, 0]
ys2 = xyzuvw_t2[:, 1]
zs2 = xyzuvw_t2[:, 2]


fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot(x, y, z, label='Analítica')
ax.plot(xs2, ys2, zs2, label='Simulada')
ax.legend()
plt.show()
# Por curiosidade, vejamos se os dois sistemas estão sendo simulados no mesmo intervalo de tempo:

print('O valor de T é', t[-1])
print('O sistema foi simulado até', t2[-1])
print('A diferença é de', t2[-1] - t[-1])
print('Compare isso com o incremento de tempo escolhido de', Δt)


# **Nota:** Todos os casos acima são estritamente bidimensionais, e poderiam ser simulados bidimensionalmente no plano $xz$. Entretanto, deixamos o caso geral para mostrar como lidar com ele tanto na simulação como nos gráficos.

# ### Trajetória de uma bala de canhão: Resistência do ar

# Vamos agora considerar a trajetória de uma bala de canhão. Como seu tamanho não é desprezível e sua velocidade é alta, o primeiro elemento a incluir é a resistência do ar. Considerando as altas velocidades envolvidas, o elemento quadrático da resistência do ar é dominante, e sua expressão é:
# 
# $$R = B v^2,$$
# 
# onde $R$ é o módulo da força de resistência do ar, $B$ é uma constante que depende do projétil e $v$ é o módulo da velocidade $v = \sqrt{v_x^2+v_y^2+v_z^2}$. Como a resistência do ar é contrária ao movimento, devemos projetar seus componentes em cada uma das direções de acordo com os componentes da velocidade em cada direção:
# 
# \begin{eqnarray}
# R_x & = & - B v v_x\\
# R_y & = & - B v v_y\\
# R_z & = & - B v v_z
# \end{eqnarray}


# Função de derivadas para a classe ode.
# Note a inversão da ordem dos parâmetros t e xyzuvw em relação à outra derivada!
def deriv_resistencia_analitica(xyzuvw_res,t, g): 
    x_res, y_res, z_res, u_res, v_res, w_res = xyzuvw_res
    mod_vel=np.sqrt(u_res**2+v_res**2+w_res**2)
    k=4e-5
    return [u_res, v_res, w_res,-k*mod_vel*u_res, -k*mod_vel*v_res, -g -k*mod_vel*w_res]

# Função de derivadas para a classe ode.
# Note a inversão da ordem dos parâmetros t e xyzuvw em relação à outra derivada!
def deriv_resistencia(t, xyzuvw_res, g): 
    x_res, y_res, z_res, u_res, v_res, w_res = xyzuvw_res
    mod_vel=np.sqrt(u_res**2+v_res**2+w_res**2)
    k=4e-5
    return [u_res, v_res, w_res,-k*mod_vel*u_res, -k*mod_vel*v_res, -g -k*mod_vel*w_res]

xyzuvw_t = odeint(deriv_resistencia_analitica, iniciais, t, args=(g,))
xs_an = xyzuvw_t[:, 0]
ys_an = xyzuvw_t[:, 1]
zs_an = xyzuvw_t[:, 2]

ang = np.zeros(91)
for i in range(91):
    aux = np.deg2rad(i)
    ang[i] = aux
print(ang[90])
k = 0.00004
g = 9.81

v0 = np.array([700])
vx = v0 * np.cos(ang)
vy = np.zeros(91)
vz = v0 * np.sin(ang)

# Criamos um objeto da classe ode associado à função correta de derivadas
r = ode(deriv_resistencia)

# Ajustamos os dados do sistema simulado nesse objeto
r.set_initial_value(iniciais, t0) # Indica as condições inicias e instante inicial
r.set_f_params(g) # Passa parâmetro adicinonal da função de derivadas

# Agora fazemos a simulação
t2 = [t0] # Cria uma lista para guardar os instantes de tempo
xyzuvw_t2 = [] # Cria uma lista para guardar estados
last_z = h # Posição z onde o projétil está atualmente
while r.successful() and last_z > 0: # r.successful() verifica que a integração deu certo
    new_t = r.t + Δt # Calcula o proximo instante (r.t é o último calculado)
    t2.append(new_t) # Adiciona novo instante de tempo na lista
    new_xyzuvw = r.integrate(new_t) # Calcula novo estado
    xyzuvw_t2.append(new_xyzuvw) # Adiciona na lista de estados
    last_z = new_xyzuvw[2] # Verifica o valor de z atual

# Extraimos os componentes x, y e z
xyzuvw_t2 = np.array(xyzuvw_t2) # Converte para array para indexação
xs3=np.zeros(9000)
xs3 = xyzuvw_t2[:, 0]
ys3 = xyzuvw_t2[:, 1]
zs3 = xyzuvw_t2[:, 2]

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot(xs2, ys2, zs2, label='Sem resistencia')
ax.plot(xs_an, ys_an, zs_an, label='Com resistencia analítica')
ax.plot(xs3, ys3, zs3, label='Com resistencia')
ax.legend()
plt.show()

#Diferença entre os máximos em x e em y

maxx_sem_res=xs2[len(xs2)-1]
maxx_com_res=xs3[len(xs3)-1]
maxz_sem_res=np.amax(zs2)
maxz_com_res=np.amax(zs3)
print(maxz_sem_res)
print(maxz_com_res)

difx=maxx_sem_res-maxx_com_res
difz=maxz_sem_res-maxz_com_res
print('A diferença no eixo x é',difx)
print('Porcentualmente significa', 100*difx/maxx_sem_res,'%')
print('A diferença no eixo z é',difz)
print('Porcentualmente significa', 100*difz/maxz_sem_res,'%')

plt.plot(xs2, zs2, label='Sem resistencia')
plt.plot(xs_an, zs_an, label='Com resistencia e analitica')
plt.plot(xs3, zs3, label='Com resistencia')

plt.xlabel('x')
plt.ylabel('z')
plt.legend()
plt.show()


v0 = np.array([700])
vx = v0 * np.cos(ang)
vy = np.zeros(91)
vz = v0 * np.sin(ang)
t0 = 0
Δt = 0.01
#ang0 = 30
vector=[]
#Agora vamos variar para vários thetas com objetivo de encontrarmos o valor de theta máximo

for i in range(0,91):

    r = ode(deriv_resistencia)

    cond_iniciais = [0,0,0,vx[i],vy[i],vz[i]]

    # Ajustamos os dados do sistema simulado nesse objeto
    r.set_initial_value(cond_iniciais, t0) # Indica as condições inicias e instante inicial
    r.set_f_params(g) # Passa parâmetro adicinonal da função de derivadas

    # Agora fazemos a simulação
    t2 = [t0] # Cria uma lista para guardar os instantes de tempo
    xyzuvw_t2 = [cond_iniciais] # Cria uma lista para guardar estados
    last_z = h # Posição z onde o projétil está atualmente
    while r.successful() and last_z > 0: # r.successful() verifica que a integração deu certo
        new_t = r.t + Δt # Calcula o proximo instante (r.t é o último calculado)
        t2.append(new_t) # Adiciona novo instante de tempo na lista
        new_xyzuvw = r.integrate(new_t) # Calcula novo estado
        xyzuvw_t2.append(new_xyzuvw) # Adiciona na lista de estados
        last_z = new_xyzuvw[2] # Verifica o valor de z atual

    # Extraimos os componentes x, y e z
    xyzuvw_t2 = np.array(xyzuvw_t2) # Converte para array para indexação
    #xs3=np.zeros(9000)
    xs3_aux = xyzuvw_t2[:, 0]
    ys3_aux = xyzuvw_t2[:, 1]
    zs3_aux = xyzuvw_t2[:, 2]
    
    vector.append(xs3_aux[-1])
    

maxx=np.amax(vector)
print(vector.index(maxx)-1,'graus')


# Podemos concluir das informações acima:
# 
# a) A simulação analítica não exclui o intervalo para Z negativos
# 
# b) A simulação com resistencia, na situação analítica e não-analítica coincidem graficamente
# 
# c) O caso com resistência afeta fortemente a distância máxima em x, com um porcentual de diferença de 57%
# 
# d) O caso com resistência afeta fortemente a distância máxima em z, com um porcentual de diferença de 45%
# 
# e) O valor theta que possui maior distância, no caso com resistência, é para 38 graus, diferente do caso sem resistência, do qual o maior valor é 45 graus
# 
# 

plt.plot(ys2, zs2, label='Com resistencia')
plt.plot(ys3, zs3, label='Sem resistencia')
plt.xlabel('y')
plt.ylabel('z')
plt.legend()
plt.show()


# ### Trajetória de uma bala de canhão: Variação da densidade do ar

# Você deve ter notado que o projétil chega a grandes altitudes antes de cair novamente no solo. Isso significa que existe a possibilidade de que variações na densidade do ar tenham efeitos importantes, principalmente considerando que o coeficiente $B$ depende da densidade do ar.
# 
# Para considerar isso, usaremos a aproximação adiabática para a densidade do ar (livro de Giordano e Nakanishi):
# 
# $$\rho = \rho_0 \left( 1 - \frac{a z}{T_0}\right)^\alpha,$$
# 
# onde $a \approx 6.5\cdot10^{-3}K/m$, $T_0$ é a temperatura ao nível do mar (em Kelvin), que faremos igual a 296.15, $\alpha\approx 2.5$ e $\rho_0$ é a densidade a nivel do mar (que é o nosso $z=0$). A resistência do ar é proporcional à densidade do ar, portanto para levar esse efeito em consideração basta substituirmos o módulo da resistência por:
# 
# $$\hat{R} = \frac{\rho}{\rho_0}R = \left( 1 - \frac{a z}{T_0}\right)^\alpha B v^2$$.
# 

def deriv_densidade(t, xyzuvw_dens, g): 
    x_dens, y_dens, z_dens, u_dens, v_dens, w_dens = xyzuvw_dens
    mod_vel=np.sqrt(u_dens**2+v_dens**2+w_dens**2)
    k=4e-5
    a=6.5e-3
    alpha=2.5
    T_0=296.15
    R=(1-a*z_dens/T_0)**alpha
    return [u_dens, v_dens, w_dens,-k*R*mod_vel*u_dens, -k*R*mod_vel*v_dens, -g -k*R*mod_vel*w_dens]


# Criamos um objeto da classe ode associado à função correta de derivadas

t0 = 0
Δt = 0.01

r = ode(deriv_densidade)

# Ajustamos os dados do sistema simulado nesse objeto
r.set_initial_value(iniciais, t0) # Indica as condições inicias e instante inicial
r.set_f_params(g) # Passa parâmetro adicinonal da função de derivadas

# Agora fazemos a simulação
t2 = [t0] # Cria uma lista para guardar os instantes de tempo
xyzuvw_t2 = [] # Cria uma lista para guardar estados
last_z = h # Posição z onde o projétil está atualmente
while r.successful() and last_z > 0: # r.successful() verifica que a integração deu certo
    new_t = r.t + Δt # Calcula o proximo instante (r.t é o último calculado)
    t2.append(new_t) # Adiciona novo instante de tempo na lista
    new_xyzuvw = r.integrate(new_t) # Calcula novo estado
    xyzuvw_t2.append(new_xyzuvw) # Adiciona na lista de estados
    last_z = new_xyzuvw[2] # Verifica o valor de z atual

# Extraimos os componentes x, y e z
xyzuvw_t2 = np.array(xyzuvw_t2) # Converte para array para indexação
xs4 = xyzuvw_t2[:, 0]
ys4 = xyzuvw_t2[:, 1]
zs4 = xyzuvw_t2[:, 2]


fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot(xs2, ys2, zs2, label='Sem resistencia')
ax.plot(xs3, ys3, zs3, label='Com resistencia')
ax.plot(xs4, ys4, zs4, label='Com densidade')

ax.legend()
plt.show()



plt.plot(xs2, zs2, label='Sem resistencia')
plt.plot(xs3, zs3, label='Com resistencia')
plt.plot(xs4, zs4, label='Com densidade')

plt.xlabel('x')
plt.ylabel('z')
plt.legend()
plt.show()


# O resultado é coerente pois sabe-se que fisicamente, como mostrado na fórmula, que a resistẽncia do ar diminui com a altura, logo espera-se que o alcance seja maior, tal fato pode ser visto no gráfico acima

# ### Trajetória de uma bola de baseball

# No livro de Giordano e Nakanishi, seção 2.3, existe uma discussão onde se mostra que o coeficiente de resistência do ar não é constante, mas depende da velocidade da bola, o que é importante considerando as velocidades típicas de bolas de baseball. Usaremos a expressão experimental da equação 2.6 desse livro, reproduzida abaixo, para calcular a trajetória da bola de baseball.
# 
# $$ \frac{B}{m} = 0.0039 + \frac{0.0058}{1 + \exp \left[(v - v_d)/\Delta\right]},$$
# 
# onde $v$ é o módulo da velocidade da bola, $v_d=35m/s$, $\Delta = 5m/s$ e tudo está em unidade SI.


def deriv_baseball(t,xyzuvw_base, g): 
    x_base, y_base, z_base, u_base, v_base, w_base = xyzuvw_base
    mod_vel=np.sqrt(u_base**2+v_base**2+w_base**2)
    fator_expo=np.exp((mod_vel-35)/5)
    b_m=0.0039 +0.0058/(1+fator_expo)
    return [u_base, v_base, w_base,-b_m*mod_vel*u_base, -b_m*mod_vel*v_base, -g -b_m*mod_vel*w_base]

def deriv_baseball_2(t,xyzuvw_base, g): 
    x_base, y_base, z_base, u_base, v_base, w_base = xyzuvw_base
    mod_vel=np.sqrt(u_base**2+v_base**2+w_base**2)
    fator_expo=np.exp((35)/5)
    b_m=0.0039 +0.0058/(1+fator_expo)
    return [u_base, v_base, w_base,-b_m*mod_vel*u_base, -b_m*mod_vel*v_base, -g -b_m*mod_vel*w_base]


iniciais = [0, 0, 1, 50*np.cos(np.deg2rad(35)), 0, 50*np.sin(np.deg2rad(35))]

r = ode(deriv_baseball)



# Ajustamos os dados do sistema simulado nesse objeto
r.set_initial_value(iniciais, t0) # Indica as condições inicias e instante inicial
r.set_f_params(g) # Passa parâmetro adicinonal da função de derivadas

# Agora fazemos a simulação
t2 = [t0] # Cria uma lista para guardar os instantes de tempo
xyzuvw_t2 = [] # Cria uma lista para guardar estados
last_z = h # Posição z onde o projétil está atualmente
while r.successful() and last_z > 0: # r.successful() verifica que a integração deu certo
    new_t = r.t + Δt # Calcula o proximo instante (r.t é o último calculado)
    t2.append(new_t) # Adiciona novo instante de tempo na lista
    new_xyzuvw = r.integrate(new_t) # Calcula novo estado
    xyzuvw_t2.append(new_xyzuvw) # Adiciona na lista de estados
    last_z = new_xyzuvw[2] # Verifica o valor de z atual

# Extraimos os componentes x, y e z
xyzuvw_t2 = np.array(xyzuvw_t2) # Converte para array para indexação
xs5 = xyzuvw_t2[:, 0]
ys5 = xyzuvw_t2[:, 1]
zs5 = xyzuvw_t2[:, 2]

r = ode(deriv_baseball_2)


# Ajustamos os dados do sistema simulado nesse objeto
r.set_initial_value(iniciais, t0) # Indica as condições inicias e instante inicial
r.set_f_params(g) # Passa parâmetro adicinonal da função de derivadas

# Agora fazemos a simulação
t2 = [t0] # Cria uma lista para guardar os instantes de tempo
xyzuvw_t2 = [] # Cria uma lista para guardar estados
last_z = h # Posição z onde o projétil está atualmente
while r.successful() and last_z > 0: # r.successful() verifica que a integração deu certo
    new_t = r.t + Δt # Calcula o proximo instante (r.t é o último calculado)
    t2.append(new_t) # Adiciona novo instante de tempo na lista
    new_xyzuvw = r.integrate(new_t) # Calcula novo estado
    xyzuvw_t2.append(new_xyzuvw) # Adiciona na lista de estados
    last_z = new_xyzuvw[2] # Verifica o valor de z atual

# Extraimos os componentes x, y e z
xyzuvw_t2 = np.array(xyzuvw_t2) # Converte para array para indexação
xs5_constante = xyzuvw_t2[:, 0]
ys5_constante = xyzuvw_t2[:, 1]
zs5_constante = xyzuvw_t2[:, 2]


plt.plot(xs5_constante, zs5_constante, label='Sem variação de b/m')
plt.plot(xs5, zs5, label='Com variação de b/m')

plt.xlabel('x')
plt.ylabel('z')
plt.legend()
plt.show()

