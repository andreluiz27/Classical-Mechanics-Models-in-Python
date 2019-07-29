#!/usr/bin/env python
# coding: utf-8

# # Projeto 2

# ## Precessão do periélio de Mercúrio
# 
# As leis de Kepler foram desenvolvidas com base em medidas (extremamente acuradas) realizadas a olho nu. Quando equipamentos mais precisos para medição de posição e tempo foram desenvolvidos, ficou claro que a trajetória de Mercúrio não era como esperado uma elipse, mas sim uma elipse com um deslocamento contínuo em seu periélio.
# 
# O deslocamento pode ser medido como de 566 *arco-segundos por século*. Posteriormente se descobriu que uma boa parte desse deslocamento é devida à influência dos outros planetas do sistema solar (além do Sol). Os cálculos prevêm o deslocamento de 523 arco-segundos por século. Resta então explicar aproximadamente 43 arco-segundos por século do deslocamento.
# 
# ---

# A explicação somente surgiu com o desenvolvimento da teoria da relatividade geral, que altera a lei da gravitação. Considerando o fator mais significativo, a força de gravitação sobre Mercúrio fica adaptada para incluir fator relativístico da seguinte forma (escrita para a interação entre o Sol e Mercúrio):
# 
# $$ F_G = \frac{GM_SM_M}{r^2}\left(1 + \frac{\alpha}{r^2}\right),$$
# 
# onde $\alpha\approx1.1\cdot10^{-8}\,\mathrm{UA}^2$ (para Mercúrio), $GM_S=4\pi^2\,\mathrm{UA}^3/\mathrm{ano}^2$ e a força é na direção radial para dentro (em direção ao Sol). Projetando a força nas direções cartesianas considerando o Sol na origem e Mercúrio no ponto $(x, y)$:
# 
# \begin{eqnarray}
# \frac{d^2x}{dt^2} & = & -\frac{GM_S}{r^3}\left(1 + \frac{\alpha}{r^2}\right)x\\
# \frac{d^2y}{dt^2} & = & -\frac{GM_S}{r^3}\left(1 + \frac{\alpha}{r^2}\right)y\\
# \end{eqnarray}
# 
# onde $r = \sqrt{x^2+y^2}.$
# 
# Para completar, usamos as seguintes condições iniciais com Mercúrio inicialmente no seu ponto mais afastado do Sol (veja seção 4.3 do livro *Computational Physics* de Giordano e Nakanishi para explicação das expressões):
# 
# \begin{eqnarray}
# x(0) & = & (1+e)a\\
# y(0) & = & 0\\
# v_x(0) & = & 0\\
# v_y(0) & = & \sqrt{\frac{GM_S(1-e)}{a(1+e)}},
# \end{eqnarray}
# 
# onde para Mercúrio $a \approx 0.39\,\mathrm{UA}$ (eixo maior da elipse) e $e\approx0.206$ (excentricidade).
# 
# ---

# O projeto consiste em avaliar **a contribuição do fator relativístico** para a precessão do periélio de Mercúrio.
# 
# Você deve proceder da seguinte forma:
# 1. Implemente código para integrar a trajetória de Mercúrio usando as equações e condições iniciais acima. Use distâncias em $\mathrm{UA}$ e tempos em anos.
# 1. Simule por aproximadamente 10 períodos de revolução para $\alpha=0$ e plote um gráfico da trajetória. Este é o *caso não-relativístico,* e a trajetória deve ser uma elipse perfeita com o Sol em um dos focos. Escolha um tamanho de passo de tempo suficientemente preciso (a curva traçada deve ser suave). Lembre-se de que o período de rotação de Mercúrio é de aproximadamente 88 dias.
# 1. Repita a simulação, mas agora com $\alpha=0.01$. Este é um caso com fator relativístico *grande*, o que significa que a precessão deve ser claramente visível na trajetória.
# 1. O valor real de $\alpha=1.1\cdot10^{-8}$ para Mercúrio é **extremamente baixo**, o que significa que seria necessário simular o sistema por um grande intervalo de tempo para permitir avaliar a taxa de precessão. Ao invés disso, vamos seguir um outro método: Vamos avaliar a taxa de precessão para diversos valores de $\alpha$. Veremos que ela varia *linearmente* com $\alpha$. Avaliaremos então o coeficiente dessa dependência linear e calcularemos a taxa de precessão usando o $\alpha$ conhecido de Mercúrio, conforme descrito nos passos abaixo.
#   1. Primeiro vamos fazer um experimento realizando a simulação do sistema com $\alpha=0.001$ por pelo menos 20 períodos de rotação.
#   1. Precisamos definir um método para marcar um ponto específico da elipse, para podermos avaliar como ele está se deslocando com o tempo devido à precessão. Para isso, usaremos o ponto mais afastado do Sol. Esse ponto pode ser determinado pelo seguinte raciocínio: O ponto mais afastado é aquele para o qual antes de chegar nele as distâncias vão aumentando, e depois que ele passa as distâncias vão diminuindo. Dizendo de outra forma, ele é um ponto onde a derivada da distância até o Sol é zero, e *a derivada é positiva antes dele e negativa depois dele.* Ao invés de usar a distância, usaremos a distância ao quadrado (pois o ponto de máximo é o mesmo). $$\frac{d}{dt}r^2 = \frac{d}{dt}(x^2 + y^2) = 2\left(x\frac{dx}{dt} + y\frac{dy}{dt}\right).$$ O fator 2 pode ser ignorado para nossos propósitos (não afeta o sinal da derivada). Queremos então os valores de $t$ para os quais a expressão $$x v_x + y v_y$$ *passa de positiva para negativa.* Os valores de $x, y, v_x$ e $v_y$ são calculados durante a integração das equações diferenciais no passo anterior. Usando os valores retornados pela rotina de integração, calcule os valores da expressão acima e encontre as coordenadas $(x,y)$ onde ele passa de positivo (ou zero) para negativo (ou zero); para nossos propósitos, *basta guardar um desses pontos* (o imediatamente anterior ou o imediatamente posterior ao máximo) como o ponto de transição; isso deve ser feito para cada uma das transições de máximo existentes na trajetória calculada. Para esses pontos precisamos agora encontrar os valores dos ângulos entre o eixo $x$ (positivo) e o raio da posição de Mercúrio (dica: Esse ângulo pode ser calculado usando `numpy.arctan2`, veja documentação). O valor desse ângulo é o que nos interessa para determinar como a elipse está precessionando com o tempo. Guarde o valor do ângulo $\theta$ de máxima distância e o instante de tempo associado ($t$ quando o máximo foi atingido).
#   1. Plote um gráfico com os ângulos de máximo contra o tempo e verifique que eles formam uma reta (aproximadamente) que passa pela origem.
#   1. Use a função `curve_fit` do módulo `scipy.optimize` para fazer um ajuste da função $\theta = \rho t$ a esses pontos, encontrando o melhor valor de $\rho$ de acordo com os dados (a função `curve_fit` irá retornar, entre outros, o valor desejado de $\rho$, que é a taxa de precessão para o $\alpha$ usado na simulação).
#   1. Agora, repita os passos A a D acima para 20 valores de $\alpha$ igualmente espaçados entre 0 e 0.002, excluindo o zero, encontrando e guardando os valores de $\rho$ para cada um dos $\alpha$.
#   1. Plote o gráfico de $\rho$ versus $\alpha$ e verifique que ele é uma linha reta que passa pela origem.
#   1. Use novamente `curve_fit`, mas agora para ajustar $\rho = c \alpha$ e econtrar o melhor valor de $c$.
#   1. Usando o valor de $c$ encontrado acima e o valor de $\alpha$ conhecido para Mercúrio (ítem 4), calcule a taxa de precessão de Mercúrio devida a efeitos relativísticos e confira com o valor esperado de 43 arco-segundos por século.

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from scipy.integrate import ode


# In[2]:


ex=0.205
a=0.39
G=6.67408e-11
#Ms=1.989e30
G_Ms=39.48 #4pi²
ALPHA=0 # será necessário ? 


# In[3]:


def deriv(t,xyuv,G_Ms,alpha): 
    x, y,u, v = xyuv
    R=np.sqrt(x**2+y**2)
    return [u, v, -G_Ms*x*(1+alpha/R**2)/R**3,-G_Ms*y*(1+alpha/R**2)/R**3]


# In[4]:


#valores iniciais
x0=(1+ex)*a
y0=0
vx0=0
vy0=np.sqrt(G_Ms*(1-ex)/(a*(1+ex)))
iniciais=[x0,y0,vx0,vy0]


# In[5]:


def integrador(func_derivadas,cond_iniciais,G_Ms,alpha,tempo_em_anos):
    t0=0
    dt=0.0001 #Assim como feito em Giordano
    r=ode(func_derivadas)
    r.set_initial_value(iniciais,t0)
    r.set_f_params(G_Ms,alpha)
    t2=[t0]
    xyuv_t=[iniciais] #lista para guardar os estados
    while r.successful() and t2[-1]<tempo_em_anos:
        new_t =r.t+ dt
        t2.append(new_t)
        new_xyuv=r.integrate(new_t)
        xyuv_t.append(new_xyuv)

    xyuv_t = np.array(xyuv_t) # Converte para array para indexação
    xs = xyuv_t[:,0]
    ys = xyuv_t[:,1]  
    vx = xyuv_t[:,2]
    vy = xyuv_t[:,3]
    
    pos_t=xs,ys,vx,vy,t2
    return pos_t


# In[6]:


pos=integrador(deriv,iniciais,G_Ms,0,2.41096) #10 revoluções = 880 dias = 2.41096 anos(terrestres)
xs=pos[0]
ys=pos[1]
vx=pos[2]
vy=pos[3]
temp=pos[4]


# In[7]:


plt.plot(xs,ys)
plt.xlabel("x (AU)",fontsize=20)
plt.ylabel("y (AU)",fontsize=20)
plt.show


# Vemos acima um gráfico em forma de uma elipse, como esperado pela órbita de mercúrio, com alpha = 0

# In[8]:


pos2=integrador(deriv,iniciais,G_Ms,0.01,2.41096)
xs2=pos2[0]
ys2=pos2[1]


# In[9]:


plt.plot(xs2,ys2)
plt.xlabel("x (UA)",fontsize=20)
plt.ylabel("y (UA)", fontsize=20)
plt.show


# Vemos claramente a precessão do periélio, sua grande expressividade vem do fato de usarmos um alpha relativamente grande (0.01) em relação ao alpha real de mercúrio!

# In[10]:


pos3=integrador(deriv,iniciais,G_Ms,0.0001,4.82192) # o tempo agora é pra 20 revoluções, repare que o alpha agora é 0.0001
xs3=pos3[0]
ys3=pos3[1]
vx3=pos3[2]
vy3=pos3[3]


# In[11]:


plt.plot(xs3,ys3)
plt.xlabel("x (UA)", fontsize=20)
plt.ylabel("y (UA)", fontsize=20)
plt.show


# Vemos uma certa espessura na órbita de mercúrio, infere-se ser devido a precessão do periélio, vemos claramente a dificuldade de detectar um grande efeito qualitativo quando usamos um alpha pequeno na ordem de 0.0001

# In[12]:


def zero_func(func_de_derivadas,alpha): #Encontras os zeros das variações radiais, isto é, os máximos da órbita
  
    estados=integrador(func_de_derivadas,iniciais,G_Ms,alpha,4.82192)
    xs=estados[0]
    ys=estados[1]
    vxs=estados[2]
    vys=estados[3]
    tempos=estados[4]
    maximos=[]
    for i in range(1,len(xs)):
        var1 = xs[i]*vxs[i]+ys[i]*vys[i]  #Iéssimo valor da expressão das derivadas descrita no texto introdutório desse notebook
        var2 =  xs[i-1]*vxs[i-1]+ys[i-1]*vys[i-1] #(Iéssimo - 1) valor da expressão das derivadas
        
        if (var1*var2 <=0): # Verifica a mudança de sinal
            maxs=(xs[i],ys[i],tempos[i]) 
            maximos.append(maxs) #Guarda os valores de máximos
    
    return maximos


# In[13]:


from scipy.optimize import curve_fit


# In[14]:


def f(x,a,b):  # Descreve uma função linear para o curve_fit
    return a*x + b
 


# In[15]:


mxs = zero_func(deriv,0.001)
mxs=np.array(mxs) #Converte array para indexação
x_m=mxs[:,0]
y_m=mxs[:,1]
t_m=mxs[:,2]


# Ao executar célula abaixo, veremos que haverá dois tipos de máximos, os para x>0 e x<0. Entretanto, precisamos apenas de um desses pontos, escolhe-se então os pontos para x<0, para isso basta fazer a cópia do mesmo vetor porém em passos de 2 a partir do primeiro valor negativo. Mesma lógica se mantém para "y_m" e "t_m".
# 

# In[16]:


x_m 


# In[17]:



x_m=x_m[1::2] #Realiza o processo descrito no texto acima
y_m=y_m[1::2]
t_m=t_m[1::2]
theta_1=np.arctan2(y_m,x_m) #Encontra-se finalmente os ângulos


# In[18]:


x_m #Repare que agora apenas os valores negativos estão no array


# In[19]:


plt.scatter(t_m,theta_1)
plt.xlabel("t (anos)", fontsize=20)
plt.ylabel(r'$\theta$ (radianos)', fontsize=20)


# Como era de se esperar, uma evolução linear do ângulo em função do tempos, passando pela origem. 

# In[20]:


#parametros,pc=curve_fit(f,t_m,theta_1) #


# In[21]:


#parametros[1]*57.29577951


# In[22]:


alphas=np.linspace(0,0.002,20)   
vetor=[]  #Perdão pela falta de criatividade do nome da variável! É apenas uma variável pra armazenar os rhos e os alphas   
vetor_new=[] #Apenas um vetor auxiliar
for i in range(1,len(alphas)):  
    zeros = zero_func(deriv,alphas[i])
    zeros=np.array(zeros)

    x_max=zeros[:,0]
    y_max=zeros[:,1]
    t_max=zeros[:,2]

    x_max=x_max[1::2]
    y_max=y_max[1::2]
    t_max=t_max[1::2]

    theta=np.arctan2(y_max,x_max)

    popt,pcov=curve_fit(f,t_max,theta) #Acha o rho para um determinado alpha


    vetor_new=(alphas[i],popt[0]) 
    vetor.append(vetor_new) #Armazena o iéssimo alpha e seu valor correspondenye de rho


# In[23]:


vetor=np.array(vetor)


# In[ ]:





# In[24]:


x_data=vetor[:,0] #extrai os alphas
y_data=vetor[:,1] #extrai os rhos

#x_data e y_data são apenas nomes genéricos para, não confundir com as coordenadas espaciais (x,y)!


# In[ ]:





# In[25]:


plt.scatter(x_data,y_data)
plt.axis([0,0.0025,0,0.40])
plt.xlabel(r'$\alpha$', fontsize=40)
plt.ylabel(r'$\rho$', fontsize=40)
plt.show


# Como era de se esperar, vemos um gráfico linear  passando pela origem

# In[26]:


popt_new,whatever=curve_fit(f,x_data,y_data) #Encontra-se agora o valor da constante linear entre rho e alpha


# In[27]:


convertido=np.rad2deg(popt_new[0]) #Converte-se para radianos


# In[28]:


convertido=convertido*3600 #Converte-se para arcsegundos


# In[29]:


def mercury_precession(c): #Calcular a taxa de precessão por século por alpha, o fator de 100 é para converter para séculos
    alpha_mercury=1.1e-8
    return c*alpha_mercury*100


# In[30]:


mercury_precession(convertido)


# Valor encontrado arredondando é 44 arcsegundos por século. O valor encontrado na literatura é de 43 arcsegundos por século.

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




