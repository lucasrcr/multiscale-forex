import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.integrate import cumtrapz,trapezoid,quad
from scipy.stats import rv_continuous,kstest,gamma,kurtosis
from scipy.optimize import curve_fit,minimize
import mpmath as mp
import scipy.special as special

'--------------------------------Tratamento inicial dos dados-------------------------------------------------'
'Importando os dados'

df = pd.read_csv('EURUSD601.csv')
df = pd.DataFrame(df)
df = df.iloc[:, 4]
df = np.log(df)
df = np.diff(df)
# print(df.std())
# print(len(df))
'Fazendo média zero e desvio padrão 1'

dados_reshape = df.reshape(-1, 1)
scaler = StandardScaler()
scaled_data = scaler.fit_transform(dados_reshape)
dados = scaled_data.flatten()
df2 = pd.DataFrame({'x': dados})

# Sem fazer a média e o desvio padrão 1    - Caso não faça a transformação linear, no eixo x vai aparecer realmente o valor que foi a variação. O Forex varia muito pouco, lá pela terceira casa decimal. Vou deixar com o reshape para ficar parecido com outros gráficos que já vi.

'Estimador de variância'

def mov_var(M):
    variance = df2['x'].rolling(window=M).var(ddof=0).dropna().reset_index(drop=True)
    return variance
# def mov_var2(M):
#     variance = df2['x'].rolling(window=M).var(ddof=1).dropna().reset_index(drop=True)
#     return variance
'----------------------------Buscando fit do melhor N a distribuição de background----------------------------------------------'

# M = 6
# N = 1
# guess=[0.1]
#
# # Histograma
# hist2, bins2 = np.histogram(mov_var(M), bins=256, density=True)
#
# # Calcular os pontos médios dos intervalos
# meio_bins2 = (bins2[1:] + bins2[:-1]) / 2
# #Eliminando valores muito pequenos
#
# # Criar a máscara booleana. Botar maior do 10^-3 já elimina os zeros.
# mascara = hist2 > 15e-4
# new_hist2 = hist2[mascara]
# new_meio_bins2 = meio_bins2[mascara]
#
# # Usando e0 como a média apenas dos dados filtrados
#
# # e0 = np.trapz(new_meio_bins2*new_hist2, new_meio_bins2)
# e0 = 1
#
#
# # print(e0.mean(),np.trapz(new_meio_bins2*new_hist2, new_meio_bins2))
#
#
# # # Sem restrições
# # new_hist2 = hist2
# # new_meio_bins2 = meio_bins2
#
# # print(new_hist2)
#
# # # removendo o primeiro termo
# #
# new_hist2_fit = new_hist2[1:]
# new_meio_bins2_fit = new_meio_bins2[1:]
#
# # # com o primeiro termo
# #
# # new_hist2_fit = new_hist2
# # new_meio_bins2_fit = new_meio_bins2
#
#
#
#
# # Plotar apenas os pontos onde a máscara é verdadeira
# tamanho_anel2 = 10
# plt.scatter(new_meio_bins2, new_hist2, s=tamanho_anel2, facecolors='none', edgecolors='orange')
#
# mp.pretty = True
# mp.dps = 8
# def f_min(x):
#     # N=1
#     A=[]
#     D=[]
#     x1=x[0] #beta1
#     # e0 = mov_var(M).mean() #epsilon_0
#     for i in range(0,N):
#         D.append(-x1-1)
#     # print(D)
#     for i,xg in enumerate(new_meio_bins2_fit):
#         c1=1/(np.power(special.gamma(x1+1),N))
#         c2=1/(e0*np.power(x1,N))
#         func=mp.meijerg([D,[]],[[],[]],(xg/(e0*np.power(x1,N))),maxprec=2000,maxterms=2000)
#         y1=c1*c2*func
#         # residual=(y1-new_hist2_fit[i])**2
#         residual=abs(mp.log10(y1)-mp.log10(new_hist2_fit[i]))
#
#         # print(residual)
#         A.append(residual)
#     return np.array(A).sum()
# # As vezes quando aparece um erro de cannot convert mpc to float é porque o guess está muito distante
# res=minimize(f_min,guess,method='Nelder-Mead')
# print(res)
# print(res.x)
# print(res.x[0])
# #o valor fun que o minimize exibe é o valor da função no ponto que ela achou como mínimo. Como minha função f_min já é a soma dos quadrados das diferenças, ao comparar duas funções, a melhor é a que tiver o menor valor para fun.
# print(f_min(res.x))
#
# # Gráfico da minha função G nos valores obtidos pelo fit
# def ig_meijer(x):
#     # N=1
#     beta = res.x[0]
#     # e0 = mov_var(M).mean() #epsilon_0
#     D=[]
#     for i in range(0,N):
#         D.append(-beta-1)
#     c1=1/(np.power(special.gamma(beta+1),N))
#     c2=1/(e0*np.power(beta,N))
#     func=mp.meijerg([D,[]],[[],[]],(x/(e0*np.power(beta,N))),maxprec=2000,maxterms=2000)
#     y1=c1*c2*func
#     return y1
# ig_meijervec = np.vectorize(ig_meijer)
#
# xp = np.linspace(new_meio_bins2[0],new_meio_bins2[-1],1000)
# plt.yscale('log')
# plt.xscale('log')
#
# plt.plot(xp,ig_meijervec(xp))
# plt.title('Distribuição de epsilon. Gama Inversa. Janela M=6. N=' + str(N))
# plt.text(5, 0.3,
#          r'$\beta=$' + str(round(res.x[0],2)),
#          {'color': 'black', 'fontsize': 10})
# plt.text(5, 0.2,
#          r'$func=$' + str(round(f_min(res.x),4)),
#          {'color': 'black', 'fontsize': 10})
#
# plt.show()

'----------------------------------Fit Direto do Sinal-----------------------------------------'

# M = 6
# N = 2
# guess=[1.1]
#
# e0 = 1
# # histograma do sinal
# hist, bins = np.histogram(df2, bins=200, density=True)
#
# # Calcular os pontos médios dos intervalos
# meio_bins = (bins[1:] + bins[:-1]) / 2
# # Buscando o erro comparando com o sinal apenas nos pontos. Para isso, temos que eliminar os zeros existentes em hist.
#
# mascara2 = hist > 6e-4
# new_hist = hist[mascara2]
# new_meio_bins = meio_bins[mascara2]
#
# def f_min_sinal(x):
#     # N=1
#     A=[]
#     D=[]
#     x1=x[0] #beta1
#
#     # e0 = mov_var(M).mean() #epsilon_0
#     for i in range(0,N):
#         D.append(-x1-1/2)
#     # print(D)
#     for i,xg in enumerate(new_meio_bins):
#         c1=1/(np.power(special.gamma(x1+1),N))
#         c2=1/(np.sqrt(2*np.pi*e0*np.power(x1,N)))
#         func=mp.meijerg([D,[]],[[0],[]],(xg**2/(2*e0*np.power(x1,N))),maxprec=5000,maxterms=10000)
#         y1=c1*c2*func
#         # print(y1,new_hist[i])
#         # residual=abs(mp.log10(y1)-mp.log10(new_hist[i]))
#         residual=(y1-new_hist[i])**2
#
#         # print(residual)
#         A.append(residual)
#     return np.array(A).sum()
#
# res=minimize(f_min_sinal,guess,method='Nelder-Mead')
# print(res)
# print(res.x)
# print(res.x[0])
#
#
# print(f_min_sinal(res.x))
# # print(ig_meijervec_sinal(0.04))
#
# def ig_meijer_sinal(x):
#     # N=1
#     beta = res.x[0]
#     # e0 = mov_var(8).mean() #epsilon_0
#     # e0 = mov_var(M).mean() #epsilon_0
#
#     D=[]
#     for i in range(0,N):
#         D.append(-beta-1/2)
#     c1=1/(np.power(special.gamma(beta+1),N))
#     c2=1/(np.sqrt(2*np.pi*e0*np.power(beta,N)))
#     func=mp.meijerg([D,[]],[[0],[]],(x**2/(2*e0*np.power(beta,N))),maxprec=5000,maxterms=10000)
#     y1=c1*c2*func
#     return y1
# ig_meijervec_sinal = np.vectorize(ig_meijer_sinal)
#
# tamanho_anel = 10
# cor_anel = 'orange'
# # Plotar os pontos usando plt.plot()
# plt.yscale('log')
#
# plt.scatter(new_meio_bins, new_hist, s=tamanho_anel, facecolors='none', edgecolors=cor_anel)
# xp = np.linspace(new_meio_bins[0],new_meio_bins[-1],200)
# plt.plot(xp,ig_meijervec_sinal(xp))
#
#
#
#
# plt.xlabel('x')
# plt.ylabel('Retornos (diferença dos Ln)')
# plt.title('Distribuição do Retornos do par EURUSD - M=6. N=' + str(N))
#
#
# plt.text(5, 0.4,
#          r'$\beta=$' + str(round(res.x[0],2)),
#          {'color': 'black', 'fontsize': 10})
# plt.text(5, 0.1,
#          r'$func=$' + str(round(f_min_sinal(res.x)
# ,5)),
#          {'color': 'black', 'fontsize': 10})
#
# # pode ser legal olhar o log log do sinal. Os valores negativos não vao entrar, mas nos ajudar a ver pelo menos o comportamento de uma cauda.
# # plt.xscale('log')
# plt.show()
'--------------------------------Fits conjunto-----------------------------------------'

M = 6
N = 2
guess=[1.1]

# Histograma
hist2, bins2 = np.histogram(mov_var(M), bins=256, density=True)

# Calcular os pontos médios dos intervalos
meio_bins2 = (bins2[1:] + bins2[:-1]) / 2
#Eliminando valores muito pequenos

# Criar a máscara booleana. Botar maior do 10^-3 já elimina os zeros.
mascara = hist2 > 15e-4
new_hist2 = hist2[mascara]
new_meio_bins2 = meio_bins2[mascara]

# Usando e0 como a média apenas dos dados filtrados

# e0 = np.trapz(new_meio_bins2*new_hist2, new_meio_bins2)
e0 = 1


# print(e0.mean(),np.trapz(new_meio_bins2*new_hist2, new_meio_bins2))


# # Sem restrições
# new_hist2 = hist2
# new_meio_bins2 = meio_bins2

# print(new_hist2)

# # removendo o primeiro termo
#
new_hist2_fit = new_hist2[1:]
new_meio_bins2_fit = new_meio_bins2[1:]

# # com o primeiro termo
#
# new_hist2_fit = new_hist2
# new_meio_bins2_fit = new_meio_bins2




# Plotar apenas os pontos onde a máscara é verdadeira
tamanho_anel2 = 10
plt.scatter(new_meio_bins2, new_hist2, s=tamanho_anel2, facecolors='none', edgecolors='orange')


# histograma do sinal
hist, bins = np.histogram(df2, bins=200, density=True)

# Calcular os pontos médios dos intervalos
meio_bins = (bins[1:] + bins[:-1]) / 2
# Buscando o erro comparando com o sinal apenas nos pontos. Para isso, temos que eliminar os zeros existentes em hist.


# restringindo
mascara2 = hist > 6e-4
new_hist = hist[mascara2]
new_meio_bins = meio_bins[mascara2]



# # sem restrições
# new_hist = hist
# new_meio_bins = meio_bins



mp.pretty = True
mp.dps = 8
def f_min(x):
    # N=1
    A=[]
    D=[]
    x1=x[0] #beta1
    # e0 = mov_var(M).mean() #epsilon_0
    for i in range(0,N):
        D.append(-x1-1)
    # print(D)
    for i,xg in enumerate(new_meio_bins2_fit):
        c1=1/(np.power(special.gamma(x1+1),N))
        c2=1/(e0*np.power(x1,N))
        func=mp.meijerg([D,[]],[[],[]],(xg/(e0*np.power(x1,N))),maxprec=2000,maxterms=2000)
        y1=c1*c2*func
        residual=(y1-new_hist2_fit[i])**2
        # residual=abs(mp.log10(y1)-mp.log10(new_hist2_fit[i]))

        # print(residual)
        A.append(residual)

    D1=[]

    # e0 = mov_var(M).mean() #epsilon_0
    for i in range(0,N):
        D1.append(-x1-1/2)
    # print(D)
    for i,xg in enumerate(new_meio_bins):
        c1=1/(np.power(special.gamma(x1+1),N))
        c2=1/(np.sqrt(2*np.pi*e0*np.power(x1,N)))
        func=mp.meijerg([D1,[]],[[0],[]],(xg**2/(2*e0*np.power(x1,N))),maxprec=5000,maxterms=10000)
        y1=c1*c2*func
        # print(y1,new_hist[i])
        # residual=abs(mp.log10(y1)-mp.log10(new_hist[i]))
        residual=(y1-new_hist[i])**2
        # print(residual)
        A.append(residual)
    return np.array(A).sum()
# As vezes quando aparece um erro de cannot convert mpc to float é porque o guess está muito distante
res=minimize(f_min,guess,method='Nelder-Mead')
print(res)
print(res.x)
print(res.x[0])
#o valor fun que o minimize exibe é o valor da função no ponto que ela achou como mínimo. Como minha função f_min já é a soma dos quadrados das diferenças, ao comparar duas funções, a melhor é a que tiver o menor valor para fun.
print(f_min(res.x))

# Gráfico da minha função G nos valores obtidos pelo fit
def ig_meijer(x):
    # N=1
    beta = res.x[0]
    # e0 = mov_var(M).mean() #epsilon_0
    D=[]
    for i in range(0,N):
        D.append(-beta-1)
    c1=1/(np.power(special.gamma(beta+1),N))
    c2=1/(e0*np.power(beta,N))
    func=mp.meijerg([D,[]],[[],[]],(x/(e0*np.power(beta,N))),maxprec=2000,maxterms=2000)
    y1=c1*c2*func
    return y1
ig_meijervec = np.vectorize(ig_meijer)

xp = np.linspace(new_meio_bins2[0],new_meio_bins2[-1],1000)
plt.yscale('log')
plt.xscale('log')

plt.plot(xp,ig_meijervec(xp))
plt.title('Distribuição de epsilon. Gama Inversa. Janela M=6. N=' + str(N))
plt.text(5, 0.3,
         r'$\beta=$' + str(round(res.x[0],2)),
         {'color': 'black', 'fontsize': 10})
plt.text(5, 0.2,
         r'$func=$' + str(round(f_min(res.x),4)),
         {'color': 'black', 'fontsize': 10})

plt.show()


def ig_meijer_sinal(x):
    # N=1
    beta = res.x[0]
    # e0 = mov_var(8).mean() #epsilon_0
    # e0 = mov_var(M).mean() #epsilon_0

    D=[]
    for i in range(0,N):
        D.append(-beta-1/2)
    c1=1/(np.power(special.gamma(beta+1),N))
    c2=1/(np.sqrt(2*np.pi*e0*np.power(beta,N)))
    func=mp.meijerg([D,[]],[[0],[]],(x**2/(2*e0*np.power(beta,N))),maxprec=5000,maxterms=10000)
    y1=c1*c2*func
    return y1
ig_meijervec_sinal = np.vectorize(ig_meijer_sinal)

tamanho_anel = 10
cor_anel = 'orange'
# Plotar os pontos usando plt.plot()
plt.yscale('log')

plt.scatter(new_meio_bins, new_hist, s=tamanho_anel, facecolors='none', edgecolors=cor_anel)
xp = np.linspace(new_meio_bins[0],new_meio_bins[-1],200)
plt.plot(xp,ig_meijervec_sinal(xp))

plt.xlabel('x')
plt.ylabel('Retornos (diferença dos Ln)')
plt.title('Distribuição do Retornos do par EURUSD - M=6. N=' + str(N))


plt.text(5, 0.4,
         r'$\beta=$' + str(round(res.x[0],2)),
         {'color': 'black', 'fontsize': 10})
plt.text(5, 0.1,
         r'$func=$' + str(round(f_min(res.x)
,5)),
         {'color': 'black', 'fontsize': 10})

# pode ser legal olhar o log log do sinal. Os valores negativos não vao entrar, mas nos ajudar a ver pelo menos o comportamento de uma cauda.
# plt.xscale('log')
plt.show()
