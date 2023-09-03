import numpy as np
import numpy.random as npr
from scipy.stats import norm
import tensorflow.keras as keras
import tensorflow.keras.backend as kb
import matplotlib.pyplot as plt
plt.style.use('ggplot')

def BlackScholes(S0,r,sigma,T,K):
    d1 =  1 / (sigma * np.sqrt(T)) * (np.log(S0/K) + (r+sigma**2/2)*T)
    d2 = d1 - sigma * np.sqrt(T)
    return norm.cdf(d1) * S0 - norm.cdf(d2) * K * np.exp(-r*T)
#callprice = BlackScholes(S0,0,sigma,1,K)
def BlackScholesCallDelta(S0,r,sigma,T,K):
    d1 =  1 / (sigma * np.sqrt(T)) * (np.log(S0/K) + (r+sigma**2/2)*T)
    return norm.cdf(d1)

mu = 0.1
sigma = 0.5
T = 100
S_0 = 1

N = 100000
xi = npr.normal(0, np.sqrt(1 / T), (N, T))
W = np.apply_along_axis(np.cumsum, 1, xi)
W = np.concatenate((np.zeros((N, 1)), W),1)
drift = np.linspace(0, mu , T + 1)
drift = np.reshape(drift, (1, T + 1))
drift = np.repeat(drift, N, axis=0)
S = S_0 * np.exp(drift + sigma * W)

dS = np.diff(S, 1, 1)

tim = np.linspace(0, 1, T+1)
X = []
for i in range(T):
    timv = np.repeat(tim[i],N)
    timv = np.reshape(timv,(N,1))
    Sv = np.reshape(S[:,i],(N,1))
    X.append(np.concatenate((timv,Sv),1))

plt.plot(tim,S[0],label="$i=0$")
plt.plot(tim,S[1],label="$i=1$")         
plt.plot(tim,S[2],label="$i=2$")
plt.xlabel(r"$\frac{t}{T}$")
plt.ylabel(r"$S^i_t$")
plt.legend()
plt.show()

inputs = []
predictions = []

layer1 = keras.layers.Dense(100, activation='relu')
layer2 = keras.layers.Dense(100, activation='relu')
layer3 = keras.layers.Dense(100, activation='relu')
layer4 = keras.layers.Dense(1, activation='sigmoid')

for i in range(T):
    sinput = keras.layers.Input(shape=(2,))
    x = layer1(sinput)
    x = layer2(x)
    x = layer3(x)
    sprediction = layer4(x)
    inputs.append(sinput)
    predictions.append(sprediction)
    
predictions = keras.layers.Concatenate(axis=-1)(predictions)
model = keras.models.Model(inputs=inputs, outputs=predictions)
model.summary()

K = 1
callprice = BlackScholes(S_0, 0, sigma, 1, K)
def loss_call(y_true,y_pred):
    return (callprice + kb.sum(y_pred * y_true,axis=-1) - kb.maximum(S_0 + kb.sum(y_true,axis=-1) - K,0.))**2

epochs = 4
model.compile(optimizer='adam', loss=loss_call, metrics=[])
model.fit(X,dS,batch_size=100,epochs=epochs)

t = 0.7
tStest = []
Sval = np.linspace(0,2,num=T)
for i in range(T):
    z = (t,Sval[i])
    z = np.reshape(z,(1,2))
    tStest.append(z)

Delta_learn = np.reshape(model.predict(tStest), (T,))
Delta_BS = BlackScholesCallDelta(Sval, 0, sigma, 1-t, K)
plt.plot(Sval, Delta_learn, label=r"$f(\frac{t}{T},S_{t})$")
plt.plot(Sval, Delta_BS, "b--", label=r"$\frac{\partial}{\partial S}\mathrm{BS}(S_t,K,1-\frac{t}{T})$")
plt.xlabel(r"$S_t$ (spot price)")
plt.ylabel(r"$\gamma_t$ (hedge ratio)")
plt.title(r'$\frac{t}{T}=$%1.2f' % t, loc='left', fontsize=11)
plt.title(r'$K=$%1.2f' % K, loc='right', fontsize=11)
plt.legend()
plt.show()
