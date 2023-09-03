import numpy as np
import numpy.random as npr
from scipy.stats import norm
import tensorflow.keras as keras
import tensorflow.keras.backend as kb
import matplotlib.pyplot as plt
plt.style.use('ggplot')

# Black-Scholes formula to determine the price of a European call option
def BlackScholes(S0,r,sigma,T,K):
    # Calculate d1 and d2 terms
    d1 =  1 / (sigma * np.sqrt(T)) * (np.log(S0/K) + (r+sigma**2/2)*T)
    d2 = d1 - sigma * np.sqrt(T)
    # Return the call price using the Black-Scholes formula
    return norm.cdf(d1) * S0 - norm.cdf(d2) * K * np.exp(-r*T)

# Delta of a Black-Scholes European call option
def BlackScholesCallDelta(S0,r,sigma,T,K):
    d1 =  1 / (sigma * np.sqrt(T)) * (np.log(S0/K) + (r+sigma**2/2)*T)
    return norm.cdf(d1)

# Parameters for the stock price model
mu = 0.1      # drift
sigma = 0.5   # volatility
T = 100       # time steps
S_0 = 1       # initial stock price

# Simulate stock price paths using the geometric Brownian motion model

N = 100000    # number of sample paths
xi = npr.normal(0, np.sqrt(1 / T), (N, T))   # Gaussian random numbers
W = np.apply_along_axis(np.cumsum, 1, xi)    # Cumulative sum simulating the Brownian motion paths
W = np.concatenate((np.zeros((N, 1)), W),1)  # Append zero at the beginning
drift = np.linspace(0, mu , T + 1)           # Linearly spaced drift values
drift = np.reshape(drift, (1, T + 1))
drift = np.repeat(drift, N, axis=0) 
S = S_0 * np.exp(drift + sigma * W)          # Compute stock price paths

# Calculate differences in stock prices for use later
dS = np.diff(S, 1, 1)

# For each time step, create an array with time and corresponding stock price value
tim = np.linspace(0, 1, T+1)
X = []
for i in range(T):
    timv = np.repeat(tim[i],N)
    timv = np.reshape(timv,(N,1))
    Sv = np.reshape(S[:,i],(N,1))
    X.append(np.concatenate((timv,Sv),1))

# Plotting the stock price paths for the first three simulations
plt.plot(tim,S[0],label="$i=0$")
plt.plot(tim,S[1],label="$i=1$")         
plt.plot(tim,S[2],label="$i=2$")
plt.xlabel(r"$\frac{t}{T}$")
plt.ylabel(r"$S^i_t$")
plt.legend()
plt.show()

# Constructing the neural network model

# Define the neural network layers
layer1 = keras.layers.Dense(100, activation='relu')
layer2 = keras.layers.Dense(100, activation='relu')
layer3 = keras.layers.Dense(100, activation='relu')
layer4 = keras.layers.Dense(1, activation='sigmoid')

# Build the neural network for each time step, sharing weights across time steps
inputs = []
predictions = []
for i in range(T):
    sinput = keras.layers.Input(shape=(2,))
    x = layer1(sinput)
    x = layer2(x)
    x = layer3(x)
    sprediction = layer4(x)
    inputs.append(sinput)
    predictions.append(sprediction)

# Concatenate the predictions from each time step
predictions = keras.layers.Concatenate(axis=-1)(predictions)
model = keras.models.Model(inputs=inputs, outputs=predictions)
model.summary()

# Define custom loss function for the option pricing problem
K = 1
callprice = BlackScholes(S_0, 0, sigma, 1, K)
def loss_call(y_true,y_pred):
    return (callprice + kb.sum(y_pred * y_true,axis=-1) - kb.maximum(S_0 + kb.sum(y_true,axis=-1) - K,0.))**2

# Compile and train the neural network
epochs = 4
model.compile(optimizer='adam', loss=loss_call, metrics=[])
model.fit(X,dS,batch_size=100,epochs=epochs)

# Evaluate the learned hedging strategy by comparing it to the Black-Scholes delta
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
plt.plot(Sval, Delta_BS, "b--", label=r"$\frac{\partial}{\partial S}\mathrm{BS}(S_t,K,1-\
