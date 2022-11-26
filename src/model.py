import numpy as np
from scipy.integrate.odepack import odeint
import matplotlib.pyplot as plt

L = 0.0014    # motor inductance
R = 0.5       #motor resistance
J = 0.003     #inertia momentum
b = 0.0002    #friction
k_ME = 0.1    #factor converting mechanical energy into electrical 
k_EM = 0.1    #factor converting electrical energy into mechanical

def force(t):
    return 24 

A = np.array([[-R/L,-k_ME/L],[k_EM/J,-b/J]])
B = np.array([[1/L],[0]])
C = np.array([0,1])
def model(x,t):
    dx = A @ np.array([[x[0]],[x[1]]]) +B * force(t)
    y = C @ np.array([[x[0]], [x[1]]])

    return [dx[0,0],dx[1,0]]
#definnig time 
t = np.linspace(0, 10, 2000)
res = odeint(model,[0,0],t)

plt.figure()
plt.title("DC motor simulation")
plt.grid()
plt.subplot(1, 2, 1)
plt.plot(t, res[:, 0])
plt.subplot(1, 2, 2)
plt.plot(t, res[:, 1])
plt.show()