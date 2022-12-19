import numpy as np
from numpy.linalg import inv
from scipy.integrate.odepack import odeint
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

L = 0.0014  # motor inductance
Res = 0.5  # motor resistance
J = 0.003  # inertia momentum
b = 0.0002  # friction
k_ME = 0.1  # factor converting mechanical energy into electrical
k_EM = 0.1  # factor converting electrical energy into mechanical

# current limitation
CURRENT_LIMIT = 40

# SET SPEED IN RPM
SPEED_RPM_SET = 2000
OMEGA_SPEED = (SPEED_RPM_SET * 2 * np.pi) / 60


Q = np.eye(2)
S = np.eye(2)
R = np.array([[1]])

# definnig time
t = np.linspace(0, 10, 2000)
t_ricc = t[::-1]  # reverset time for riccati equation
# t_ricc = np.linspace(10, 0, 1000)

X0 = np.array([[0, OMEGA_SPEED]]).T
U0 = 0


A = np.array([[-Res / L, -k_ME / L], [k_EM / J, -b / J]])
B = np.array([[1 / L], [0]])
C = np.array([0, 1])


def riccati(p, t):
    dP = np.reshape(p, [2, 2])
    ddP = -(dP @ A - dP @ B @ inv(R) @ B.T @ dP + A.T @ dP + Q)

    return np.ndarray.tolist(np.reshape(ddP, [1, -1])[0])


res = odeint(riccati, [1, 0, 0, 1], t_ricc)
P11 = interp1d(t_ricc, res[:, 0], fill_value="extrapolate")
P12 = interp1d(t_ricc, res[:, 1], fill_value="extrapolate")
P21 = interp1d(t_ricc, res[:, 2], fill_value="extrapolate")
P22 = interp1d(t_ricc, res[:, 3], fill_value="extrapolate")


def model(x, t):
    x = np.array([x]).T

    u = 20

    dx = A @ x + B * u
    return [dx[0, 0], dx[1, 0]]


def model_ricc(x, t):
    x = np.array([x]).T
    P = np.array([[P11(t), P12(t)], [P21(t), P22(t)]])

    K = inv(R) @ B.T @ P

    nx = x - X0
    nx[1] = np.clip(nx[1], -CURRENT_LIMIT, CURRENT_LIMIT)

    nu = -K @ nx + U0

    dx = A @ nx + B * nu

    return np.ndarray.tolist(dx.T[0])


res_model = odeint(model, [0, 0], t)
ricc_model = odeint(model_ricc, [0, 0], t, full_output=0)

print(f"OMEGA SET: {OMEGA_SPEED}")

fig, axs = plt.subplots(2, 2)
fig.suptitle("DC motor simulation")
fig.set_size_inches(20, 10)
axs[0, 0].set_title("current")
axs[0, 0].plot(t, res_model[:, 0])

axs[0, 1].set_title("rotational speed")
axs[0, 1].plot(t, res_model[:, 1])

axs[1, 0].set_title("Ricc current")
axs[1, 0].plot(t, ricc_model[:, 0])

axs[1, 1].set_title(f"Ricc rotational speed $\omega$ = {OMEGA_SPEED:.2f}")
axs[1, 1].plot(t, ricc_model[:, 1])


plt.show()
