import numpy as np
from numpy.linalg import inv
from scipy.integrate import solve_ivp

# from scipy.integrate.odepack import odeint
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

import base64
from io import BytesIO


L = 0.0014  # motor inductance
Res = 0.5  # motor resistance
J = 0.003  # inertia momentum
b = 0.0002  # friction
k_ME = 0.1  # factor converting mechanical energy into electrical
k_EM = 0.1  # factor converting electrical energy into mechanical

# current limitation
CURRENT_LIMIT = 40

# SET SPEED IN RPM
# SPEED_RPM_SET = 2000
# OMEGA_SPEED = (SPEED_RPM_SET * 2 * np.pi) / 60

NUM_SAMPLES = 2000

Q = np.eye(2)
S = np.eye(2)
R = np.array([[1]])

# definnig time
t_eval = np.linspace(0, 10, NUM_SAMPLES)
t_ricc_eval = t_eval[::-1]

t = [0, 10]
t_ricc = t[::-1]  # reverset time for riccati equation

# X0 = np.array([[0, OMEGA_SPEED]]).T
U0 = 0


A = np.array([[-Res / L, -k_ME / L], [k_EM / J, -b / J]])
B = np.array([[1 / L], [0]])
C = np.array([0, 1])


def riccati(t, p):
    p = np.array([p]).T
    dP = np.reshape(p, [2, 2])
    ddP = -(dP @ A - dP @ B @ inv(R) @ B.T @ dP + A.T @ dP + Q)

    return np.ndarray.tolist(np.reshape(ddP, [1, -1])[0])


res = solve_ivp(riccati, t_ricc, [1, 0, 0, 1], t_eval=t_ricc_eval)
P11 = interp1d(res.t, res.y[0], fill_value="extrapolate")
P12 = interp1d(res.t, res.y[1], fill_value="extrapolate")
P21 = interp1d(res.t, res.y[2], fill_value="extrapolate")
P22 = interp1d(res.t, res.y[3], fill_value="extrapolate")


def model(t, x):
    x = np.array([x]).T

    u = 20

    dx = A @ x + B * u
    return [dx[0, 0], dx[1, 0]]


def model_ricc(t, x, omega_speed):
    x = np.array([x]).T
    P = np.array([[P11(t), P12(t)], [P21(t), P22(t)]])

    K = inv(R) @ B.T @ P
    X0 = np.array([[0, omega_speed]]).T
    nx = x - X0
    nx[1] = np.clip(nx[1], -CURRENT_LIMIT, CURRENT_LIMIT)

    nu = -K @ nx + U0

    dx = A @ nx + B * nu

    return np.ndarray.tolist(dx.T[0])


def graph(speed):
    omega_speed = (speed * 2 * np.pi) / 60
    rad_to_rpm = 9.549297
    # res_model = odeint(model, [0, 0], t)
    # ricc_model = odeint(model_ricc, [0, 0], t, args=(omega_speed,), full_output=0)
    res_model = solve_ivp(model, t, [0, 0], t_eval=t_eval)
    ricc_model = solve_ivp(model_ricc, t, [0, 0], args=(omega_speed,), t_eval=t_eval)

    print(f"OMEGA SET: {omega_speed}")

    fig, axs = plt.subplots(2, 2)
    fig.suptitle("DC motor simulation")
    fig.set_size_inches(20, 10)
    axs[0, 0].set_title("Current [A]")
    axs[0, 0].plot(res_model.t, res_model.y[0])
    axs[0, 0].grid(True)

    axs[0, 1].set_title("Rotational speed [rpm]")
    axs[0, 1].plot(res_model.t, res_model.y[1] * rad_to_rpm)
    axs[0, 1].grid(True)

    axs[1, 0].set_title("LQR current [A]")
    axs[1, 0].plot(ricc_model.t, ricc_model.y[0])
    axs[1, 0].grid(True)

    axs[1, 1].set_title(f"LQR rotational speed $\omega$ = {int(omega_speed * rad_to_rpm)}[RPM]")
    axs[1, 1].plot(ricc_model.t, ricc_model.y[1] * rad_to_rpm)
    axs[1, 1].grid(True)
    # Save it to a temporary buffer.
    buf = BytesIO()
    fig.savefig(buf, format="png")
    fig.savefig(f"lqr_{speed}.png")

    # Embed the result in the html output.
    data = base64.b64encode(buf.getbuffer()).decode("ascii")
    return data
