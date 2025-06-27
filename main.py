import numpy as np
import casadi as ca


# Define the cart-pole system parameters
m_cart = 1.0  # mass of the cart (kg)
m_pole = 0.3  # mass of the pole (kg)
l_pole = 0.5  # length of the pole (m)
g = 9.81  # acceleration due to gravity (m/s^2) 

# Define the state and control variables
x = ca.SX.sym('x')  # position of the cart
theta = ca.SX.sym('theta')  # angle of the pole
v = ca.SX.sym('v')  # velocity of the cart
omega = ca.SX.sym('omega')  # angular velocity of the pole
u = ca.SX.sym('u')  # force applied to the cart

# Define the state vector
states = ca.vertcat(x, theta, v, omega)
controls = ca.vertcat(u)
n_states = states.size1()
n_controls = controls.size1()

# Define the dynamics of the system
def dynamics(s, u):
    x, theta, v, omega = ca.vertsplit(s)
    dx = v
    dtheta = omega
    dv = (u + m_pole * l_pole * omega**2 * ca.sin(theta) + m_pole * g * ca.cos(theta) * ca.sin(theta)) / (m_cart + m_pole*(1 - ca.cos(theta)**2))
    domega = (-u * ca.cos(theta) - m_pole * l_pole * omega**2 * ca.sin(theta) *ca.cos(theta) - (m_cart + m_pole) * g * ca.sin(theta)) / (l_pole * (m_cart + m_pole*(1 - ca.cos(theta)**2)))
    return ca.vertcat(dx, dtheta, dv, domega)

# Define the time step and total time
T = 2.0  # total time (s)
N = 10   # number of time steps
dt= T / N  # time step size (s)

# Define the bounds for the state and control variables
d_max = 2  # maximum displacement of the cart (m)
u_max = 20 # maximum force applied to the cart (N)
lbw = [-d_max]*n_states
ubw = [d_max]*n_states
lbu = [-u_max]*n_controls
ubu = [u_max]*n_controls

# Apply trapezoidal colocation for the dynamics constraints
w0 = ca.MX.sym('w', n_states,)
u0 = ca.MX.sym('u', n_controls)
g = []
for i in range(N):
    g.append(w0[i+1] - w0[i] - (dt) * (dynamics(w0[i], u0[i])+ dynamics(w0[i+1], u0[i+1])) / 2)

# Bounds on the initial and final states
w0
g = ca.vertcat(*g)

# 

# Create the optimization problem
nlp = {'x': ca.vertcat(w0, u0), 'f': 0, 'g': g}