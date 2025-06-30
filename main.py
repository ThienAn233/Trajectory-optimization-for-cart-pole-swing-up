import numpy as np
import casadi as ca

# Define the time step and total time
T = 2.0  # total time (s)
N = 10   # number of time steps
dt= T / N  # time step size (s)

# Define the cart-pole system parameters
m_cart = 1.0  # mass of the cart (kg)
m_pole = 0.3  # mass of the pole (kg)
l_pole = 0.5  # length of the pole (m)
gr     = 9.81  # acceleration due to gravity (m/s^2) 

# Define the state and control variables
x = ca.SX.sym('x',N)  # position of the cart
theta = ca.SX.sym('theta',N)  # angle of the pole
v = ca.SX.sym('v',N)  # velocity of the cart
omega = ca.SX.sym('omega',N)  # angular velocity of the pole
u = ca.SX.sym('u',N)  # force applied to the cart

# Define the state vector
states = ca.horzcat(x, theta, v, omega)
controls = ca.horzcat(u)
# Stack the state and control variables
vars = ca.horzcat(states, controls)
n_states = states.size1()
n_controls = controls.size1()

# Define the dynamics of the system
def dynamics(s, u):
    x, theta, v, omega = ca.horzsplit(s)
    dx = v
    dtheta = omega
    dv = (u + m_pole * l_pole * omega**2 * ca.sin(theta) + m_pole * gr * ca.cos(theta) * ca.sin(theta)) / (m_cart + m_pole*(1 - ca.cos(theta)**2))
    domega = (-u * ca.cos(theta) - m_pole * l_pole * omega**2 * ca.sin(theta) *ca.cos(theta) - (m_cart + m_pole) * gr * ca.sin(theta)) / (l_pole * (m_cart + m_pole*(1 - ca.cos(theta)**2)))
    return ca.horzcat(dx, dtheta, dv, domega)

# Define the bounds for the state and control variables
d_max = 2  # maximum displacement of the cart (m)
u_max = 20 # maximum force applied to the cart (N)
lbx = ca.repmat(ca.SX([-d_max,-ca.inf,-ca.inf,-ca.inf,-u_max]),N)
ubx = ca.repmat(ca.SX([ d_max, ca.inf, ca.inf, ca.inf, u_max]),N)

# Equality constraints
g = []
for i in range(N-1):
    # Apply trapezoid colocation for the dynamics constraints
    g.append(states[i+1,:] - states[i,:] - (dt) * (dynamics(states[i,:], controls[i,:]) + dynamics(states[i+1,:], controls[i+1,:])) / 2)
g.append(states[0,0:4]  - ca.SX([[0,0,0,0]]))
g.append(states[-1,0:4] - ca.SX([[d_max,np.pi,0,0]]))

# objective function for minmal control effort
f = 0
for i in range(N-1):
    f += ca.mtimes(controls[i,:].T, controls[i,:])  # minimize the control effort
 

# Create the optimization problem
nlp = {'x': vars, 'f': 0, 'g': g}
sol = ca.nlpsol()