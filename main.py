import numpy as np
import casadi as ca

# Simulation parameters
T = 2.0      # Total time (s)
N = 30       # Number of time steps
dt = T / N   # Time step

# Cart-pole parameters
m_cart = 1.0
m_pole = 0.3
l_pole = 0.5
gr = 9.81

# Create symbolic variables for each time step
x_list, theta_list, v_list, omega_list, u_list = [], [], [], [], []

for i in range(N):
    x_list.append(ca.SX.sym(f'x_{i}'))
    theta_list.append(ca.SX.sym(f'theta_{i}'))
    v_list.append(ca.SX.sym(f'v_{i}'))
    omega_list.append(ca.SX.sym(f'omega_{i}'))
    u_list.append(ca.SX.sym(f'u_{i}'))

# Stack all variables into a single optimization vector
X = ca.vertcat(*x_list, *theta_list, *v_list, *omega_list, *u_list)

# Helpers to extract state/control at a given timestep
def get_state(i):
    return ca.vertcat(x_list[i], theta_list[i], v_list[i], omega_list[i])

def get_control(i):
    return u_list[i]

# Dynamics function
def dynamics(s, u):
    x, theta, v, omega = s[0], s[1], s[2], s[3]
    dx = v
    dtheta = omega
    denom = m_cart + m_pole * (1 - ca.cos(theta)**2)
    dv = (u + m_pole * l_pole * omega**2 * ca.sin(theta) + m_pole * gr * ca.cos(theta) * ca.sin(theta)) / denom
    domega = (-u * ca.cos(theta) - m_pole * l_pole * omega**2 * ca.sin(theta) * ca.cos(theta) - (m_cart + m_pole) * gr * ca.sin(theta)) / (l_pole * denom)
    return ca.vertcat(dx, dtheta, dv, domega)

# Dynamics and boundary constraints
g = []

for i in range(N - 1):
    s_i = get_state(i)
    s_ip1 = get_state(i + 1)
    u_i = get_control(i)
    u_ip1 = get_control(i + 1)
    f_i = dynamics(s_i, u_i)
    f_ip1 = dynamics(s_ip1, u_ip1)
    g.append(s_ip1 - s_i - (dt/2)*(f_i + f_ip1))
# Initial and final conditions
g.append(get_state(0) - ca.vertcat(0, 0, 0, 0))               # Start at rest at 0
g.append(get_state(N - 1) - ca.vertcat(2, np.pi, 0, 0))       # End at rest at position 2, pole upright
# Objective: minimize total control effort
f = 0
for i in range(N - 1):
    u_i = get_control(i)
    u_ip1 = get_control(i + 1)
    f += 0.5 * dt * (u_i**2 + u_ip1**2)

# Stack all constraints
G = ca.vertcat(*g)
# Define NLP
nlp = {'x': X, 'f': f, 'g': G}

# Create solver
solver = ca.nlpsol("S", "ipopt", nlp)

# Bounds on variables
d_max = 2
u_max = 20
lbx = [-d_max]*N + [-ca.inf]*N + [-ca.inf]*N + [-ca.inf]*N + [-u_max]*N
ubx = [ d_max]*N + [ ca.inf]*N + [ ca.inf]*N + [ ca.inf]*N + [ u_max]*N

# Bounds on constraints (equality constraints: g == 0)
lbg = [0]*((N+1)*4) #+ [ca.vertcat(0, 0, 0, 0)]*2
ubg = [0]*((N+1)*4) #+ [ca.vertcat(0, 0, 0, 0)]*2

# Initial guess for optimization variables
x0 = []
for i in range(N):
    x0 += [d_max*i/N]        # Linearly increasing cart position
for i in range(N):
    x0 += [np.pi*i/N]        # Linearly increasing pole angle
for _ in range(2*N):
    x0 += [0]            # Zero for  v, omega
for _ in range(N):
    x0 += [0]            # Zero control input


# Solve
sol = solver(x0=x0,lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)

# Extract solution
x_opt = sol['x'].full().flatten()
x_vals = x_opt[0:N]
theta_vals = x_opt[N:2*N]
v_vals = x_opt[2*N:3*N]
omega_vals = x_opt[3*N:4*N]
u_vals = x_opt[4*N:5*N]

plot = False
sim  = True

if sim:
    from env import simulate_cart_pole
    config = {
        "m1": m_cart,  # Cart mass
        "m2": m_pole,  # Pole mass
        "l": l_pole,   # Pole length
    }
    simulate_cart_pole(r=int(1/dt),config=config,controls=u_vals)



if plot:
    # (Optional) Plot results
    import matplotlib.pyplot as plt
    t = np.linspace(0, T, N)
    plt.figure(figsize=(12, 8))
    plt.subplot(3, 1, 1)
    plt.plot(t, x_vals,label='x (cart position)')
    plt.scatter(t, x_vals, color='red', s=50, marker='o')
    plt.scatter(t, x_vals, color='white', s=20, marker='o')
    plt.axhline(2, color='gray', linestyle='--', label='target x=2')
    plt.axhline(-2, color='gray', linestyle='--', label='target x=-2')
    plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    plt.ylabel("position [m]")

    plt.subplot(3, 1, 2)
    plt.plot(t, theta_vals, label='theta (pole angle)')
    plt.scatter(t, theta_vals, color='red', s=50, marker='o')
    plt.scatter(t, theta_vals, color='white', s=20, marker='o')
    plt.axhline(np.pi, color='gray', linestyle='--', label='target θ=π')
    plt.axhline(-np.pi, color='gray', linestyle='--', label='target θ=π')
    plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    plt.ylabel("angle [rad]")

    plt.subplot(3, 1, 3)
    plt.plot(t, u_vals, label='u (control input)', color='black')
    plt.scatter(t, u_vals, color='purple', s=50, marker='o')
    plt.scatter(t, u_vals, color='white', s=20, marker='o')
    plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    plt.axhline(-20, color='gray', linestyle='--')
    plt.axhline(20, color='gray', linestyle='--')
    plt.xlabel("Time [s]")
    plt.ylabel("force [N]")

    plt.tight_layout()
    plt.show()
