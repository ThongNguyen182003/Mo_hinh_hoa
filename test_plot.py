#Do not change these codes!
import numpy as np
from collections.abc import Callable

import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (12, 12)

Vector = np.array
Matrix = np.array
Function = Callable[[float, Vector], Vector]

#Explicit Euler Algorithm
def explicit_euler(f: Function, t: float, x: Vector, h: float) -> Vector:
    return x + h * f(t,x)


def solve(f: Function, x0: Vector, t0: float, tf: float, h: float) -> Matrix:
    xs = [x0]
    ts = [t0]
    x = x0
    t = t0
    while t < tf:
        x = explicit_euler(f, t, x, h)
        t += h
        xs.append(x)
        ts.append(t)
    return np.array(xs), np.array(ts)

# example 1 : Nonlinear case: Lotka-Volterra equations, which model the population dynamics of predator-prey systems
def example1(t:float, x:Vector) -> Vector:
    a, b, c , d = 2, 1 ,1.5 ,1
    dxdt = a*x[0] - b*x[0]*x[1]
    dydt = c*x[0]*x[1] - d*x[1]
    return np.array([dxdt, dydt])

# example 2 : Nonlinear case: the Lorenz system, which is a simplified model of atmospheric convection:
def example2(t:float, x:Vector) ->Vector:
    sigma , rho, beta = 10, 28 , 8/3
    dxdt = sigma * (x[1] - x[0])
    dydt = x[0]*(rho - x[2]) - x[1]
    dzdt = x[0]*x[1] - beta*x[2]
    return np.array([dxdt, dydt, dzdt])

#example3 : Nonlinear case: the Van der Pol oscillator, which models the oscillation in a circuit with a vacuum tube:
def example3(t : float , x : Vector) -> Vector:
    mu =  1
    dxdt = x[1]
    dydt = mu *(1 - x[0]**2) * x[1]-x[0]
    return np.array([dxdt, dydt])


#example4: Linear case: a decaying harmonic oscillator with damping coefficient gamma and angular frequency omega:
def example4(t : float, x: Vector) -> Vector:
    gama, omega = 0.5, 2
    dxdt = x[1]
    dydt = - gama * x[1] - omega**2*x[0]
    return np.array([dxdt, dydt])

#example5: Linear case: a system of linear equations with constant coefficients:

def example5(t: float, x: Vector) -> Vector:
    A = np.array([[1, -2], [2, 1]])
    b = np.array([1, 2])
    return A @ x + b

# plotting the solution of example 1
x0 = np.array([3, 2])  
t0, tf, h = 0, 10, 0.01 
xs, ts = solve(example1, x0, t0, tf, h)

# plot the solution
plt.plot(ts, xs[:,0], label='Prey')
plt.plot(ts, xs[:,1], label='Predator')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('Example 1')
plt.legend()
plt.show()

# plotting the sdolution of example 2
# plotting the solution of example 2
x0 = np.array([1, 1, 1])  
t0, tf, h = 0, 50, 0.01 
xs, ts = solve(example2, x0, t0, tf, h)


# plotting the solution of example 2
# define the initial conditions and time range
x0 = np.array([1, 1, 1])  
t0, tf, h = 0, 50, 0.01 

# solve the system using the explicit Euler method
xs, ts = solve(example2, x0, t0, tf, h)

# plot the solution
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(xs[:,0], xs[:,1], xs[:,2])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Example 2')
plt.show()

#plotting the solution of example 3
x0 = np.array([1, 1])  
t0, tf, h = 0, 10, 0.01 

# solving the differential equation using the Euler method
xs, ts = solve(example3, x0, t0, tf, h)

# plotting the solution
plt.plot(ts, xs[:,0], label='x')
plt.plot(ts, xs[:,1], label='y')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('Example 3')
plt.legend()
plt.show()

# plotting the solution of example 4
x0 = np.array([1, 0])  
t0, tf, h = 0, 10, 0.01 
xs, ts = solve(example4, x0, t0, tf, h)

# plot the solution
plt.plot(ts, xs[:,0], label='Position')
plt.plot(ts, xs[:,1], label='Velocity')
plt.xlabel('Time')
plt.title('Example 4')
plt.legend()
plt.show()

#plotting the solution of example 5
# initial conditions
x0 = np.array([1, 1])
t0, tf, h = 0, 10, 0.01 

# solve the system of equations using the Euler method
xs, ts = solve(example5, x0, t0, tf, h)

# plot the solution
plt.plot(ts, xs[:,0], label='x1')
plt.plot(ts, xs[:,1], label='x2')
plt.xlabel('Time')
plt.ylabel('State')
plt.title('Example 5')
plt.legend()
plt.show()




