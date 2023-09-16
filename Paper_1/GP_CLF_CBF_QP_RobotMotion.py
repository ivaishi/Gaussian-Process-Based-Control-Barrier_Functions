import numpy as np
import cvxpy as cp  
import matplotlib.pyplot as plt
# For GPR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, DotProduct, ConstantKernel as C
from itertools import product
#np.random.seed(0)

N_datapoints = 100
# Trying random points to fit GPR
X_try = np.zeros((2, N_datapoints))
for ii in range(N_datapoints):
    X_try[0,ii] = np.random.uniform(0,10)  
    X_try[1,ii] = np.random.uniform(0,10)
print(X_try.shape)

X_data = X_try  # Data used to fit GPR

# Function Evaluation: to find f(x) for any given state
def function_evaluation(X):
    f_x = np.zeros(X.shape)
    for ii in range(X.shape[1]):
        f_x[0, ii] = (-X[1,ii] - (3/2)*(X[0,ii]**2) - (1/2)*(X[0,ii]*3))/6
        f_x[1, ii] = X[0,ii]/6
        
    return f_x

f_X = function_evaluation(X_data)
n_train_pts = 6
n_tst_pts = N_datapoints - n_train_pts
X_train = X_data.T
z_train_f1 = f_X[0,0:n_train_pts].reshape(-1,1)
z_train_f2 = f_X[1,0:n_train_pts].reshape(-1,1)

kern = 1 * RBF(length_scale=1.0, length_scale_bounds=(1e-3, 1e3))
gpr1 = GaussianProcessRegressor(kernel = kern, n_restarts_optimizer=9)
gpr1.fit(X_train[0:n_train_pts],z_train_f1)

gpr2 = GaussianProcessRegressor(kernel = kern, n_restarts_optimizer=9)
gpr2.fit(X_train[0:n_train_pts],z_train_f2)

z_mean_f1, z_sd_f1 = gpr1.predict(X_train[-n_tst_pts:], return_std = True)
z_mean_f2, z_sd_f2 = gpr2.predict(X_train[-n_tst_pts:], return_std = True)

true_f = function_evaluation(X_train[-n_tst_pts:].T)
true_f1 = true_f[0,:]
x_f1 = np.linspace(1, np.shape(true_f1)[0], np.shape(true_f1)[0])
fig, ax = plt.subplots(2,1)
ax[0].plot(x_f1, true_f1, 'k', label = 'True f1')
x_test_f1 = np.linspace(1, np.shape(z_mean_f1)[0], np.shape(z_mean_f1)[0])
ax[0].plot(x_test_f1, z_mean_f1, 'r', label = 'Predicted f1')
lower = z_mean_f1 - 1.96*z_sd_f1
upper = z_mean_f1 + 1.96*z_sd_f1
ax[0].fill_between(x_test_f1, lower, upper, alpha=0.6, label = '95% Confidence Interval' )
ax[0].legend()

true_f2 = true_f[1,:]
x_f2 = np.linspace(1, np.shape(true_f2)[0], np.shape(true_f2)[0])
print(true_f2.shape)
ax[1].plot(x_f2, true_f2, 'k', label = 'True f2')
x_test_f2 = np.linspace(1, np.shape(z_mean_f2)[0], np.shape(z_mean_f2)[0])
ax[1].plot(x_test_f2, z_mean_f2, 'r', label = 'Predicted f2')
lower = z_mean_f2 - 1.96*z_sd_f2
upper = z_mean_f2 + 1.96*z_sd_f2
ax[1].fill_between(x_test_f2, lower, upper, alpha=0.6, label = '95% Confidence Interval' )
ax[1].legend()
plt.show()

# plt.close()

u = cp.Variable((2,1))
delta = cp.Variable()
lfh = cp.Parameter((1,1))
lgh = cp.Parameter((1,2))
lfV = cp.Parameter((1,1))
lgV = cp.Parameter((1,2))
h = cp.Parameter()
V = cp.Parameter()
alpha = 0.5
k = 0.3
const = [ lfh + lgh @ u >= -alpha * h ]
const += [ lfV + lgV @ u<= -k * V + delta]
const += [cp.abs(u[0,0]) <= 3]
const += [cp.abs(u[1,0]) <= 3]
objective = cp.Minimize( cp.sum_squares( u ) + 1000*cp.sum_squares(delta) )
problem = cp.Problem( objective, const )

class robot:
    
    def __init__(self, x0, dt, ax):
        self.X = x0
        self.dt = dt
        self.body = ax.scatter([],[],c='r',s=10)
        self.render()
        
    def f(self):
        f1 = np.sin(self.X[0])
        f2 = np.sin(self.X[1])
        f = np.array([f1, f2])
        f = function_evaluation(self.X)
        return f.reshape(-1,1)
    
    def f_gpr(self): 
        #return self.f()
        mu_f1, std_f1 = gpr1.predict((self.X).T, return_std = True)
        mu_f2, std_f2 = gpr2.predict((self.X).T, return_std = True)
        f1_max = mu_f1 + std_f1
        f2_max = mu_f2 + std_f2
        f_x_max = np.array([f1_max, f2_max]).reshape(2,1)
        f1_min = mu_f1 - std_f1
        f2_min = mu_f2 - std_f2
        f_x_min = np.array([f1_min, f2_min]).reshape(2,1)
        f_x_mean = np.array([mu_f1, mu_f2]).reshape(2,1)
        print(f_x_mean)
        return f_x_max, f_x_min
        
    def g(self):
        return np.array([ [1, 0],
                         [0, 1]   
                         ])
        
    def barrier(self, obs):
        d_min = 0.3
        h = np.linalg.norm( self.X - obs )**2 - d_min**2
        dh_dx = (self.X - obs).T
        return h, dh_dx

    def lyapunov(self, goal):
        V = np.linalg.norm( self.X - goal )**2
        dV_dx = (self.X - goal).T
        return V, dV_dx
                
    def step(self, U):
        self.X = self.X + ( self.f() + self.g() @ U )*self.dt
        self.render()
        
    def render(self):
        self.body.set_offsets( [ self.X[0,0], self.X[1,0] ] )
        
      
plt.ion() # interactive mode ON
fig = plt.figure()
ax = plt.axes(xlim=(-5,5),ylim=(-5,5)) 
ax.set_xlabel("X")
ax.set_ylabel("Y")

circ = plt.Circle((1,1),0.3,linewidth = 1, edgecolor='k',facecolor='k')
ax.add_patch(circ)
  
initial_location = np.array([0,0]).reshape(-1,1)
dt  = 0.01
t_final = 3 
my_robot = robot( initial_location, dt, ax )

obs = np.array([1,1]).reshape(-1,1)
goal = np.array([2.5,2.5]).reshape(-1,1)

for i in range(int(t_final/dt)):
        
    h.value, dh_dx = my_robot.barrier(obs)
    V.value, dV_dx = my_robot.lyapunov(goal)
    f_max, f_min = my_robot.f_gpr()
    lfh_1 = dh_dx @ f_max
    lfh_2 = dh_dx @ f_min
    lfh.value = np.minimum(lfh_1,lfh_2)
    lgh.value = dh_dx @ my_robot.g()

    lfV_1 = dV_dx @ f_max
    lfV_2 = dV_dx @ f_min
    lfV.value = np.maximum(lfV_1,lfV_2)
    lgV.value = dV_dx @ my_robot.g()

    problem.solve()
    if problem.status != 'optimal':
        print("QP not solvable")
        
    my_robot.step(u.value)
    
    fig.canvas.draw()
    fig.canvas.flush_events()
    
    
plt.show()
    
