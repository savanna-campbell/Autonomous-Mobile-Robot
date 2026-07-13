#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy.interpolate import interp1d
import mdmm



# In[2]:


# initial guesses of parameters
r_R = 0.03
r_L = 0.03
l = 0.08
kg_R = 25.0
kg_L = 25.0
ka = 0.5
kf = 0.15
kq = 0.02
alpha = 8.0
beta = 8.0

# length of time robot runs in seconds
time_span = 10

# hyperparameters
epochs = 1000


# In[3]:


class TrajectoryData(Dataset):
    def __init__(self, csv_path, skiprows=10):
        df = pd.read_csv(csv_path, skiprows=skiprows)
        self.t = torch.tensor(df["time"].values, dtype=torch.float32).unsqueeze(1)
        self.states = torch.tensor(df[['x','y','theta', 'omega_R', 'omega_L']].values, dtype=torch.float32)
        self.controls = torch.tensor(df[['u_R', 'u_L']].values, dtype=torch.float32)

    def __len__(self):
        return len(self.t)

    def __getitem__(self, idx):
        return self.t[idx], self.states[idx], self.controls[idx]

# create object to hold the csv file created by the simulation
dataset = TrajectoryData('output.csv')
dataloader = DataLoader(dataset, batch_size=512, shuffle=True)


# In[4]:


# create the neural network

class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()

        # layers
        self.hidden1 = nn.Linear(3, 20)
        self.hidden2 = nn.Linear(20,20)
        self.output = nn.Linear(20, 5)


        # trainable parameters
        self.params = nn.Parameter(torch.tensor([r_R, r_L, l, kg_R, kg_L, ka, kf, kq], dtype=torch.float32))

    def forward(self, x):
        x = torch.tanh(self.hidden1(x))
        x = torch.tanh(self.hidden2(x))
        x = self.output(x)
        return x

model = NeuralNet()


# In[5]:


# grab collocation points
sobol = torch.quasirandom.SobolEngine(dimension=1)
t_collocation = sobol.draw(1024) * time_span
t_collocation.requires_grad_(True)

# create interpolation function
t_data = dataset.t.squeeze().numpy()
u_data = dataset.controls.numpy()

interpolation_function = interp1d(t_data, u_data, axis=0)

# interpolate for control values
t_interp = t_collocation.squeeze().detach().numpy()
u_interp = interpolation_function(t_interp)

# get tensors
u_collocation = torch.tensor(u_interp, dtype=torch.float32)


# In[6]:


# find residuals
def physics_residual(model, t_col, u_col, eta):
    out_col = model(torch.cat([t_col, u_col], dim=1))
    r_R = eta[0]
    r_L = eta[1]
    l = eta[2]
    kg_R = eta[3]
    kg_L = eta[4]
    ka = eta[5]
    kf = eta[6]
    kq = eta[7]
    u_R = u_col[:, 0:1]
    u_L = u_col[:, 1:2]

    # compute the gradients
    x = out_col[:, 0:1]
    y = out_col[:, 1:2]
    theta = out_col[:, 2:3]
    omega_R = out_col[:, 3:4]
    omega_L = out_col[:, 4:5]

    x_t = torch.autograd.grad(x, t_col, grad_outputs=torch.ones_like(x), create_graph=True)[0]
    y_t =  torch.autograd.grad(y, t_col, grad_outputs=torch.ones_like(y), create_graph=True)[0]
    theta_t =  torch.autograd.grad(theta, t_col, grad_outputs=torch.ones_like(theta), create_graph=True)[0]
    omega_R_t =  torch.autograd.grad(omega_R, t_col, grad_outputs=torch.ones_like(omega_R), create_graph=True)[0]
    omega_L_t =  torch.autograd.grad(omega_L, t_col, grad_outputs=torch.ones_like(omega_L), create_graph=True)[0]

    # calculate residuals    
    omega_r_residual = kg_R*u_R - ka*omega_R - kf*torch.tanh(alpha*omega_R) - kq*torch.tanh(beta*omega_R)*(omega_R**2) - omega_R_t
    omega_l_residual = kg_L*u_L - ka*omega_L - kf*torch.tanh(alpha*omega_L) - kq*torch.tanh(beta*omega_L)*(omega_L**2) - omega_L_t
    x_residual = ((r_R*omega_R/2) + (r_L*omega_L/2)) * torch.cos(theta) - x_t
    y_residual = ((r_R*omega_R/2) + (r_L*omega_L/2)) * torch.sin(theta) - y_t
    theta_residual = ((r_R*omega_R/(2*l)) - (r_L*omega_L/(2*l))) - theta_t

    return x_residual.pow(2).mean(), y_residual.pow(2).mean(), theta_residual.pow(2).mean(), omega_r_residual.pow(2).mean(), omega_l_residual.pow(2).mean()


# setup data loss
loss_fn = nn.MSELoss()


# In[7]:


# mdmm setup
constraint_x = mdmm.EqConstraint(lambda: physics_residual(model, t_collocation, u_collocation, model.params)[0], 0) 
constraint_y = mdmm.EqConstraint(lambda: physics_residual(model, t_collocation, u_collocation, model.params)[1], 0) 
constraint_theta = mdmm.EqConstraint(lambda: physics_residual(model, t_collocation, u_collocation, model.params)[2], 0) 
constraint_omega_R = mdmm.EqConstraint(lambda: physics_residual(model, t_collocation, u_collocation, model.params)[3], 0) 
constraint_omega_L = mdmm.EqConstraint(lambda: physics_residual(model, t_collocation, u_collocation, model.params)[4], 0) 

mdmm_module = mdmm.MDMM([
    constraint_x,
    constraint_y,
    constraint_theta,
    constraint_omega_R,
    constraint_omega_L
])

opt = mdmm_module.make_optimizer(model.parameters(), lr=1e-4)




# In[8]:


print(dataset.states.max(dim=0))
print(dataset.states.min(dim=0))




# In[9]:


# training loop
for epoch in range(epochs):
    for t_batch, states_batch, controls_batch in dataloader:
        outputs = model(torch.cat([t_batch, controls_batch], dim=1))
        loss = loss_fn(outputs, states_batch)
        mdmm_return = mdmm_module(loss)
        opt.zero_grad()
        mdmm_return.value.backward()
        opt.step()
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

