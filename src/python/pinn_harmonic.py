# python/pinn_harmonic.py
import math, torch, numpy as np
import torch.nn as nn, torch.optim as optim
import matplotlib.pyplot as plt

omega = 2.0
A_true, B_true = 1.0, 0.5
def x_true(t): return A_true*np.cos(omega*t)+B_true*np.sin(omega*t)

t_obs = np.linspace(0, 2*math.pi, 10)[:,None]
x_obs_noisy = x_true(t_obs) + 0.02*np.random.randn(*t_obs.shape)
t_colloc = np.linspace(0, 2*math.pi, 200)[:,None]

device = torch.device("cpu")
t_obs_t = torch.tensor(t_obs, dtype=torch.float32, device=device, requires_grad=True)
x_obs_t = torch.tensor(x_obs_noisy, dtype=torch.float32, device=device)
t_colloc_t = torch.tensor(t_colloc, dtype=torch.float32, device=device, requires_grad=True)

class PINN(nn.Module):
    def __init__(self, hidden=64, nlayers=3):
        super().__init__()
        layers=[]
        in_dim=1
        for _ in range(nlayers):
            layers += [nn.Linear(in_dim, hidden), nn.Tanh()]
            in_dim=hidden
        layers.append(nn.Linear(hidden,1))
        self.net=nn.Sequential(*layers)
    def forward(self,t): return self.net(t)

model = PINN().to(device)
def ode_residual(model,t):
    x = model(t)
    x_t = torch.autograd.grad(x, t, grad_outputs=torch.ones_like(x), create_graph=True)[0]
    x_tt = torch.autograd.grad(x_t, t, grad_outputs=torch.ones_like(x_t), create_graph=True)[0]
    return x_tt + (omega**2) * x

opt = optim.Adam(model.parameters(), lr=1e-3)
for epoch in range(3000):
    opt.zero_grad()
    loss_data = nn.MSELoss()(model(t_obs_t), x_obs_t)
    loss_phys = torch.mean(ode_residual(model, t_colloc_t)**2)
    loss = loss_data + 1.0*loss_phys
    loss.backward(); opt.step()
    if (epoch+1)%500==0:
        print(epoch+1, loss.item(), loss_data.item(), loss_phys.item())

# save TorchScript
model.eval()
scripted = torch.jit.script(model)
# Save the scripted ScriptModule using its own save method
scripted.save("pinn_harmonic.pt")
print("Saved: pinn_harmonic.pt")

# quick plot
t_test = np.linspace(0,2*math.pi,400)[:,None]
with torch.no_grad():
    t_test_t = torch.tensor(t_test, dtype=torch.float32)
    y = model(t_test_t).cpu().numpy()
plt.plot(t_test, x_true(t_test), label='true')
plt.plot(t_test, y, '--', label='pinn')
plt.scatter(t_obs, x_obs_noisy, color='red')
plt.legend(); plt.savefig('pinn_result.png'); print('Saved pinn_result.png')