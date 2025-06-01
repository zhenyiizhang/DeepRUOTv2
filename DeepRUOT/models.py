import torch, torch.nn as nn
import math
import joblib


import torch.nn.init as init


class velocityNet(nn.Module):
    # input x, t to get v= dx/dt
    def __init__(self, in_out_dim, hidden_dim, n_hiddens, activation='Tanh', use_spatial = False):
        super().__init__()
        Layers = [in_out_dim+1]
        for i in range(n_hiddens):
            Layers.append(hidden_dim)
        Layers.append(in_out_dim)
        
        if activation == 'Tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'leakyrelu':
            self.activation = nn.LeakyReLU()
        
        self.use_spatial = use_spatial

        if use_spatial:
            # Spatial velocity
            self.spatial_net = nn.ModuleList(
            [nn.Sequential(
                nn.Linear(Layers[i], Layers[i + 1]),
                self.activation,
            )
                for i in range(len(Layers) - 2)
            ]
            )
            self.spatial_out = nn.Linear(Layers[-2], 2)

            # Gene velocity
            self.gene_net = nn.ModuleList(
            [nn.Sequential(
                nn.Linear(Layers[i], Layers[i + 1]),
                self.activation,
            )
                for i in range(len(Layers) - 2)
            ]
            )
            self.gene_out = nn.Linear(Layers[-2], in_out_dim - 2)
        else:
            self.net = nn.ModuleList(
                [nn.Sequential(
                    nn.Linear(Layers[i], Layers[i + 1]),
                    self.activation,
                )
                    for i in range(len(Layers) - 2)
                ]
            )
            self.out = nn.Linear(Layers[-2], Layers[-1])

    def forward(self, t, x):
        # x is N*2
        num = x.shape[0]
        #print(num)
        t = t.expand(num, 1)  # Maintain gradient information of t and expand its shape
        #print(t)
        state  = torch.cat((t,x),dim=1)
        #print(state)
        if self.use_spatial:
            ii = 0
            for layer in self.spatial_net:
                if ii == 0:
                    x = layer(state)
                else:
                    x = layer(x)
                ii =ii+1
            spatial_x = self.spatial_out(x)

            ii = 0
            for layer in self.gene_net:
                if ii == 0:
                    x = layer(state)
                else:
                    x = layer(x)
                ii =ii+1
            gene_x = self.gene_out(x)
            x = torch.cat([spatial_x, gene_x], dim = 1)
        else:
            ii = 0
            for layer in self.net:
                if ii == 0:
                    x = layer(state)
                else:
                    x = layer(x)
                ii =ii+1
            x = self.out(x)
        return x

class growthNet(nn.Module):
    # input x, t to get g
    def __init__(self, in_out_dim, hidden_dim, activation='Tanh'):
        super().__init__()
        if activation == 'Tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'leakyrelu':
            self.activation = nn.LeakyReLU()

        self.net = nn.Sequential(
            nn.Linear(in_out_dim+1, hidden_dim),
            self.activation,
            nn.Linear(hidden_dim,hidden_dim),
            self.activation,
            nn.Linear(hidden_dim,hidden_dim),
            self.activation,
            nn.Linear(hidden_dim,1))
    def forward(self, t, x):
        # x is N*2
        num = x.shape[0]
        t = t.expand(num, 1)  # Maintain gradient information of t and expand its shape
        state  = torch.cat((t,x),dim=1)
        return self.net(state)

class scoreNet(nn.Module):
    # input x, t to get g
    def __init__(self, in_out_dim, hidden_dim, activation='Tanh'):
        super().__init__()
        if activation == 'Tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'leakyrelu':
            self.activation = nn.LeakyReLU()

        self.net = nn.Sequential(
            nn.Linear(in_out_dim+1, hidden_dim),
            self.activation,
            nn.Linear(hidden_dim,hidden_dim),
            self.activation,
            nn.Linear(hidden_dim,hidden_dim),
            self.activation,
            nn.Linear(hidden_dim,1))
    def forward(self, t, x):
        # x is N*2
        num = x.shape[0]
        t = t.expand(num, 1)  # Maintain gradient information of t and expand its shape
        state  = torch.cat((t,x),dim=1)
        return self.net(state)

class dediffusionNet(nn.Module):
    # input x, t to get g
    def __init__(self, in_out_dim, hidden_dim, activation='Tanh'):
        super().__init__()
        if activation == 'Tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'leakyrelu':
            self.activation = nn.LeakyReLU()

        self.net = nn.Sequential(
            nn.Linear(in_out_dim+1, hidden_dim),
            self.activation,
            nn.Linear(hidden_dim,hidden_dim),
            self.activation,
            nn.Linear(hidden_dim,hidden_dim),
            self.activation,
            nn.Linear(hidden_dim,1))
    def forward(self, t, x):
        # x is N*2
        num = x.shape[0]
        t = t.expand(num, 1)  # Maintain gradient information of t and expand its shape
        state  = torch.cat((t,x),dim=1)
        return self.net(state)

class indediffusionNet(nn.Module):
    # input x, t to get g
    def __init__(self, in_out_dim, hidden_dim, activation='Tanh'):
        super().__init__()
        if activation == 'Tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'leakyrelu':
            self.activation = nn.LeakyReLU()

        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            self.activation,
            nn.Linear(hidden_dim,hidden_dim),
            self.activation,
            nn.Linear(hidden_dim,hidden_dim),
            self.activation,
            nn.Linear(hidden_dim,1))
    def forward(self, t, x):
        # x is N*2
        num = x.shape[0]
        t = t.expand(num, 1)  # Maintain gradient information of t and expand its shape
        #state  = torch.cat((t,x),dim=1)
        return self.net(t)

class FNet(nn.Module):
    def __init__(self, in_out_dim, hidden_dim, n_hiddens, activation):
        super(FNet, self).__init__()
        self.in_out_dim = in_out_dim
        self.hidden_dim = hidden_dim
        self.v_net = velocityNet(in_out_dim, hidden_dim, n_hiddens, activation)  # v = dx/dt
        self.g_net = growthNet(in_out_dim, hidden_dim, activation)  # g
        self.s_net = scoreNet(in_out_dim, hidden_dim, activation)  # s = log rho
        self.d_net = indediffusionNet(in_out_dim, hidden_dim, activation)  # d = sigma(t)

    def forward(self, t, z):
        with torch.set_grad_enabled(True):
            z.requires_grad_(True)
            t.requires_grad_(True)

            v = self.v_net(t, z).float()
            g = self.g_net(t, z).float()
            s = self.s_net(t, z).float()
            d = self.d_net(t, z).float()

        return v, g, s, d



class ODEFunc2(nn.Module):
    def __init__(self, f_net, use_mass = True):
        super(ODEFunc2, self).__init__()
        self.f_net = f_net
        self.use_mass = use_mass

    def forward(self, t, state):
        z, lnw, _= state
        v, g, _, _ = self.f_net(t, z)
        
        dz_dt = v
        dlnw_dt = g
        w = torch.exp(lnw)
        if self.use_mass:
            dm_dt = (torch.norm(v, p=2,dim=1).unsqueeze(1)**2 + g**2) * w
        else:
            dm_dt = torch.norm(v, p=2,dim=1).unsqueeze(1)**2
        
        return dz_dt.float(), dlnw_dt.float(), dm_dt.float()

class ODEFunc(nn.Module):
    def __init__(self, v_net):
        super(ODEFunc, self).__init__()
        self.v_net = v_net

    def forward(self, t, z):
        dz_dt = self.v_net(t, z)
        return dz_dt.float()

class scoreNet2(nn.Module):
    # input x, t to get g
    def __init__(self, in_out_dim, hidden_dim, activation='Tanh'):
        super().__init__()
        if activation == 'Tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'leakyrelu':
            self.activation = nn.LeakyReLU()

        self.net = nn.Sequential(
            nn.Linear(in_out_dim+1, hidden_dim),
            self.activation,
            nn.Linear(hidden_dim,hidden_dim),
            self.activation,
            nn.Linear(hidden_dim,hidden_dim),
            self.activation,
            nn.Linear(hidden_dim,1))
    def forward(self, t, x):

        state  = torch.cat((t,x),dim=1)
        return self.net(state)
    
    def compute_gradient(self, t, x):
        x = x.requires_grad_(True)
        output = self.forward(t, x)
        gradient = torch.autograd.grad(outputs=output, inputs=x,
                                       grad_outputs=torch.ones_like(output),
                                       create_graph=True)[0]
        return gradient



class ODEFunc3(nn.Module):
    def __init__(self, f_net,sf2m_score_model,sigma, use_mass):
        super(ODEFunc3, self).__init__()
        self.f_net = f_net
        self.sf2m_score_model = sf2m_score_model
        self.sigma=sigma
        self.use_mass = use_mass


    def forward(self, t, state):
        z, lnw, m = state
        w = torch.exp(lnw)
        z.requires_grad_(True)
        lnw.requires_grad_(True)
        m.requires_grad_(True)
        t.requires_grad_(True)

        
        v, g, _, _ = self.f_net(t, z)
        v.requires_grad_(True)
        g.requires_grad_(True)
        #s.requires_grad_(True)
        time=t.expand(z.shape[0],1)
        time.requires_grad_(True)
        s=self.sf2m_score_model(time,z)
        
        dz_dt = v
        dlnw_dt = g

        z=z.requires_grad_(True)
        #grad_s = torch.autograd.grad(s.sum(), z)[0].requires_grad_() #need to change 
        grad_s = torch.autograd.grad(outputs=s, inputs=z,grad_outputs=torch.ones_like(s),create_graph=True)[0]

        norm_grad_s = torch.norm(grad_s, dim=1).unsqueeze(1).requires_grad_(True)
        
        
        #w = torch.exp(lnw)
        #dm_dt = (torch.norm(v, p=2,dim=1).unsqueeze(1) + g**2) * w
        if self.use_mass:
            dm_dt = (torch.norm(v, p=2, dim=1).unsqueeze(1) ** 2 / (2) + 
                (norm_grad_s ** 2) / 2 -
                (1 / 2 * self.sigma ** 2 *g + s* g) + g ** 2) * w
        else:
            dm_dt = (torch.norm(v, p=2, dim=1).unsqueeze(1) ** 2 / (2) + 
                    (norm_grad_s ** 2) / 2 + torch.norm(v, p=2, dim=1).unsqueeze(1) * torch.norm(grad_s, p=2, dim=1).unsqueeze(1))
        
        return dz_dt.float(), dlnw_dt.float(), dm_dt.float()