import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical
import bisect
import torch.nn.init as init
from dgl.nn.pytorch import GraphConv
import dgl

torch.manual_seed(2024)

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

class GCN(torch.nn.Module):
    def __init__(self, in_feats, hidden_size, out_len):
        super(GCN, self).__init__()

        self.conv1 = GraphConv(in_feats, hidden_size).to(device)
        self.conv2 = GraphConv(hidden_size, hidden_size).to(device)
        self.conv3 = GraphConv(hidden_size, hidden_size).to(device)
        self.conv4 = GraphConv(hidden_size, out_len).to(device)
        # self.ST_model = SetTransformer(dim_input=64, num_outputs=1, dim_output=64).to(device)

    def forward(self, g):
        g = g.to(device)
        g.ndata['feat'] = g.ndata['feat'].to(device)
        g = dgl.add_self_loop(g)
        h = self.conv1(g, g.ndata['feat'])
        h = torch.relu(h)
        h = self.conv2(g, h)
        h = torch.relu(h)
        #h = self.conv3(g, h)
        #h = torch.relu(h)
        h = self.conv4(g, h)
        g.ndata['h'] = h
        
        hg = dgl.mean_nodes(g, 'h')
        # hg = self.ST_model(h.unsqueeze(0))
        # print(hg.shape)
        return torch.squeeze(hg)

class FcModel(nn.Module):
    def __init__(self, numFeats, outChs):
        super(FcModel, self).__init__()
        self._numFeats = numFeats
        self._outChs = outChs

        self.fc1 = nn.Linear(numFeats, 64).to(device)
        self.act1 = nn.ReLU()

        self.fc2 = nn.Linear(64, 64).to(device)
        self.act2 = nn.ReLU()

        self.fc3 = nn.Linear(64, 64).to(device)
        self.act3 = nn.ReLU()

        self.fc4 = nn.Linear(64, 64).to(device)
        self.act4 = nn.ReLU()

        self.fc5 = nn.Linear(64, 32).to(device)
        self.act5 = nn.ReLU()

        self.fc6 = nn.Linear(32, outChs).to(device)

        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if module.bias is not None:
                    init.constant_(module.bias, 0)

    def forward(self, x, graph):
        x = x.to(device)

        x = self.fc1(x)
        x = self.act1(x)
        x_res = x

        x = self.fc2(x)
        x = self.act2(x)
        x = x + x_res

        x = self.fc3(x)
        x = self.act3(x)
        x_res = x

        x = self.fc4(x)
        x = self.act4(x)
        x = x + x_res

        x = self.fc5(x)
        x = self.act5(x)
        
        x = self.fc6(x)

        return x
    

class FcModelGraph(nn.Module):
    def __init__(self, numFeats, outChs):
        super(FcModelGraph, self).__init__()
        self._numFeats = numFeats
        self._outChs = outChs
        
        self.fc1 = nn.Linear(numFeats, 64).to(device)
        self.act1 = nn.ReLU()

        self.fc2 = nn.Linear(64+4, 64).to(device)      #with GCN
        # self.fc2 = nn.Linear(64, 64).to(device)      #without GCN
        self.act2 = nn.ReLU()

        self.fc3 = nn.Linear(64, 64).to(device)
        self.act3 = nn.ReLU()

        self.fc4 = nn.Linear(64, 64).to(device)
        self.act4 = nn.ReLU()

        self.fc5 = nn.Linear(64, 32).to(device)
        self.act5 = nn.ReLU()

        self.fc6 = nn.Linear(32, outChs).to(device)

        self.gcn = GCN(7, 12, 4)    # D256
        
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if module.bias is not None:
                    init.constant_(module.bias, 0)

    def forward(self, x, graph):
        x = x.to(device)
        graph_state = self.gcn(graph)

        x = self.fc1(x)
        x = self.act1(x)
        x_res = x

        # x = self.fc2(x)                               #without GCN
        x = self.fc2(torch.cat((x, graph_state), 0))    #with GCN
        x = self.act2(x)
        x = x + x_res

        x = self.fc3(x)
        x = self.act3(x)
        x_res = x

        x = self.fc4(x)
        x = self.act4(x)
        x = x + x_res

        x = self.fc5(x)
        x = self.act5(x)

        x = self.fc6(x)

        return x, graph_state


# Policy Network
class PolicyNetwork(object):
    def __init__(self, dimStates, numActs, alpha, network):
        self._dimStates = dimStates
        self._numActs = numActs
        self._alpha = alpha
        self._network = network(dimStates, numActs).to(device)
        self._old_network = network(dimStates, numActs).to(device)
        self._old_network.load_state_dict(self._network.state_dict())
        self._optimizer = torch.optim.Adam(self._network.parameters(), alpha, [0.9, 0.999])
        #self.tau = .5
        self.tau = 1 # temperature for gumbel_softmax # more random when tau > 1
        self.count_print = 0

        self.explore = 0
        self.exp_prob = torch.ones(numActs).to(device) * (self.explore / numActs)

    def load_model(self, path):
        self._old_network.load_state_dict(torch.load(path))
    
    def save_model(self, path):
        torch.save(self._network.state_dict(), path)

    def __call__(self, s, graph, phaseTrain=True, ifprint = False):
        self._old_network.eval()
        s = s.to(device).float()
        out, graph_state = self._old_network(s, graph)
        probs = F.softmax(out / self.tau, dim=-1) * (1 - self.explore) + self.exp_prob

        if phaseTrain:
            m = Categorical(probs)
            action = m.sample()
            # if ifprint: print(f"{action.data.item()} ({out[action.data.item()].data.item():>6.3f})", end=" > ")
            self.count_print += 1
        else:
            action = torch.argmax(out)
            # if ifprint: print(f"{action.data.item()}", end=" > ")
        return action.data.item(), graph_state
    
    def update_old_policy(self):
        self._old_network.load_state_dict(self._network.state_dict())

    def update(self, s, graph, a, gammaT, delta, vloss, epsilon = 0.1, beta = 0.1, vbeta = 0.01):
        # PPO
        self._network.train()

        # now log_prob
        s = s.to(device).float()
        logits, graph_state = self._network(s, graph)
        log_prob = torch.log_softmax(logits / self.tau, dim=-1)[a]

        # old log_prob
        with torch.no_grad():
            old_logits, graph_state = self._old_network(s, graph)
            old_log_prob = torch.log_softmax(old_logits / self.tau, dim=-1)[a]

        #ratio
        ratio = torch.exp(log_prob - old_log_prob)

        # entropy
        entropy = -torch.sum(F.softmax(logits, dim=-1) * log_prob, dim=-1).mean()
        

        # PPO clipping
        clipped_ratio = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
        loss = -torch.min(ratio * delta, clipped_ratio * delta) - beta * entropy + vbeta * vloss

        # gradient
        self._optimizer.zero_grad()
        loss.backward(retain_graph=True)

        # torch.nn.utils.clip_grad_norm_(self._network.parameters(), max_norm=1.0)
        # torch.nn.utils.clip_grad_norm_(self._network.parameters(), max_norm=100.0)

        self._optimizer.step()

        return graph_state
    def episode(self):
        pass

class ValueNetwork(object):
    def __init__(self, dimStates, numActs, alpha, network):
        self._dimStates = dimStates
        self._numActs = numActs
        self._alpha = alpha
        self._network = network(dimStates, 1).to(device)
        self._old_network = network(dimStates, 1).to(device)
        self._old_network.load_state_dict(self._network.state_dict())
        self._optimizer = torch.optim.Adam(self._network.parameters(), alpha, [0.9, 0.999])
        self.count_print = 0
        self.vloss = 0
    
    # def load_model(self, path):
    #     self._network.load_state_dict(torch.load(path))
    
    # def save_model(self, path):
    #     torch.save(self._network.state_dict(), path)
    
    def load_model(self, path):
        self._old_network.load_state_dict(torch.load(path))
    
    def save_model(self, path):
        torch.save(self._network.state_dict(), path)

    def __call__(self, state, graph):
        self._old_network.eval()
        state = state.to(device).float()
        #return self.value(state, action, graph).data
        return self.value(state, graph).data

    def value(self, state, graph):
        self._old_network.eval()
        state = state.to(device).float()
        out = self._old_network(state, graph)
        return out
    
    def newvalue(self, state, graph):
        self._network.eval()
        state = state.to(device).float()
        out = self._network(state, graph)
        return out
    
    def update_old_policy(self):
        self._old_network.load_state_dict(self._network.state_dict())

    def update(self, state, action, G, graph):
        self._network.train()
        state = state.to(device).float()
        vApprox = self.newvalue(state, graph)  # Estimate Q-value
        loss = (torch.tensor([G], device=device) - vApprox[-1]) ** 2 / 2

        self.vloss = loss
        self._optimizer.zero_grad()
        loss.backward()
    
        self.count_print += 1

        # Apply gradient clipping
        gradient_cut = 1000.0

        torch.nn.utils.clip_grad_norm_(self._network.parameters(), max_norm = gradient_cut)  # Adjust max_norm as needed

        self._optimizer.step()
