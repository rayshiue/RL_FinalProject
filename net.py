import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical
import bisect
from dgl.nn.pytorch import GraphConv
import dgl
import torch.nn.init as init

torch.manual_seed(2024)

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class FcModel(nn.Module):
    def __init__(self, numFeats, outChs):
        super(FcModel, self).__init__()
        self._numFeats = numFeats
        self._outChs = outChs

        self.fc1 = nn.Linear(numFeats, 32).to(device)
        self.act1 = nn.ReLU()
        
        # self.gcn = GCN(7, 64, 8)
        #self.gcn = GCN(6, 64, 16)
        

        # self.fc2 = nn.Linear(32 + 8, 32).to(device)
        # self.act2 = nn.ReLU()

        self.fc3 = nn.Linear(32, 32).to(device)
        self.act3 = nn.ReLU()

        self.fc4 = nn.Linear(32, 32).to(device)
        self.act4 = nn.ReLU()

        self.fc5 = nn.Linear(32, 32).to(device)
        self.act5 = nn.ReLU()

        self.fc6 = nn.Linear(32, outChs).to(device)

        # Custom initialization
        self._initialize_weights()

    def _initialize_weights(self):
        # Initialize each layer
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if module.bias is not None:
                    init.constant_(module.bias, 0)

            '''
            elif isinstance(module, GCN):
                # If GCN has parameters, initialize them here
                for param in module.parameters():
                    if param.dim() > 1:
                        init.xavier_uniform_(param)
            '''

    def forward(self, x, graph):
        x = x.to(device)
        #graph_state = self.gcn(graph)

        x = self.fc1(x)
        x = self.act1(x)
        # x_res = x

        x = self.fc3(x)
        x = self.act3(x)
        # x = x + x_res

        x = self.fc4(x)
        x = self.act4(x)
        # x_res = x

        x = self.fc5(x)
        x = self.act5(x)
        # x = x + x_res
        
        x = self.fc6(x)
        return x
    

class FcModelGraph(nn.Module):
    def __init__(self, numFeats, outChs):
        super(FcModelGraph, self).__init__()
        self._numFeats = numFeats
        self._outChs = outChs
        
        self.fc1 = nn.Linear(numFeats, 32).to(device)
        # self.gcn = GCN(7, 64, 16)
        #self.gcn = GCN(6, 64, 16)
        self.act1 = nn.ReLU()

        # self.fc2 = nn.Linear(32 + 16, 32).to(device)
        self.fc2 = nn.Linear(32, 32).to(device)

        self.act2 = nn.ReLU()

        self.fc3 = nn.Linear(32, 32).to(device)
        self.act3 = nn.ReLU()

        self.fc4 = nn.Linear(32, 32).to(device)
        self.act4 = nn.ReLU()

        self.fc5 = nn.Linear(32, 32).to(device)
        self.act5 = nn.ReLU()

        self.fc6 = nn.Linear(32, outChs).to(device)

        # Custom initialization
        self._initialize_weights()

    def _initialize_weights(self):
        # Initialize each layer
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if module.bias is not None:
                    init.constant_(module.bias, 0)
            
            '''
            elif isinstance(module, GCN):
                # If GCN has parameters, initialize them here
                for param in module.parameters():
                    if param.dim() > 1:
                        init.xavier_uniform_(param)
            '''
    def forward(self, x, graph):
        x = x.to(device)
        # graph_state = self.gcn(graph)

        x = self.fc1(x)
        x = self.act1(x)

        # x = self.fc2(torch.cat((x, graph_state), dim = 0))
        x = self.fc2(x)
        x = self.act2(x)
        # x_res = x

        x = self.fc3(x)
        x = self.act3(x)
        # x = x + x_res

        x = self.fc4(x)
        x = self.act4(x)
        # x_res = x

        x = self.fc5(x)
        x = self.act5(x)
        # x = x + x_res
        
        x = self.fc6(x)

        return x


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
        self._network.load_state_dict(torch.load(path))
    
    def save_model(self, path):
        torch.save(self._network.state_dict(), path)

    def __call__(self, s, graph, phaseTrain=True, ifprint = False):
        self._old_network.eval()
        s = s.to(device).float()
        out = self._old_network(s, graph)
        probs = F.softmax(out / self.tau, dim=-1) * (1 - self.explore) + self.exp_prob

        if phaseTrain:
            m = Categorical(probs)
            action = m.sample()
            if ifprint: print(f"{action.data.item()} ({out[action.data.item()].data.item():>6.3f})", end=" > ")
            self.count_print += 1
        else:
            action = torch.argmax(out)
            if ifprint: print(f"{action.data.item()}", end=" > ")
        return action.data.item()
    
    def update_old_policy(self):
        self._old_network.load_state_dict(self._network.state_dict())

    def update(self, s, graph, a, gammaT, delta, vloss, epsilon = 0.4, beta = 0.1, vbeta = 0.01):
        # PPO
        self._network.train()

        # now log_prob
        s = s.to(device).float()
        logits = self._network(s, graph)
        log_prob = torch.log_softmax(logits / self.tau, dim=-1)[a]

        # old log_prob
        with torch.no_grad():
            old_logits = self._old_network(s, graph)
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
        torch.nn.utils.clip_grad_norm_(self._network.parameters(), max_norm=10.0)

        self._optimizer.step()

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
    
    def load_model(self, path):
        self._network.load_state_dict(torch.load(path))
    
    def save_model(self, path):
        torch.save(self._network.state_dict(), path)
    
    def __call__(self, state, graph):
        self._old_network.eval()
        state = state.to(device).float()
        #return self.value(state, action, graph).data
        return self.value(state, graph).data
    
    """
    def maxvalue(self, state, graph):
        self._old_network.eval()
        state = state.to(device).float()
        
        # Initialize vmax with the value of the first action
        vmax = self.value(state, 0, graph).data
        
        for i in range(1, self._numActs):
            v_current = self.value(state, i, graph).data
            vmax = torch.max(vmax, v_current)
        
        return vmax
    """

    def value(self, state, graph):
        self._old_network.eval()
        state = state.to(device).float()
        #action_tensor = torch.tensor([action], dtype=state.dtype, device=state.device)
        #action_features = torch.cat((state, action_tensor), dim=-1)  # Concatenate state and action
        #out = self._old_network(action_features, graph)  # Pass both combined features and graph
        out = self._old_network(state, graph)
        return out
    
    def newvalue(self, state, graph):
        self._network.eval()
        state = state.to(device).float()
        #action_tensor = torch.tensor([action], dtype=state.dtype, device=state.device)
        #action_features = torch.cat((state, action_tensor), dim=-1)  # Concatenate state and action
        #out = self._network(action_features, graph)  # Pass both combined features and graph
        out = self._network(state, graph)
        return out
    
    def update_old_policy(self):
        self._old_network.load_state_dict(self._network.state_dict())

    def update(self, state, action, G, graph):
        self._network.train()
        state = state.to(device).float()
        #action = action.to(device).float().unsqueeze(0)
        vApprox = self.newvalue(state, graph)  # Estimate Q-value
        loss = (torch.tensor([G], device=device) - vApprox[-1]) ** 2 / 2

        self.vloss = loss
        self._optimizer.zero_grad()
        loss.backward()
    
        #if self.count_print % 25 == 0:
        #    total_norm = torch.nn.utils.clip_grad_norm_(self._network.parameters(), float('inf'))
        #    print(f"Value Network Gradient norm before clipping: {total_norm}")
        self.count_print += 1

        # Apply gradient clipping
        gradient_cut = 1000.0
        #total_norm = torch.nn.utils.clip_grad_norm_(self._network.parameters(), float('inf'))
        #if total_norm > gradient_cut: 
        #    print("Cut!", end=" ")
        #print(f"Value Network Gradient norm before clipping: {total_norm}", end=" ")

        torch.nn.utils.clip_grad_norm_(self._network.parameters(), max_norm = gradient_cut)  # Adjust max_norm as needed

        self._optimizer.step()

'''
class GCN(torch.nn.Module):
    def __init__(self, in_feats, hidden_size, out_len):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, hidden_size).to(device)
        self.conv2 = GraphConv(hidden_size, hidden_size).to(device)
        self.conv3 = GraphConv(hidden_size, hidden_size).to(device)
        self.conv4 = GraphConv(hidden_size, hidden_size).to(device)
        
        self.conv5 = GraphConv(hidden_size, int(hidden_size / 2)).to(device)
        self.conv6 = GraphConv(int(hidden_size / 2), int(hidden_size / 2)).to(device)

        self.conv7 = GraphConv(int(hidden_size / 2), out_len).to(device)



    def forward(self, g):
        g = g.to(device)
        g.ndata['feat'] = g.ndata['feat'].to(device)
        g = dgl.add_self_loop(g)

        h = self.conv1(g, g.ndata['feat'])
        h = torch.relu(h)
        h_res = h

        h = self.conv2(g, h)
        h = torch.relu(h)
        h = h + h_res

        h = self.conv3(g, h)
        h = torch.relu(h)
        h_res = h

        h = self.conv4(g, h)
        h = torch.relu(h)
        h = h + h_res
        
        h = self.conv5(g, h)
        h = torch.relu(h)
        h_res = h

        h = self.conv6(g, h)
        h = torch.relu(h)
        h = h + h_res

        h = self.conv7(g, h)
        h = torch.relu(h)
        g.ndata['h'] = h
        hg = dgl.mean_nodes(g, 'h')
        
        #hg = self.pool(g, h)
        #hg = self.fc(hg)
        
        return torch.squeeze(hg)
'''
