import random

class Trajectory(object):
    """
    @brief The experience of a trajectory
    """
    def __init__(self, states, rewards, actions, value):
        self.states = states
        self.rewards = rewards
        self.actions = actions
        self.value = value
    def __lt__(self, other):
        return self.value < other.value

class RLTrainer(object):
    def __init__(self, env, gamma, pi, value_network, GAE):
        self._env = env
        self._gamma = gamma
        self._pi = pi
        self.value_network = value_network
        self.memTrajectory = [] # the memorized trajectories. sorted by value
        self.memLength = 4
        self.sumRewards = []
        self.lenSeq = 0
        self.count_update = 0
        self.TRewards = 0
        self.TNumAnd = 0
        self.GAE = GAE

    def episode(self, gen_traj = 10, phaseTrain=True):
        self.lenSeq = 0
        graph_state_list = self.updateTrajectory(gen_traj, phaseTrain)
        self._pi.episode()
        return self.TNumAnd / gen_traj, self.TRewards / gen_traj, graph_state_list

    def save_policy_net(self, path):
        self._pi.save_model(path)
    
    def load_policy_net(self, path):
        self._pi.load_model(path)

    def save_value_net(self, path):
        self.value_network.save_model(path)
    
    def load_value_net(self, path):
        self.value_network.load_model(path)

    def updateTrajectory(self, gen_traj, phaseTrain=True):
        self.TRewards = 0
        self.TNumAnd = 0

        steplen = self._env.total_action_len
        update_time = 0
        graph_state_list = []
        for gg in range(gen_traj):
            self._env.reset()
            state = self._env.state()
            term = 0
            states, advantages, Gs, actions, vlosses = [], [], [], [], []

            thiseporeward = 0
            inner_graph_states = []
            while term < steplen:

                action, graph_state = self._pi(state[0], state[1], phaseTrain, 1)
                
                print(int(self._env.returns()[0]))
                inner_graph_states.append(graph_state)
                term = self._env.takeAction(action)

                nextState = self._env.state()
                nextReward = self._env.reward()

                if term < steplen:
                    g = nextReward + self._gamma * self.value_network.value(nextState[0], nextState[1])
                else:
                    g = nextReward

                value = self.value_network(state[0], state[1])
                delta = g - value

                states.append(state)
                actions.append(action)
                advantages.append(delta)
                Gs.append(g)

                state = nextState

                self.lenSeq += 1
                self.count_update += 1

                self.TRewards += nextReward
                thiseporeward += nextReward
                if term == steplen: self.TNumAnd += self._env.returns()[0]
            graph_state_list.append(inner_graph_states)
            # print(f"|| TR = {thiseporeward:>7.3f}, RA = {int(self._env.curStatsValue()):4d}", end=" || \n")

            if phaseTrain:

                for i in range(steplen):
                    state = states[i]
                    action = actions[i]
                    g = Gs[i]
                    ending = min(i+1,steplen) 
                    culmu_advantage = sum(advantages[k]*(self.GAE**(k-i)) for k in range(i, steplen))
                    # culmu_advantage = sum(advantages[k]*(self.GAE**(k-i)) for k in range(i, ending))

                    self.value_network.update(state[0], action, g, state[1])
                    graph_state = self._pi.update(state[0], state[1], action, 1, culmu_advantage, self.value_network.vloss.item())
                    
                ## Update Frequency Ablation !!
                value_upt_period = 2    # 1, 1, 2, 2, 5
                policy_upt_period = 2   # 1, 2, 2, 5, 5

                update_time += 1
                if update_time % value_upt_period == 0:  # origin = 5
                    self.value_network.update_old_policy()
                if update_time % policy_upt_period == 0:  # origin = 5
                    self._pi.update_old_policy()
        return graph_state_list
