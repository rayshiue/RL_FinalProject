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
        
'''
class RLTrainer(object):
    def __init__(self, env, gamma, pi, baseline):
        self._env = env
        self._gamma = gamma
        self._pi = pi
        self._baseline = baseline
        self.memTrajectory = [] # the memorized trajectories. sorted by value
        self.memLength = 4
        self.sumRewards = []
        self.lenSeq = 0
        self.count_update = 0
        self.TRewards = 0
        self.TNumAnd = 0
    
    def episode(self, gen_traj = 10, phaseTrain=True):
        
        self.lenSeq = 0
        self.updateTrajectory(gen_traj, phaseTrain)
        self._pi.episode()
        return self.TNumAnd / gen_traj, self.TRewards / gen_traj
        #return [self._env._curstate]
    
    def updateTrajectory(self, gen_traj, phaseTrain=True):
        self.TRewards = 0
        self.TNumAnd = 0

        steplen = self._env.total_action_len
        update_time = 0
        for gg in range(gen_traj):
            self._env.reset()
            state = self._env.state()
            term = 0
            states, advantages, Gs, actions = [], [], [], []

            thiseporeward = 0

            while term < steplen:

                action = self._pi(state[0], state[1], phaseTrain, 1)
                term = self._env.takeAction(action)

                nextState = self._env.state()
                nextReward = self._env.reward()

                if term < steplen:
                    g = nextReward + self._gamma * self._baseline.value(nextState[0], nextState[1])
                else:
                    g = nextReward

                baseline = self._baseline(state[0], state[1])
                delta = g - baseline
                
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

            print(f"|| TR = {thiseporeward:>6.3f}, RA = {int(self._env.curStatsValue()):4d}", end=" || \n")

            if phaseTrain:

                for i in range(steplen):
                    state = states[i]
                    action = actions[i]
                    g = Gs[i]
                    culmu_advantage = sum(advantages[k] for k in range(i, steplen))

                    self._baseline.update(state[0], action, g, state[1])
                    self._pi.update(state[0], state[1], action, 1, culmu_advantage)

                update_time += 1
                if update_time % 2 == 0:  # origin = 5
                    self._baseline.update_old_policy()
                if update_time % 2 == 0:  # origin = 5
                    self._pi.update_old_policy()

'''

class RLTrainer(object):
    def __init__(self, env, gamma, pi, value_network):
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

    def genTrajectory(self, phaseTrain=True):
        self._env.reset()
        state = self._env.state()
        term = False
        states, rewards, actions = [], [0], []
        while not term:
            action = self._pi(state[0], state[1], phaseTrain)
            term = self._env.takeAction(action)

            nextState = self._env.state()
            nextReward = self._env.reward()

            states.append(state)
            rewards.append(nextReward)
            actions.append(action)

            state = nextState

            if len(states) > 20:
                term = True

        return Trajectory(states, rewards, actions, self._env.curStatsValue())
    
    def episode(self, gen_traj = 10, phaseTrain=True):
        #trajectories = []
        #for _ in range(gen_traj):
        #    trajectory = self.genTrajectory(phaseTrain=phaseTrain) # Generate a trajectory of episode of states, actions, rewards
        #    trajectories.append(trajectory)
        
        self.lenSeq = 0
        self.updateTrajectory(gen_traj, phaseTrain)
        self._pi.episode()
        return self.TNumAnd / gen_traj, self.TRewards / gen_traj
        #return [self._env._curstate]
    
    def updateTrajectory(self, gen_traj, phaseTrain=True):
        #TRewards = []
        #avgnodes = []
        #avgedges = []
        self.TRewards = 0
        self.TNumAnd = 0

        steplen = self._env.total_action_len
        update_time = 0
        for gg in range(gen_traj):
            self._env.reset()
            state = self._env.state()
            term = 0
            states, advantages, Gs, actions, vlosses = [], [], [], [], []

            thiseporeward = 0
            #states = trajectory.states
            #rewards = trajectory.rewards
            #actions = trajectory.actions

            #bisect.insort(self.memTrajectory, trajectory) # memorize this trajectory
            #self.lenSeq = len(states) # Length of the episode

            #for tIdx in range(self.lenSeq):
            while term < steplen:

                action = self._pi(state[0], state[1], phaseTrain, 1)
                term = self._env.takeAction(action)

                nextState = self._env.state()
                nextReward = self._env.reward()

                #G = sum(self._gamma ** (k - tIdx - 1) * rewards[k] for k in range(tIdx + 1, self.lenSeq + 1))
                #G = nextReward + self._gamma * self._baseline.maxvalue(nextState[0], nextState[1])
                if term < steplen:
                    #next_action = self._pi(nextState[0], nextState[1], phaseTrain, 0)
                    #G = nextReward + self._gamma * self._baseline(nextState[0], next_action, nextState[1])
                    g = nextReward + self._gamma * self.value_network.value(nextState[0], nextState[1])
                    #G = nextReward + self._gamma * self._baseline.maxvalue(nextState[0], nextState[1])
                else:
                    g = nextReward

                value = self.value_network(state[0], state[1])
                delta = g - value
                #delta = baseline
                #print("(The delta = ", delta.data.item(), ", baseline = ", baseline.data.item(), end=") ")
                #print(f"(Delta = {delta.data.item():.3f}, G = {G.item():.3f}, Baseline = {baseline.item():.3f}", end=") | ")
                #print(f"(The delta = {delta.data.item():.3f}, G = {nextReward:.3f}", end=") | ")
                #print(f"(The delta = {delta.data.item():.3f}, G + baseline = {G.item():.3f}", end=") | ")
                
                #if phaseTrain:
                #    self._baseline.update(state[0], action, g, state[1])
                #    self._pi.update(state[0], state[1], action, 1, delta)
                
                states.append(state)
                actions.append(action)
                advantages.append(delta)
                Gs.append(g)
                #vlosses.append(self._baseline.vloss)

                state = nextState

                self.lenSeq += 1
                self.count_update += 1

                self.TRewards += nextReward
                thiseporeward += nextReward
                if term == steplen: self.TNumAnd += self._env.returns()[0]

                #print(term)
            #print(thiseporeward, self._env.statValue(state))
            print(f"|| TR = {thiseporeward:>7.3f}, RA = {int(self._env.curStatsValue()):4d}", end=" || \n")

            if phaseTrain:

                for i in range(steplen):
                    state = states[i]
                    action = actions[i]
                    g = Gs[i]
                    culmu_advantage = sum(advantages[k] for k in range(i, steplen))
                    #vloss = vlosses[i]

                    self.value_network.update(state[0], action, g, state[1])
                    self._pi.update(state[0], state[1], action, 1, culmu_advantage, self.value_network.vloss.item())


                update_time += 1
                if update_time % 2 == 0:  # origin = 5
                    self.value_network.update_old_policy()
                if update_time % 2 == 0:  # origin = 5
                    self._pi.update_old_policy()

            #print("-----------------------------------------------")
            #print("Total Reward = ", self.TRewards)    
            #print(state[1].edata['feat'])
            #self.sumRewards.append(sum(rewards))
            #TRewards.append(sum(rewards))
            #avgnodes.append(state[1].num_nodes())
            #avgedges.append(state[1].num_edges())
        
        
        #print('Sum Reward = ', sum(TRewards) / gen_traj)
        #print(rewards)
        #print('Avg Nodes  = ', sum(avgnodes) / gen_traj)
        #print('Avg Edges  = ', sum(avgedges) / gen_traj, "\n")
