#import sys
#sys.path.append('/home/rayksm/rlfinal/abc_py-master/abc_py-master/build')
import abc_py as abcPy
import numpy as np
import graphExtractor as GE
import torch
from dgl.nn.pytorch import GraphConv
import dgl

def custom_sign(x):
    sign = np.sign(x)
    sign[sign == 0] = -1
    return sign

class EnvGraph(object):
    """
    @brief the overall concept of environment, the different. use the compress2rs as target
    """
    def __init__(self, aigfile):
        self._abc = abcPy.AbcInterface()
        self._aigfile = aigfile
        self._abc.start()
        self.lenSeq = 0
        self._abc.read(self._aigfile)
        self.initStats = self._abc.aigStats() # The initial AIG statistics
        self.boundNumAnd = self._abc.numNodes()
        self.initNumAnd = float(self.initStats.numAnd)
        self.initLev = float(self.initStats.lev)
        self.resyn2() # run a compress2rs as target
        self.resyn2()
        resyn2Stats = self._abc.aigStats()
        totalReward = self.statValue(self.initStats) - self.statValue(resyn2Stats)
        self._rewardBaseline = totalReward / 20.0 # 18 is the length of compress2rs sequence
        self._andbasline = np.abs(resyn2Stats.numAnd - self.initStats.numAnd)
        self._levbaseline = np.abs(self.statValue_lev(resyn2Stats) - self.statValue_lev(self.initStats))
        self.total_action_len = 20
        self.target = 100
        print("baseline num AND ", resyn2Stats.numAnd, "\nBasline And Redution = ", self._andbasline, ", Basline Level Redution = ", self._levbaseline )

    def resyn2(self):
        self._abc.balance(l=False)
        self._abc.rewrite(l=False)
        self._abc.refactor(l=False)
        self._abc.balance(l=False)
        self._abc.rewrite(l=False)
        self._abc.rewrite(l=False, z=True)
        self._abc.balance(l=False)
        self._abc.refactor(l=False, z=True)
        self._abc.rewrite(l=False, z=True)
        self._abc.balance(l=False)
    def reset(self):
        self.lenSeq = 0
        self._abc.end()
        self._abc.start()
        self._abc.read(self._aigfile)
        self._lastStats = self._abc.aigStats() # The initial AIG statistics
        self._curStats = self._lastStats # the current AIG statistics
        
        self.lastAct = self.numActions()
        self.lastAct2 = self.numActions()
        self.lastAct3 = self.numActions()
        self.lastAct4 = self.numActions()
        
        self.lastand = 0
        self.lastand2 = 0
        self.lastand3 = 0
        self.lastand4 = 0

        self.actsTaken = np.zeros(self.numActions())
        return self.state()
    def close(self):
        self.reset()
    def step(self, actionIdx):
        self.takeAction(actionIdx)
        nextState = self.state()
        reward = self.reward()
        done = False
        if (self.lenSeq >= 20):
            done = True
        return nextState,reward,done,0
    def takeAction(self, actionIdx):
        """
        @return true: episode is end
        """
        # "b -l; rs -K 6 -l; rw -l; rs -K 6 -N 2 -l; rf -l; rs -K 8 -l; b -l; rs -K 8 -N 2 -l; rw -l; rs -K 10 -l; rwz -l; rs -K      10 -N 2 -l; b -l; rs -K 12 -l; rfz -l; rs -K 12 -N 2 -l; rwz -l; b -l
        self.lastAct4 = self.lastAct3
        self.lastAct3 = self.lastAct2
        self.lastAct2 = self.lastAct
        self.lastAct = actionIdx

        #self.actsTaken[actionIdx] += 1
        self.lenSeq += 1

        if actionIdx == 0:
            self._abc.balance(l=False) # b
        elif actionIdx == 1:
            self._abc.rewrite(l=False) # rw
        elif actionIdx == 2:
            self._abc.refactor(l=False) # rf
        elif actionIdx == 3:
            self._abc.rewrite(l=False, z=True) #rw -z
        elif actionIdx == 4:
            #self._abc.refactor(l=True, z=True) # rfz -l
            self._abc.refactor(l=False, z=True) #rf -z
        elif actionIdx == 5:
            self._abc.resub(k=6, l=False)
            #self._abc.resub(k=6, l=True)
            #self._abc.end()
            #return True
        #elif actionIdx == 6:
        #    self._abc.refactor(l=False, z=True) #rs
        else:
            assert(False)
        """
        elif actionIdx == 3:
            self._abc.rewrite(z=True) #rwz
        elif actionIdx == 4:
            self._abc.refactor(z=True) #rfz
        """

        # update the statitics
        self._lastStats = self._curStats
        self._curStats = self._abc.aigStats()

        self.lastand4 = self.lastand3
        self.lastand3 = self.lastand2
        self.lastand2 = self.lastand
        self.lastand = self.statValue(self._lastStats) - self.statValue(self._curStats)

        return self.lenSeq
    def state(self):
        """
        @brief current state
        """
        #oneHotAct = np.zeros(self.numActions())
        #np.put(oneHotAct, self.lastAct, 1)
        
        #lastOneHotActs  = np.zeros(self.numActions())
        #lastOneHotActs[self.lastAct2] += 1/3
        #lastOneHotActs[self.lastAct3] += 1/3
        #lastOneHotActs[self.lastAct] += 1/3

        lastOneHotActs = np.array([self.lastAct, self.lastAct2, self.lastAct3, self.lastAct4])
        lastOneHotAnds = np.array([self.lastand, self.lastand2, self.lastand3, self.lastand4]) / self.initLev

        
        stateArray = np.array([self._curStats.numAnd, self._curStats.lev,
            self._lastStats.numAnd, self._lastStats.lev]) / self.initLev
        
        #stepArray = np.array([float(self.lenSeq) / 20.0])
        #stepArray = np.array([float(self.lenSeq)])
        stepArray = np.zeros(self.total_action_len + 1)
        stepArray[self.lenSeq] = 10.0

        #combined = np.concatenate((stateArray, lastOneHotActs, stepArray), axis=-1)
        combined = np.concatenate((stateArray, lastOneHotActs, lastOneHotAnds, stepArray), axis=-1)
        #combined = np.expand_dims(combined, axis=0)
        #return stateArray.astype(np.float32)
        combined_torch =  torch.from_numpy(combined.astype(np.float32)).float()
        graph = None #GE.extract_dgl_graph(self._abc, self.boundNumAnd)
        return (combined_torch, graph)
    
    def reward(self):
        
        val = np.abs(self.statValue(self._lastStats) - self.statValue(self._curStats))
        val_sign = np.sign(int(self.statValue(self._lastStats)) - int(self.statValue(self._curStats)))

        ## Penalty
        ## ============================================
        ## with penalty
        if self.lastAct == self.lastAct2:
            penalty = -0.3
        else:
            penalty = 0
        ## ============================================
        ## without penalty
        # penalty = 0
        ## ============================================

        
        ## Advance
        ## ============================================
        ## with advance
        candy = 10
        if self.lenSeq > self.total_action_len - 1:
            if self.statValue(self._curStats) < self.nowtarget:
                advance = candy
                self.nowtarget = self.statValue(self._curStats)
            else:
                advance = candy - 1.5 * ((self.statValue(self._curStats) - self.nowtarget) // 10 + 1)
        else:
            advance = 0 
        ## ============================================
        ## without penalty
        # advance = 0 
        ## ============================================

        

        return (val_sign * val / 50 + penalty + advance)
            
    def numActions(self):
        return 5
    def dimState(self):
        #return 4 + self.numActions() * 1 + 1
        #return 4 + self.numActions() * 1 + self.total_action_len + 1
        return 4 + 4 * 2 + self.total_action_len + 1
    def returns(self):
        return [self._curStats.numAnd , self._curStats.lev]
    def statValue(self, stat):
        #return float(stat.lev)  / float(self.initLev)
        return float(stat.numAnd) #  + float(stat.lev)  / float(self.initLev)
        #return stat.numAnd + stat.lev * 10
    def statValue_lev(self, stat):
        return float(stat.lev)
    def curStatsValue(self):
        return self.statValue(self._curStats)
    def seed(self, sd):
        pass
    def compress2rs(self):
        self._abc.compress2rs()

