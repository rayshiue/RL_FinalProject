##
# @file testReinforce.py
# @author Keren Zhu
# @date 10/31/2019
# @brief The main for test REINFORCE
#
from datetime import datetime
import os
import torch

import trainer
import net
from env import EnvGraph as Env

import numpy as np
import statistics

import wandb
from wandb.integration.sb3 import WandbCallback

BENCHMARK_ROOT = '/home/ray/RL/hdl-benchmarks-master/'

class AbcReturn:
    def __init__(self, returns):
        self.numNodes = float(returns[0])
        self.level = float(returns[1])
    def __lt__(self, other):
        if (int(self.level) == int(other.level)):
            return self.numNodes < other.numNodes
        else:
            return self.level < other.level
    def __eq__(self, other):
        return int(self.level) == int(other.level) and int(self.numNodes) == int(self.numNodes)

def testReinforce(filename, design_name):
    
    # wandb.init(
    #     project="RLFinal_Ablation_Study_GCN",
    #     # name="10step state_ablation v0",
    #     # name= f"{design_name} Epsilon 0.2",
    #     name= f"{design_name} without GCN",
    #     # sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    #     # id = "v5_PPO_v3"
    # )
    
    now = datetime.now()
    dateTime = now.strftime("%m/%d/%Y, %H:%M:%S") + "\n"
    print("Time ", dateTime)

    env = Env(filename)
    policy_network = net.PolicyNetwork(env.dimState(), env.numActions(), 1e-4, net.FcModelGraph)
    #policy_network.load_state_dict(torch.load("xxx.pth"))

    value_network = net.ValueNetwork(env.dimState(), env.numActions(), 1e-4, net.FcModel)
    #value_network.load_state_dict(torch.load("xxx.pth"))

    ## Accumulation Ablation !!
    gamma = 1 # 0.1, 0.3, 0.5, 0.7, 0.9, 1.0
    GAE = 1
    rl_trainer = trainer.RLTrainer(env, gamma, policy_network, value_network, GAE)

    for idx in range(400):

        returns = rl_trainer.episode(phaseTrain=True)

        seqLen = rl_trainer.lenSeq
        line = "Iter " + str(idx) + ", NumAnd "+ str(returns[0]) + ", Seq Length " + str(seqLen) + "\n"
        
        # wandb.log({"Episode (x10)": idx,"NumAnd": returns[0],"avg_score": returns[1], "Best NumAnd": env.nowtarget})
        print(line)
        print("-----------------------------------------------")
        print("Action (Policy Value) > ... > || Total Reward, Remain AndGate ||\n")
        if idx%50==0:
            returns = rl_trainer.episode(phaseTrain=False)
            
            # wandb.log({"Episode (x10)": idx, "TestNumAnd": returns[0]})
    
    # for testing
    #returns = reinforce.episode(phaseTrain=False)
    #seqLen = reinforce.lenSeq
    #line = "Iter " + str(idx + 1) + ", NumAnd "+ str(returns[0]) + ", Level "+ str(returns[1]) + ", Seq Length " + str(seqLen) + "\n"
    print("Testing ")
    print("-----------------------------------------------")
    #lastfive.sort(key=lambda x : x.level)
    #lastfive = sorted(lastfive)
    returns = rl_trainer.episode(phaseTrain=False)
    seqLen = rl_trainer.lenSeq
    line = "Iter " + str(idx) + ", NumAnd "+ str(returns[0]) + ", Seq Length " + str(seqLen) + "\n"
    # wandb.log({"Episode (x10)": idx, "TestNumAnd": returns[0], "Best NumAnd": env.nowtarget})
    # wandb.finish()
    print(line)
    print("-----------------------------------------------")

    # save model
    # policy_network.save_model("model/vApprox2_dummynode.pth")
    # value_network.save_model("model/vbaseline2_dummynode.pth")


if __name__ == "__main__":
    """
    env = Env("./bench/i10.aig")
    vbaseline = RF.BaselineVApprox(4, 3e-3, RF.FcModel)
    for i in range(10000000):
        with open('log', 'a', 0) as outLog:
            line = "iter  "+ str(i) + "\n"
            outLog.write(line)
        vbaseline.update(np.array([2675.0 / 2675, 50.0 / 50, 2675. / 2675, 50.0 / 50]), 422.5518 / 2675)
        vbaseline.update(np.array([2282. / 2675,   47. / 50, 2675. / 2675,   47. / 50]), 29.8503 / 2675)
        vbaseline.update(np.array([2264. / 2675,   45. / 50, 2282. / 2675,   45. / 50]), 11.97 / 2675)
        vbaseline.update(np.array([2255. / 2675,   44. / 50, 2264. / 2675,   44. / 50]), 3 / 2675)
    """

    

    testReinforce(os.path.join(BENCHMARK_ROOT, "mcnc/Combinational/blif/dalu.blif"), "dalu")
    testReinforce(os.path.join(BENCHMARK_ROOT, "mcnc/Combinational/blif/k2.blif"), "k2")
    testReinforce(os.path.join(BENCHMARK_ROOT, "mcnc/Combinational/blif/mainpla.blif"), "mainpla")
    testReinforce(os.path.join(BENCHMARK_ROOT, "mcnc/Combinational/blif/apex1.blif"), "apex1")
    testReinforce(os.path.join(BENCHMARK_ROOT, "mcnc/Combinational/blif/bc0.blif"), "bc0")
    testReinforce(os.path.join(BENCHMARK_ROOT, "mcnc/Combinational/blif/C1355.blif"), "C1355")
    testReinforce(os.path.join(BENCHMARK_ROOT, "mcnc/Combinational/blif/C6288.blif"), "C6288")
    testReinforce(os.path.join(BENCHMARK_ROOT, "mcnc/Combinational/blif/C5315.blif"), "C5315")

    #testReinforce("/home/rayksm/rlfinal/benchmarks/flowtune_BLIF/bflyabc.blif", "bfly_abc")
    #testReinforce("./bench/MCNC/Combinational/blif/prom1.blif", "prom1")
    #testReinforce("./bench/MCNC/Combinational/blif/mainpla.blif", "mainpla")
    #testReinforce("./bench/MCNC/Combinational/blif/k2.blif", "k2")
    #testReinforce("./bench/ISCAS/blif/c5315.blif", "c5315")
    #testReinforce("./bench/ISCAS/blif/c6288.blif", "c6288")
    #testReinforce("./bench/MCNC/Combinational/blif/apex1.blif", "apex1")
    #testReinforce("./bench/MCNC/Combinational/blif/bc0.blif", "bc0")
    #testReinforce("./bench/i10.aig", "i10")
    #testReinforce("./bench/ISCAS/blif/c1355.blif", "c1355")
    #testReinforce("./bench/ISCAS/blif/c7552.blif", "c7552")
