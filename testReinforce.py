##
# @file testReinforce.py
# @author Keren Zhu
# @date 10/31/2019
# @brief The main for test REINFORCE
#

from datetime import datetime
import os

import reinforce as RF
from env import EnvGraph_mtl_xmg as Env # change this to switch abc/mtl/mtl_xmg

import numpy as np
import statistics


class AbcReturn:
    def __init__(self, returns):
        self.numNodes = float(returns[0]) # in fact this is numGates
        self.level = float(returns[1])
    def __lt__(self, other):
        if (int(self.level) == int(other.level)):
            return self.numNodes < other.numNodes
        else:
            return self.level < other.level
    def __eq__(self, other):
        return int(self.level) == int(other.level) and int(self.numNodes) == int(self.numNodes)
def takeSecond(elem):
    return elem[0]
def testReinforce(filename, ben):
    now = datetime.now()
    StartTime = now.strftime("%m/%d/%Y, %H:%M:%S") + "\n"
    print("StartTime ", StartTime)
    processes = 4
    envCopyList = []
    for i in range(processes):
        envCopyList.append(Env(filename))
    env = Env(filename)
    #vApprox = Linear(env.dimState(), env.numActions())
    print("Env init ok")
    vApprox = RF.PiApprox(env.dimState(), env.numActions(), 8e-4, RF.FcModelGraph) #dimStates, numActs, alpha, network
    print("dimstate:",env.dimState())
    print("numAction:", env.numActions())
    # print("vApprox:", vApprox)
    # baseline = RF.Baseline(0)
    vbaseline = RF.BaselineVApprox(env.dimState(), 3e-3, RF.FcModel)

    #print("vbaseline:", vbaseline)
    reinforce = RF.Reinforce(env, 0.9, vApprox, vbaseline, ben, filename, envCopyList, processes) #env, gamma, pi, baseline

    lastfive = []
    record = []
    resultName = "./results/" + ben + "basic-info.csv"
    TestRecordName = "./results/" + ben + "detailed-TestRecord.csv"
    for idx in range(200):
        print("Start epoch:", idx)
        returns = reinforce.episode(phaseTrain=True, epoch=idx) # [nodes of Aig, depth of Aig]

        seqLen = reinforce.lenSeq
        line = "iter " + str(idx) + " returns " + str(returns) + " seq Length " + str(seqLen) + "\n"
        record.append(AbcReturn(returns))
        if idx >= 0:
            lastfive.append(AbcReturn(returns))
        print(line)

        with open(TestRecordName, 'a') as tr:
            line = ""
            line += str(lastfive[-1].numNodes)
            line += " "
            line += str(lastfive[-1].level)
            line += " "
            line += str(reinforce.sumRewards[-1])
            line += "\n"
            tr.write(line)


        print("\n\n")
        #reinforce.replay()

    #lastfive.sort(key=lambda x : x.level)
    lastfive = sorted(lastfive)
    #print("lastfive:", lastfive)
    random_test = 0
    rewards = reinforce.sumRewards
    print("rewards:", rewards)

    now = datetime.now()
    EndTime = now.strftime("%m/%d/%Y, %H:%M:%S") + "\n"
    print("EndTime ", EndTime)
    with open(resultName, 'a') as andLog:
        line = "processes: "
        line += str(processes)
        line += " "
        line += "best_num_of_gate: "
        line += str(lastfive[0].numNodes)
        line += " "
        line += str(lastfive[0].level)
        #line += "\n"
        line += "random_test: "
        for i in envCopyList:
            random_test += i.random

        line += str(random_test/processes)
        line += str(StartTime)
        line += " "
        line += str(EndTime)
        andLog.write(line)



    # with open('./results/sum_rewards.csv', 'a') as rewardLog:
    #     line = ""
    #     for idx in range(len(rewards)):
    #         line += str(rewards[idx])
    #         if idx != len(rewards) - 1:
    #             line += ","
    #     line += "\n"
    #     rewardLog.write(line)
    # with open ('./results/converge.csv', 'a') as convergeLog:
    #     line = ""
    #     returns = reinforce.episode(phaseTrain=False)
    #     line += str(returns[0])
    #     line += ","
    #     line += str(returns[1])
    #     line += "\n"
    #     convergeLog.write(line)
    #




if __name__ == "__main__":
    '''
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
    '''
    #i10 c1355 c7552 c6288 c5315 dalu k2 mainpla apex1 bc0
    #testReinforce("./bench/MCNC/Combinational/blif/dalu.blif", "dalu")
    #testReinforce("./bench/MCNC/Combinational/blif/prom1.blif", "prom1")
    #testReinforce("./bench/MCNC/Combinational/blif/mainpla.blif", "mainpla")
    #testReinforce("./bench/MCNC/Combinational/blif/k2.blif", "k2")
    #testReinforce("./bench/MCNC/c1355_syn_out_opt_1.v", "c1355_syn_out_opt_1")

    #testReinforce("/home/lcy/Downloads/MIG_project-main_3_21/MIG_project-main/mult_4_syn_out_opt_1.v", "mult_4")
    #testReinforce("/home/lcy/Downloads/MIG_project-main_3_21/MIG_project-main/mult_64_syn_out_opt_1.v", "mult_64")
    #testReinforce("/home/lcy/Downloads/MIG_project-main_4-16/MIG_project-main/mult_8_syn_out_opt_1.v", "mult_8_xmg_10steps-4-18")
    testReinforce("/home/MIG_project-main/epfl_max_syn_out_opt_1.v", "epfl_max_xmg_9steps_5-7-v5.0 4-in-1")
    #testReinforce("/home/lcy/Downloads/MIG_project-main_3_21/MIG_project-main/div_32_syn_out_opt_1.v", "div_32_xmg_10steps-4-21")
    #testReinforce("/home/lcy/Downloads/MIG_project-main-4-19/MIG_project-main/div_16_syn_out_opt_1.v", "div_16_xmg_9steps-5-3-Release4.0 serial4-in-1")
    #testReinforce("/home/lcy/Downloads/MIG_project-main-4-19/MIG_project-main/add_64_syn_out_opt_1.v", "add_64_xmg_9steps-4-23")
    #testReinforce("/home/lcy/Downloads/MIG_project-main-4-19/MIG_project-main/epfl_priority_syn_out_opt_1.v", "epfl_priority_xmg_9steps-4-23")
    #testReinforce("/home/lcy/Downloads/MIG_project-main-4-19/MIG_project-main/div_32_syn_out_opt_1.v", "div32_xmg_9steps-4-26-mp")
    #testReinforce("/home/lcy/Downloads/MIG_project-main-4-19/MIG_project-main/epfl_voter_syn_out_opt_1.v", "epfl_voter_xmg_9steps-4-23")
    #testReinforce("/home/lcy/Downloads/MIG_project-main_3_21/MIG_project-main/epfl_log2_syn_out_opt_1.v", "epfl_log2_xmg")
    #testReinforce("/home/lcy/Downloads/MIG_project-main_3_21/MIG_project-main/epfl_sin_syn_out_opt_1.v", "epfl_sin_xmg_9steps-4-22")
    #testReinforce("/home/lcy/Downloads/MIG_project-main_3_21/MIG_project-main/epfl_sqrt_syn_out_opt_1.v", "epfl_sqrt_xmg")
    #testReinforce("./bench/MCNC/add_64_syn_out_opt_1.v", "add_64")

    #testReinforce("./bench/MCNC/add_64.v", "add_4_syn_out")
    #testReinforce("./bench/i10.aig", "i10-" + str('test'))
    #testReinforce("./bench/ISCAS/Verilog/c2670.v", "c2670.v")
    #testReinforce("./bench/MCNC/add_4_syn_out.v", "add_4_syn_out")



    #testReinforce("./bench/ISCAS/blif/c5315.blif", "c5315-mig")
    #testReinforce("./bench/ISCAS/blif/c6288.blif", "c6288-mig")
    #testReinforce("./bench/MCNC/Combinational/blif/apex1.blif", "apex1")
    #testReinforce("./bench/MCNC/Combinational/blif/bc0.blif", "bc0")
    '''for i in range(7):
        testReinforce("./bench/i10.aig", "i10-"+str(i+4))'''
    #
    #testReinforce("./bench/ISCAS/blif/c1355.blif", "c1355-mig")
    #testReinforce("./bench/ISCAS/blif/c7552.blif", "c7552-mig")
