#!/usr/bin/python
# -*- coding: UTF-8 -*-

from datetime import datetime
import os
import sys, getopt

import reinforce as RF
from env import EnvGraph_mtl_xmg as Env # change this to switch abc/mtl/mtl_xmg

import numpy as np
import statistics


class MtlReturn:
    def __init__(self, returns):
        self.numGates = float(returns[0]) # in fact this is numGates
        self.level = float(returns[1])
    def __lt__(self, other):
        if (int(self.level) == int(other.level)):
            return self.numGates < other.numGates
        else:
            return self.level < other.level
    def __eq__(self, other):
        return int(self.level) == int(other.level) and int(self.numGates) == int(self.numGates)
def takeSecond(elem):
    return elem[0]
def testReinforce(filename, ben, process, brief_name, end2end_target):
    now = datetime.now()
    StartTime = now.strftime("%m/%d/%Y, %H:%M:%S") + "\n"
    print("StartTime ", StartTime)
    env = Env(filename, 0, end2end_target)
    global_baseline_reward = env.baseline_command_sequence_test()
    print("global_baseline_reward_for_each_step", global_baseline_reward)
    processes = process
    print("processes:", processes)
    envCopyList = []
    for i in range(processes):
        envCopyList.append(Env(filename, global_baseline_reward, end2end_target))

    #vApprox = Linear(env.dimState(), env.numActions())

    vApprox = RF.PiApprox(env.dimState(), env.numActions(), 8e-4, RF.FcModelGraph) #dimStates, numActs, alpha, network
    print("dimstate:", env.dimState())
    print("numAction:", env.numActions())
    # print("vApprox:", vApprox)
    # baseline = RF.Baseline(0)
    vbaseline = RF.BaselineVApprox(env.dimState(), 3e-3, RF.FcModel)

    #print("vbaseline:", vbaseline)
    reinforce = RF.Reinforce(env, 0.9, vApprox, vbaseline, ben, filename, envCopyList, processes, brief_name) #env, gamma, pi, baseline

    lastfive = []
    record = []
    resultName = "/home/abcRL2.0-4-24/results/" + ben + "basic-info.csv"
    TestRecordName = "/home/abcRL2.0-4-24/results/" + ben + "detailed-TestRecord.csv"
    for idx in range(200):
        print("Start epoch:", idx)
        returns, command_sequence, mean_rewards, gate_num, latency, energy, row_usage = reinforce.episode(phaseTrain=True, epoch=idx)

        # seqLen = reinforce.lenSeq
        line = "Epoch: " + str(idx) + " returns " + str(returns) + " mean_rewards: " + str(mean_rewards) + "\n"
        record.append(MtlReturn(returns))
        if idx >= 0:
            lastfive.append(MtlReturn(returns))
        print(line)

        with open(TestRecordName, 'a') as tr:
            line = ""
            line += command_sequence
            line += " "
            line += str(lastfive[-1].numGates)
            line += " "
            line += str(reinforce.sumRewards[-1])
            line += " "
            line += str(mean_rewards)
            line += " "
            line += str(gate_num)
            line += " "
            line += str(latency)
            line += " "
            line += str(energy)
            line += " "
            line += str(row_usage)
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
        line += str(lastfive[0].numGates)
        line += " "
        line += str(lastfive[0].level)
        #line += "\n"
        line += "random_test:"
        line += str(env.random_action_test())
        line += " "
        line += "baseline_run_result:"
        line += str(env.baseline_result)
        line += " baseline_double_run_result:"
        line += str(env.baseline_double_run_result)
        line += "\n"
        line += str(StartTime)
        # line += " "
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

    argv_ = sys.argv[1:]
    inputfile = ''
    name = ''
    process = ''
    target = ''
    try:
        opts, args = getopt.getopt(argv_, "hi:n:p:t", ["ifile=", "name=", "process=", "target="])
    except getopt.GetoptError:
        print('testReinforce.py -i <inputfile> -n <name> -p <process> -t <target: gate_num=0, latency=1, energy=2, row_usage=3>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('test.py -i <inputfile> -n <name> -p <process> -t <target: gate_num=0, latency=1, energy=2, row_usage=3>')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg
        elif opt in ("-n", "--name"):
            name = arg
        elif opt in ("-p", "--process"):
            process = arg
        elif opt in ("-t", "--target"):
            target = arg
    print("input file:", inputfile)
    print("name:", name)
    # print("process", process)
    testReinforce(inputfile, name+"_xmg_9steps_"+process+"-in-1-latency-gate-", int(process), name, int(target))

    #i10 c1355 c7552 c6288 c5315 dalu k2 mainpla apex1 bc0
    #testReinforce("./bench/MCNC/Combinational/blif/dalu.blif", "dalu")
    #testReinforce("./bench/MCNC/Combinational/blif/prom1.blif", "prom1")
    #testReinforce("./bench/MCNC/Combinational/blif/mainpla.blif", "mainpla")
    #testReinforce("./bench/MCNC/Combinational/blif/k2.blif", "k2")
    #testReinforce("./bench/MCNC/c1355_syn_out_opt_1.v", "c1355_syn_out_opt_1")

    #testReinforce("/home/lcy/Downloads/MIG_project-main_3_21/MIG_project-main/mult_4_syn_out_opt_1.v", "mult_4")
    #testReinforce("/home/lcy/Downloads/MIG_project-main_3_21/MIG_project-main/mult_64_syn_out_opt_1.v", "mult_64")
    #testReinforce("/home/lcy/Downloads/MIG_project-main_4-16/MIG_project-main/mult_8_syn_out_opt_1.v", "mult_8_xmg_10steps-4-18")
    # testReinforce("/home/MIG_project-main/epfl_max_syn_out_opt_1.v", "epfl_max_xmg_9steps_5-7-v5.0 4-in-1")
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
