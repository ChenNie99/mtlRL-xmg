import os
import random
from perf_analysis import xmg_evaluation

import numpy as np
import graphExtractor as GE
import torch
from datetime import datetime
import dgl
from dgl.nn.pytorch import GraphConv
import time
import mtlPy as mtlpy

length_of_command = 9

class EnvGraph_mtl_xmg(object):
    """
    @brief the overall concept of environment, the different. use the compress2rs as target
    """
    def __init__(self, xmgfile, rewardBaseline, end2end_target):
        self._abc = mtlpy.MtlInterface()
        #self._abc = abcPy.AbcInterface()
        self._xmgfile = xmgfile
        
        self.target_end2end_index = end2end_target
        # gate_num=0, latency=1, energy=2, row_usage=3

        self._abc.xmg_start()
        print("read start ok")
        self.lenSeq = 0
        self._abc.xmg_read(self._xmgfile)
        print("read xmgfile ok")
        initStats = self._abc.xmgStats() # The initial XMG statistics
        self.initNumXmgNodes = float(initStats.numXmgNodes)
        self.initNumXmgGates = float(initStats.numXmgGates)
        self.initLev = float(initStats.xmg_lev)
        self._rewardBaseline = rewardBaseline
        # print("Inintial state:")
        # print("initNumXmgNodes", self.initNumXmgNodes)
        # print("initNumXmgGates", self.initNumXmgGates)
        # print("baseline Depth", self.initLev)
        print("Initial value", self.statValue(initStats))

        self.end_to_end_result = self.get_end2end_states()
        print("Initial end2end result", self.end_to_end_result)
        self.baselineActions()
        self.baseline_end2end_result = self.get_end2end_states()
        self._rewardBaseline = []
        for i in range(5):
            self._rewardBaseline.append((1 - self.baseline_end2end_result[i]/self.end_to_end_result[i])/length_of_command)

        
        # print("test the runtime for each action")

        # self.random = self.random_action_test()
        #input()
        #os.system("pause")
        #self.reset()
        #self.test_action_runtime_2()
    def baseline_command_sequence_test(self):
        initStats = self._abc.xmgStats()
        self.baselineActions()
        print("After run of baseline:")
        resynStats = self._abc.xmgStats()
        print("baseline num of XmgNodes ",
              resynStats.numXmgNodes,
              "baseline num of XmgGates ",
              float(resynStats.numXmgGates),
              " total reward ", self.statValue(resynStats))
        self.baselineActions()
        resyn2Stats = self._abc.xmgStats()
        self.baseline_result = resynStats.numXmgGates
        self.baseline_double_run_result = resyn2Stats.numXmgGates
        totalReward = self.statValue(initStats) - self.statValue(resyn2Stats)
        # note that state value is normalized
        self._rewardBaseline = totalReward / length_of_command  # 9 is the length of compress2rs sequence
        print("After double runs of baseline:")
        print("baseline num of XmgNodes ", resyn2Stats.numXmgNodes,
              "baseline num of XmgGates ", float(resyn2Stats.numXmgGates),
              " total reward ", self.statValue(resyn2Stats))
        self.reset()
        return self._rewardBaseline
    def random_action_test(self):

        act_list = [0, 1, 2, 3, 4, 5, 6, 7]
        sum = 0
        epoch = 10
        for j in range(epoch):
            self.reset()
            for i in range(9):
                select = random.choice(act_list)
                self.takeAction(select)
            resyn2Stats = self._abc.xmgStats()
            print(resyn2Stats.numXmgNodes)
            sum += resyn2Stats.numXmgNodes



        print("average num of random action", sum/epoch)
        return sum/epoch
    def test_action_runtime(self):
        starttime = time.time()
        for i in range(1):
            self._abc.xmg_dco()
        endtime = time.time()
        print("xmg_dco runtime:%s ms", (endtime - starttime)*1000)

        starttime = time.time()
        for i in range(1):
            self._abc.xmg_resub()
        endtime = time.time()
        print("xmg_resub runtime:%s ms", (endtime - starttime) * 1000)

        starttime = time.time()
        for i in range(1):
            self._abc.xmg_cut_rewrite()
        endtime = time.time()
        print("xmg_cut_rewrite runtime:%s ms", (endtime - starttime) * 1000)

        starttime = time.time()
        for i in range(1):
            self._abc.xmg_depth_rewrite(allow_size_increase=True, start='a', overhead=1.2)
        endtime = time.time()
        print("xmg_depth_rewrite(allow_size_increase=True, start='a', overhead=1.2) runtime:%s ms", (endtime - starttime) * 1000)

        starttime = time.time()
        for i in range(1):
            self._abc.xmg_node_resynthesis()
        endtime = time.time()
        print("xmg_node_resyn runtime:%s ms", (endtime - starttime) * 1000)

        starttime = time.time()
        for i in range(1):
            self._abc.xmg_depth_rewrite(start='s', allow_size_increase=False)
        endtime = time.time()
        print("xmg_depth_rewrite(start='s', allow_size_increase=False) runtime:%s ms", (endtime - starttime) * 1000)

        os.system("pause")

    def test_action_runtime_2(self):
        starttime = datetime.now()
        for i in range(10):
            self._abc.xmg_dco()
        endtime = datetime.now()
        print("xmg_dco runtime:%s ms", (endtime - starttime))

        starttime = datetime.now()
        for i in range(10):
            self._abc.xmg_resub()
        endtime = datetime.now()
        print("xmg_resub runtime:%s ms", (endtime - starttime))

        starttime = datetime.now()
        for i in range(10):
            self._abc.xmg_cut_rewrite()
        endtime = datetime.now()
        print("xmg_cut_rewrite runtime:%s ms", (endtime - starttime))

        starttime = datetime.now()
        for i in range(10):
            self._abc.xmg_depth_rewrite(allow_size_increase=True, start='a', overhead=1.2)
        endtime = datetime.now()
        print("xmg_depth_rewrite(allow_size_increase=True, start='a', overhead=1.2) runtime:%s ms", (endtime - starttime))

        starttime = datetime.now()
        for i in range(10):
            self._abc.xmg_node_resynthesis()
        endtime = datetime.now()
        print("xmg_node_resyn runtime:%s ms", (endtime - starttime))

        starttime = datetime.now()
        for i in range(10):
            self._abc.xmg_depth_rewrite(start='s', allow_size_increase=False)
        endtime = datetime.now()
        print("xmg_depth_rewrite(start='s', allow_size_increase=False) runtime:%s ms", (endtime - starttime))

        os.system("pause")
        input()
    def baselineActions(self):

        self._abc.xmg_resub()
        self._abc.xmg_cut_rewrite(cut_size=4)
        self._abc.xmg_cut_rewrite(cut_size=3)
        self._abc.xmg_resub()
        self._abc.xmg_node_resynthesis(cut_size=3)
        self._abc.xmg_resub()
        self._abc.xmg_cut_rewrite(cut_size=4)
        self._abc.xmg_cut_rewrite(cut_size=3)
        self._abc.xmg_resub()


    def reset(self):
        self.lenSeq = 0
        self._abc.xmg_end()
        self._abc.xmg_start()
        self._abc.xmg_read(self._xmgfile)
        self._lastStats = self._abc.xmgStats() # The initial XMG statistics
        self._lastStats_end2end = self.end_to_end_result
        self._curStats = self._lastStats # the current XMG statistics
        self._curStats_end2end = self._lastStats_end2end
        self.lastAct = self.numActions() - 1
        self.lastAct2 = self.numActions() - 1
        self.lastAct3 = self.numActions() - 1
        self.lastAct4 = self.numActions() - 1
        self.lastAct5 = self.numActions() - 1
        self.lastAct6 = self.numActions() - 1
        self.lastAct7 = self.numActions() - 1
        self.actsTaken = np.zeros(self.numActions())
        return self.state()

    def close(self):
        self.reset()
    def step(self, actionIdx):
        self.takeAction(actionIdx)
        nextState = self.state()
        reward = self.reward()
        done = False
        if (self.lenSeq >= length_of_command):
            done = True
        return nextState, reward, done, 0
    def takeAction(self, actionIdx):
        """
        @return true: episode is end
        """
        # "b -l; rs -K 6 -l; rw -l; rs -K 6 -N 2 -l; rf -l; rs -K 8 -l; b -l; rs -K 8 -N 2 -l; rw -l; rs -K 10 -l; rwz -l; rs -K      10 -N 2 -l; b -l; rs -K 12 -l; rfz -l; rs -K 12 -N 2 -l; rwz -l; b -l
        self.lastAct7 = self.lastAct6
        self.lastAct6 = self.lastAct5
        self.lastAct5 = self.lastAct4
        self.lastAct4 = self.lastAct3
        self.lastAct3 = self.lastAct2
        self.lastAct2 = self.lastAct
        self.lastAct = actionIdx
        #self.actsTaken[actionIdx] += 1
        self.lenSeq += 1

        if actionIdx == 0:
            self._abc.xmg_depth_rewrite(allow_size_increase=True, start='a', overhead=1.2)
        elif actionIdx == 1:
            self._abc.xmg_depth_rewrite(start='s', allow_size_increase=False)
        elif actionIdx == 2:
            self._abc.xmg_resub()
        elif actionIdx == 3:
            self._abc.xmg_node_resynthesis(cut_size=3)
        elif actionIdx == 4:
            self._abc.xmg_node_resynthesis(cut_size=4)
        elif actionIdx == 5:
            self._abc.xmg_cut_rewrite(cut_size=2)
        elif actionIdx == 6:
            self._abc.xmg_cut_rewrite(cut_size=3)
        elif actionIdx == 7:
            self._abc.xmg_cut_rewrite(cut_size=4)
        elif actionIdx == 8:
            self._abc.end()
            return True
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
        self._lastStats_end2end = self._curStats_end2end
        self._curStats = self._abc.xmgStats()
        self._curStats_end2end = self.get_end2end_states()
        return False
    def get_end2end_states(self):
        self.write_verilog("temp.v")
        command_temp = "./converter -d --xor3 temp.v"
        os.system(command_temp)
        command_temp2 = "python3 map_priority.py temp_temp.txt > temp_map.txt"
        # max_mem = map_priority(str(self.brief_name) +"_syn_out_opt_1_temp.txt", self.brief_name)
        os.system(command_temp2)
        return xmg_evaluation("temp_map.txt")

    def state(self):
        """
        @brief current state
        """
        oneHotAct = np.zeros(self.numActions()) #self.numActions=5
        #print("self.lastAct:", self.lastAct)
        np.put(oneHotAct, self.lastAct, 1) #oneHotAct[self.lastAct]=1
        lastOneHotActs  = np.zeros(self.numActions())
        lastOneHotActs[self.lastAct2] += 1/3
        lastOneHotActs[self.lastAct3] += 1/3
        lastOneHotActs[self.lastAct] += 1/3
        lastOneHotActs[self.lastAct4] += 1/3
        lastOneHotActs[self.lastAct5] += 1/3
        stateArray = np.array(self._curStats_end2end[self.target_end2end_index]/self.end_to_end_result[self.target_end2end_index],self._lastStats_end2end[self.target_end2end_index]/self.end_to_end_result[self.target_end2end_index])
        stepArray = np.array([float(self.lenSeq) / length_of_command])
        combined = np.concatenate((stateArray, lastOneHotActs, stepArray), axis=-1)
        # print("combined Input state :", combined)
        # combined = np.expand_dims(combined, axis=0)
        # return stateArray.astype(np.float32)
        combined_torch = torch.from_numpy(combined.astype(np.float32)).float()
        # combine_torch share the same memary with combined
        # print("GE\n", datetime.now())
        '''for i in range(100):
            graph = GE.extract_dgl_graph(self._abc)
        print("endGE", datetime.now())'''
        graph = GE.extract_dgl_graph_xmg(self._abc)
        #print(graph)
        #print("input graph:", graph)
        #combined_torch.requires_grad = False

        return (combined_torch , graph)
    def write_verilog(self, filename):
        self._abc.write_verilog(filename)
    def reward(self):
        if self.lastAct == 8:  # term
            return 0

        gate_num, latency, energy, row_usage = self.get_end2end_states()
        weighted_reward = (self._lastStats_end2end[self.target_end2end_index] - self._curStats_end2end[self.target_end2end_index])/self.end_to_end_result[self.target_end2end_index] - self._rewardBaseline[self.target_end2end_index]
        # print("lastStats:", self.statValue(self._lastStats), "curStats:", self.statValue(self._curStats),"rewardBaseline:", self._rewardBaseline)
        # print("final rewards:", self.statValue(self._lastStats) - self.statValue(self._curStats) - self._rewardBaseline)
        # old_reward = self.statValue(self._lastStats) - self.statValue(self._curStats) - self._rewardBaseline
        # note that statValue is normalized
        return weighted_reward
        # return self._lastStats.numAnd + self._curStats.numAnd - 1
        if (self._curStats.numAnd < self._lastStats.numAnd and self._curStats.xmg_lev < self._lastStats.xmg_lev):
            return 2
        elif (self._curStats.numAnd < self._lastStats.numAnd and self._curStats.xmg_lev == self._lastStats.xmg_lev):
            return 0
        elif (self._curStats.numAnd == self._lastStats.numAnd and self._curStats.xmg_lev < self._lastStats.xmg_lev):
            return 1
        else:
            return -2
    def numActions(self):
        return 8
    def dimState(self):
        return 4 + self.numActions() * 1 + 1
    def returns(self):
        return [self._curStats.numXmgGates, self._curStats.xmg_lev]
    def statValue(self, stat):
        #return float(stat.xmg_lev)  / float(self.initLev)
        return float(stat.numXmgNodes) / float(self.initNumXmgNodes) #  + float(stat.xmg_lev)  / float(self.initLev)
        #return stat.numAnd + stat.xmg_lev * 10
    def curStatsValue(self):
        return self.statValue(self._curStats)
    def seed(self, sd):
        pass
    def compress2rs(self):
        self._abc.compress2rs()
