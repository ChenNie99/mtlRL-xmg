##
# @file env.py
# @author Keren Zhu
# @date 10/25/2019
# @brief The environment classes
#
import os
import random
import abc_py as abcPy
#from abc_py import abcpy as abcPy
import numpy as np
import graphExtractor as GE
import torch
from datetime import datetime
import dgl
from dgl.nn.pytorch import GraphConv
import time
import mtlPy as mtlpy

length_of_command = 9

class EnvNaive2(object):
    """
    @brief the overall concept of environment, the different. use the compress2rs as target
    """
    def __init__(self, aigfile):
        print("this is AbcInterface")
        self._abc = abcPy.AbcInterface()

        self._aigfile = aigfile
        self._abc.start()
        self.lenSeq = 0
        self._abc.read(self._aigfile)
        initStats = self._abc.aigStats()
        # The initial AIG statistics
        self.initNumAnd = float(initStats.numAnd)
        self.initLev = float(initStats.lev)
        self.resyn2()
        # run a compress2rs as target
        self.resyn2()




        resyn2Stats = self._abc.aigStats()
        totalReward = self.statValue(initStats) - self.statValue(resyn2Stats)
        self._rewardBaseline = totalReward / 10.0 # 18 is the length of compress2rs sequence
        print("baseline num AND ", resyn2Stats.numAnd, " total reward ", totalReward )
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
        self.lastAct = self.numActions() - 1
        self.lastAct2 = self.numActions() - 1
        self.lastAct3 = self.numActions() - 1
        self.lastAct4 = self.numActions() - 1
        self.actsTaken = np.zeros(self.numActions())
        return self.state()
    def close(self):
        self.reset()
    def step(self, actionIdx):
        self.takeAction(actionIdx)
        nextState = self.state()
        reward = self.reward()
        done = False
        if (self.lenSeq >= 10):
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
        """
        # Compress2rs actions
        if actionIdx == 0:
            self._abc.balance(l=True) # b -l
        elif actionIdx == 1:
            self._abc.resub(k=6, l=True) # rs -K 6 -l
        #elif actionIdx == 2:
        #    self._abc.resub(k=6, n=2, l=True) # rs -K 6 -N 2 -l
        #elif actionIdx == 3:
        #    self._abc.resub(k=8, l=True) # rs -K 8 -l
        #elif actionIdx == 4:
        #    self._abc.resub(k=10, l=True) # rs -K 10 -l
        #elif actionIdx == 5:
        #    self._abc.resub(k=12, l=True) # rs -K 12 -l
        #elif actionIdx == 6:
        #    self._abc.resub(k=10, n=2, l=True) # rs -K 10 -N 2 -l
        elif actionIdx == 2:
            self._abc.resub(k=12, n=2, l=True) # rs - K 12 -N 2 -l
        elif actionIdx == 3:
            self._abc.rewrite(l=True) # rw -l
        #elif actionIdx == 3:
        #    self._abc.rewrite(l=True, z=True) # rwz -l
        elif actionIdx == 4:
            self._abc.refactor(l=True) # rf -l
        #elif actionIdx == 4:
        #    self._abc.refactor(l=True, z=True) # rfz -l
        elif actionIdx == 5: # terminal
            self._abc.end()
            return True
        else:
            assert(False)
        """
        if actionIdx == 0:
            self._abc.balance(l=False) # b
        elif actionIdx == 1:
            self._abc.rewrite(l=False) # rw
        elif actionIdx == 2:
            self._abc.refactor(l=False) # rf
        elif actionIdx == 3:
            self._abc.rewrite(l=False, z=True) #rw -z
        elif actionIdx == 4:
            self._abc.refactor(l=False, z=True) #rs
        elif actionIdx == 5:
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
        self._curStats = self._abc.aigStats()
        return False
    def state(self):
        """
        @brief current state
        """
        oneHotAct = np.zeros(self.numActions())
        np.put(oneHotAct, self.lastAct, 1)
        lastOneHotActs  = np.zeros(self.numActions())
        lastOneHotActs[self.lastAct2] += 1/3
        lastOneHotActs[self.lastAct3] += 1/3
        lastOneHotActs[self.lastAct] += 1/3
        stateArray = np.array([self._curStats.numAnd / self.initNumAnd, self._curStats.lev / self.initLev,
            self._lastStats.numAnd / self.initNumAnd, self._lastStats.lev / self.initLev])
        stepArray = np.array([float(self.lenSeq) / 10.0])
        combined = np.concatenate((stateArray, lastOneHotActs, stepArray), axis=-1)
        #combined = np.expand_dims(combined, axis=0)
        #return stateArray.astype(np.float32)
        return torch.from_numpy(combined.astype(np.float32)).float()
    def reward(self):
        if self.lastAct == 5: #term
            return 0
        return self.statValue(self._lastStats) - self.statValue(self._curStats) - self._rewardBaseline
        #return -self._lastStats.numAnd + self._curStats.numAnd - 1
        if (self._curStats.numAnd < self._lastStats.numAnd and self._curStats.lev < self._lastStats.lev):
            return 2
        elif (self._curStats.numAnd < self._lastStats.numAnd and self._curStats.lev == self._lastStats.lev):
            return 0
        elif (self._curStats.numAnd == self._lastStats.numAnd and self._curStats.lev < self._lastStats.lev):
            return 1
        else:
            return -2
    def numActions(self):
        return 5
    def dimState(self):
        return 4 + self.numActions() * 1 + 1
    def returns(self):
        return [self._curStats.numAnd , self._curStats.lev]
    def statValue(self, stat):
        return float(stat.numAnd)  / float(self.initNumAnd) #  + float(stat.lev)  / float(self.initLev)
        #return stat.numAnd + stat.lev * 10
    def curStatsValue(self):
        return self.statValue(self._curStats)
    def seed(self, sd):
        pass
    def compress2rs(self):
        self._abc.compress2rs()






class EnvGraph(object):
    """
    @brief the overall concept of environment, the different. use the compress2rs as target
    """
    def __init__(self, aigfile):
        #self._abc = mtlpy.MtlInterface()
        self._abc = abcPy.AbcInterface()
        self._aigfile = aigfile
        self._abc.start()
        self.lenSeq = 0
        self._abc.read(self._aigfile)
        initStats = self._abc.aigStats() # The initial AIG statistics
        self.initNumAnd = float(initStats.numAnd)
        self.initLev = float(initStats.lev)

        now = datetime.now()
        dateTime = now.strftime("%m/%d/%Y, %H:%M:%S") + "\n"
        print("TestStartTime for 100 runs of get aig state", dateTime)
        for i in range(10000):
            initStats = self._abc.aigStats()
            self.initNumAnd = float(initStats.numAnd)
            self.initLev = float(initStats.lev)
        now = datetime.now()
        dateTime = now.strftime("%m/%d/%Y, %H:%M:%S") + "\n"
        print("TestEndTime ", dateTime)

        print("Inintial state:")
        print("initNumAnd", self.initNumAnd, "baseline Depth ", self.initLev, "Initial value", self.statValue(initStats))
        self.resyn2() # run a compress2rs as target
        print("After run of resyn2:")
        resynStats = self._abc.aigStats()
        print("baseline num AND ", resynStats.numAnd, "baseline Depth ", resynStats.lev, " total reward ", self.statValue(resynStats))
        self.resyn2()
        resyn2Stats = self._abc.aigStats()
        totalReward = self.statValue(initStats) - self.statValue(resyn2Stats)
        self._rewardBaseline = totalReward / 10.0 # 18 is the length of compress2rs sequence
        print("After two runs of resyn2:")
        print("baseline num AND ", resyn2Stats.numAnd, "baseline Depth ", resyn2Stats.lev, " total reward ", totalReward)
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
        self.lastAct = self.numActions() - 1
        self.lastAct2 = self.numActions() - 1
        self.lastAct3 = self.numActions() - 1
        self.lastAct4 = self.numActions() - 1
        self.actsTaken = np.zeros(self.numActions())
        return self.state()
    def close(self):
        self.reset()
    def step(self, actionIdx):
        self.takeAction(actionIdx)
        nextState = self.state()
        reward = self.reward()
        done = False
        if (self.lenSeq >= 10):
            done = True
        return nextState, reward, done, 0
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
        """
        # Compress2rs actions
        if actionIdx == 0:
            self._abc.balance(l=True) # b -l
        elif actionIdx == 1:
            self._abc.resub(k=6, l=True) # rs -K 6 -l
        #elif actionIdx == 2:
        #    self._abc.resub(k=6, n=2, l=True) # rs -K 6 -N 2 -l
        #elif actionIdx == 3:
        #    self._abc.resub(k=8, l=True) # rs -K 8 -l
        #elif actionIdx == 4:
        #    self._abc.resub(k=10, l=True) # rs -K 10 -l
        #elif actionIdx == 5:
        #    self._abc.resub(k=12, l=True) # rs -K 12 -l
        #elif actionIdx == 6:
        #    self._abc.resub(k=10, n=2, l=True) # rs -K 10 -N 2 -l
        elif actionIdx == 2:
            self._abc.resub(k=12, n=2, l=True) # rs - K 12 -N 2 -l
        elif actionIdx == 3:
            self._abc.rewrite(l=True) # rw -l
        #elif actionIdx == 3:
        #    self._abc.rewrite(l=True, z=True) # rwz -l
        elif actionIdx == 4:
            self._abc.refactor(l=True) # rf -l
        #elif actionIdx == 4:
        #    self._abc.refactor(l=True, z=True) # rfz -l
        elif actionIdx == 5: # terminal
            self._abc.end()
            return True
        else:
            assert(False)
        """
        if actionIdx == 0:
            self._abc.balance(l=False) # b
        elif actionIdx == 1:
            self._abc.rewrite(l=False) # rw
        elif actionIdx == 2:
            self._abc.refactor(l=False) # rf
        elif actionIdx == 3:
            self._abc.rewrite(l=False, z=True) #rw -z
        elif actionIdx == 4:
            self._abc.refactor(l=False, z=True) #rs
        elif actionIdx == 5:
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
        self._curStats = self._abc.aigStats()
        return False
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
        stateArray = np.array([self._curStats.numAnd / self.initNumAnd, self._curStats.lev / self.initLev,
            self._lastStats.numAnd / self.initNumAnd, self._lastStats.lev / self.initLev])
        stepArray = np.array([float(self.lenSeq) / 10.0])
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
        graph = GE.extract_dgl_graph(self._abc)
        #print("input graph:", graph)
        return (combined_torch, graph)
    def reward(self):

        if self.lastAct == 5: #term
            return 0

        #print("lastStats:", self.statValue(self._lastStats), "curStats:", self.statValue(self._curStats),"rewardBaseline:", self._rewardBaseline)
        #print("final rewards:", self.statValue(self._lastStats) - self.statValue(self._curStats) - self._rewardBaseline)
        return self.statValue(self._lastStats) - self.statValue(self._curStats) - self._rewardBaseline
        #return -self._lastStats.numAnd + self._curStats.numAnd - 1
        if (self._curStats.numAnd < self._lastStats.numAnd and self._curStats.lev < self._lastStats.lev):
            return 2
        elif (self._curStats.numAnd < self._lastStats.numAnd and self._curStats.lev == self._lastStats.lev):
            return 0
        elif (self._curStats.numAnd == self._lastStats.numAnd and self._curStats.lev < self._lastStats.lev):
            return 1
        else:
            return -2
    def numActions(self):
        return 5
    def dimState(self):
        return 4 + self.numActions() * 1 + 1
    def returns(self):
        return [self._curStats.numAnd , self._curStats.lev]
    def statValue(self, stat):
        #return float(stat.lev)  / float(self.initLev)
        return float(stat.numAnd) / float(self.initNumAnd) #  + float(stat.lev)  / float(self.initLev)
        #return stat.numAnd + stat.lev * 10
    def curStatsValue(self):
        return self.statValue(self._curStats)
    def seed(self, sd):
        pass
    def compress2rs(self):
        self._abc.compress2rs()




class EnvGraph_mtl(object):
    """
    @brief the overall concept of environment, the different. use the compress2rs as target
    """
    def __init__(self, migfile):
        self._abc = mtlpy.MtlInterface()
        #self._abc = abcPy.AbcInterface()
        self._migfile = migfile

        self._abc.start()
        self.lenSeq = 0
        self._abc.read(self._migfile)

        initStats = self._abc.migStats() # The initial MIG statistics
        self.initNumMigNodes = float(initStats.numMigNodes)
        self.initLev = float(initStats.lev)






        print("Inintial state:")
        print("initNumMigNodes", self.initNumMigNodes, "baseline Depth ", self.initLev, "Initial value", self.statValue(initStats))
        self.baselineActions_bl10()
        print("After run of resyn2:")
        resynStats = self._abc.migStats()
        print("baseline num of MigNodes ", resynStats.numMigNodes, "baseline Depth ", resynStats.lev, " total reward ", self.statValue(resynStats))
        self.baselineActions_bl10()
        resyn2Stats = self._abc.migStats()
        totalReward = self.statValue(initStats) - self.statValue(resynStats)
        self._rewardBaseline = totalReward / 9.0 # 18 is the length of compress2rs sequence
        print("After two runs of resyn2:")
        print("baseline num of numMigNodes ", resyn2Stats.numMigNodes, "baseline Depth ", resyn2Stats.lev, " total reward ", totalReward)
    def baselineActions_bl10(self):
        #self._abc.rewrite()
        self._abc.balance()
        self._abc.rewrite()
        self._abc.balance()
        self._abc.rewrite()
        self._abc.balance()
        self._abc.rewrite()
        self._abc.balance()
        self._abc.rewrite()
        self._abc.balance()
        self._abc.rewrite()

    def reset(self):
        self.lenSeq = 0
        self._abc.end()
        self._abc.start()
        self._abc.read(self._migfile)
        self._lastStats = self._abc.migStats() # The initial AIG statistics
        self._curStats = self._lastStats # the current AIG statistics
        self.lastAct = self.numActions() - 1
        self.lastAct2 = self.numActions() - 1
        self.lastAct3 = self.numActions() - 1
        self.lastAct4 = self.numActions() - 1
        self.lastAct5 = self.numActions() - 1
        self.actsTaken = np.zeros(self.numActions())
        return self.state()
    def close(self):
        self.reset()
    def step(self, actionIdx):
        self.takeAction(actionIdx)
        nextState = self.state()
        reward = self.reward()
        done = False
        if (self.lenSeq >= 10):
            done = True
        return nextState, reward, done, 0
    def takeAction(self, actionIdx):
        """
        @return true: episode is end
        """
        # "b -l; rs -K 6 -l; rw -l; rs -K 6 -N 2 -l; rf -l; rs -K 8 -l; b -l; rs -K 8 -N 2 -l; rw -l; rs -K 10 -l; rwz -l; rs -K      10 -N 2 -l; b -l; rs -K 12 -l; rfz -l; rs -K 12 -N 2 -l; rwz -l; b -l
        self.lastAct5 = self.lastAct4
        self.lastAct4 = self.lastAct3
        self.lastAct3 = self.lastAct2
        self.lastAct2 = self.lastAct
        self.lastAct = actionIdx
        #self.actsTaken[actionIdx] += 1
        self.lenSeq += 1
        """
        # Compress2rs actions
        if actionIdx == 0:
            self._abc.balance(l=True) # b -l
        elif actionIdx == 1:
            self._abc.resub(k=6, l=True) # rs -K 6 -l
        #elif actionIdx == 2:
        #    self._abc.resub(k=6, n=2, l=True) # rs -K 6 -N 2 -l
        #elif actionIdx == 3:
        #    self._abc.resub(k=8, l=True) # rs -K 8 -l
        #elif actionIdx == 4:
        #    self._abc.resub(k=10, l=True) # rs -K 10 -l
        #elif actionIdx == 5:
        #    self._abc.resub(k=12, l=True) # rs -K 12 -l
        #elif actionIdx == 6:
        #    self._abc.resub(k=10, n=2, l=True) # rs -K 10 -N 2 -l
        elif actionIdx == 2:
            self._abc.resub(k=12, n=2, l=True) # rs - K 12 -N 2 -l
        elif actionIdx == 3:
            self._abc.rewrite(l=True) # rw -l
        #elif actionIdx == 3:
        #    self._abc.rewrite(l=True, z=True) # rwz -l
        elif actionIdx == 4:
            self._abc.refactor(l=True) # rf -l
        #elif actionIdx == 4:
        #    self._abc.refactor(l=True, z=True) # rfz -l
        elif actionIdx == 5: # terminal
            self._abc.end()
            return True
        else:
            assert(False)
        """
        if actionIdx == 0:
            self._abc.rewrite() # performs basic rewriting of MIG
        elif actionIdx == 1:
            self._abc.rewrite(use_dont_cares=True) # rewriting while using don't cares
        elif actionIdx == 2:
            self._abc.rewrite(allow_zero_gain=True) # rewriting while allowing 0 gain substitution
        elif actionIdx == 3:
            self._abc.rewrite(use_dont_cares=True, allow_zero_gain=True) # rewriting using don't cares with 0 gain substitutions
        elif actionIdx == 4:
            self._abc.balance() # performs basic balancing of the MIG
        elif actionIdx == 5:
            self._abc.balance(crit=True) # balancing only the critical path of MIG
        elif actionIdx == 6:
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
        self._curStats = self._abc.migStats()
        return False
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
        stateArray = np.array([self._curStats.numMigNodes / self.initNumMigNodes, self._curStats.lev / self.initLev,
            self._lastStats.numMigNodes / self.initNumMigNodes, self._lastStats.lev / self.initLev])
        stepArray = np.array([float(self.lenSeq) / 10.0])
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
        graph = GE.extract_dgl_graph_mig(self._abc)
        #print("input graph:", graph)
        return (combined_torch, graph)
    def reward(self):

        if self.lastAct == 6: #term
            return 0

        #print("lastStats:", self.statValue(self._lastStats), "curStats:", self.statValue(self._curStats),"rewardBaseline:", self._rewardBaseline)
        #print("final rewards:", self.statValue(self._lastStats) - self.statValue(self._curStats) - self._rewardBaseline)
        return self.statValue(self._lastStats) - self.statValue(self._curStats) - self._rewardBaseline
        #return -self._lastStats.numAnd + self._curStats.numAnd - 1
        if (self._curStats.numAnd < self._lastStats.numAnd and self._curStats.lev < self._lastStats.lev):
            return 2
        elif (self._curStats.numAnd < self._lastStats.numAnd and self._curStats.lev == self._lastStats.lev):
            return 0
        elif (self._curStats.numAnd == self._lastStats.numAnd and self._curStats.lev < self._lastStats.lev):
            return 1
        else:
            return -2
    def numActions(self):
        return 6
    def dimState(self):
        return 4 + self.numActions() * 1 + 1
    def returns(self):
        return [self._curStats.numMigNodes, self._curStats.lev]
    def statValue(self, stat):
        #return float(stat.lev)  / float(self.initLev)
        return float(stat.numMigNodes) / float(self.initNumMigNodes) #  + float(stat.lev)  / float(self.initLev)
        #return stat.numAnd + stat.lev * 10
    def curStatsValue(self):
        return self.statValue(self._curStats)
    def seed(self, sd):
        pass
    def compress2rs(self):
        self._abc.compress2rs()


class EnvGraph_mtl_xmg(object):
    """
    @brief the overall concept of environment, the different. use the compress2rs as target
    """
    def __init__(self, xmgfile):
        self._abc = mtlpy.MtlInterface()
        #self._abc = abcPy.AbcInterface()
        self._xmgfile = xmgfile


        self._abc.xmg_start()
        print("read start ok")
        self.lenSeq = 0
        self._abc.xmg_read(self._xmgfile)
        print("read xmgfile ok")
        initStats = self._abc.xmgStats() # The initial XMG statistics
        self.initNumXmgNodes = float(initStats.numXmgNodes)
        self.initNumXmgGates = float(initStats.numXmgGates)
        self.initLev = float(initStats.xmg_lev)






        # print("Inintial state:")
        # print("initNumXmgNodes", self.initNumXmgNodes)
        # print("initNumXmgGates", self.initNumXmgGates)
        # print("baseline Depth", self.initLev)
        print("Initial value", self.statValue(initStats))

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
        totalReward = self.statValue(initStats) - self.statValue(resyn2Stats)
        self._rewardBaseline = totalReward / length_of_command # 10 is the length of compress2rs sequence
        print("After double runs of baseline:")
        print("baseline num of XmgNodes ", resyn2Stats.numXmgNodes,
              "baseline num of XmgGates ", float(resyn2Stats.numXmgGates),
              " total reward ", self.statValue(resyn2Stats))
        # print("test the runtime for each action")

        # self.random = self.random_action_test()
        #input()
        #os.system("pause")
        #self.reset()
        #self.test_action_runtime_2()
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
        self._curStats = self._lastStats # the current XMG statistics
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
        self._curStats = self._abc.xmgStats()
        return False
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
        stateArray = np.array([self._curStats.numXmgNodes / self.initNumXmgNodes, self._curStats.xmg_lev / self.initLev,
            self._lastStats.numXmgNodes / self.initNumXmgNodes, self._lastStats.xmg_lev / self.initLev])
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

        return (combined_torch, graph)
    def reward(self):

        if self.lastAct == 8: #term
            return 0

        #print("lastStats:", self.statValue(self._lastStats), "curStats:", self.statValue(self._curStats),"rewardBaseline:", self._rewardBaseline)
        #print("final rewards:", self.statValue(self._lastStats) - self.statValue(self._curStats) - self._rewardBaseline)
        return self.statValue(self._lastStats) - self.statValue(self._curStats) - self._rewardBaseline
        #return -self._lastStats.numAnd + self._curStats.numAnd - 1
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
