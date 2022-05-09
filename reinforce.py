##
# @file reinforce.py
# @author Keren Zhu
# @date 10/30/2019
# @brief The REINFORCE algorithm
#

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical
import bisect
import random
from dgl.nn.pytorch import GraphConv
import dgl
import multiprocessing as mp
from multiprocessing.dummy import Pool
import random
from env import EnvGraph_mtl_xmg as Env

length_of_command = 9
processes = 4

class GCN(torch.nn.Module):
    def __init__(self, in_feats, hidden_size, out_len):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, hidden_size, allow_zero_in_degree=True)
        self.conv2 = GraphConv(hidden_size, hidden_size, allow_zero_in_degree=True)
        self.conv3 = GraphConv(hidden_size, hidden_size, allow_zero_in_degree=True)
        self.conv4 = GraphConv(hidden_size, out_len, allow_zero_in_degree=True)

    def forward(self, g):
        #print("g_input:", g)
        #print("g_ndata:", g.ndata['feat'])
        h = self.conv1(g, g.ndata['feat'])
        #print("after conv1:", h)
        h = torch.relu(h)
        h = self.conv2(g, h)
        h = torch.relu(h)
        h = self.conv4(g, h)
        g.ndata['h'] = h
        #print("g:", g)
        hg = dgl.mean_nodes(g, 'h')
        #print("hg:", hg)
        #hg: 1*4 tensor
        return torch.squeeze(hg)
        #ndata: Return a node data view for setting/getting node features
        #torch.squeeze: compress tensor and abandon the dim of 1


class FcModel(nn.Module):
    def __init__(self, numFeats, outChs):
        super(FcModel, self).__init__()
        self._numFeats = numFeats
        self._outChs = outChs
        self.fc1 = nn.Linear(numFeats, 32)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(32, 32)
        self.act2 = nn.ReLU()
        self.fc3 = nn.Linear(32, outChs)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        #x = self.fc2(x)
        #x = self.act2(x)
        x = self.fc3(x)
        return x


class FcModelGraph(nn.Module):
    def __init__(self, numFeats, outChs):
        super(FcModelGraph, self).__init__()
        self._numFeats = numFeats
        self._outChs = outChs
        self.fc1 = nn.Linear(numFeats, 32-4)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(32, 32)
        self.act2 = nn.ReLU()
        self.fc3 = nn.Linear(32, outChs)
        #self.gcn = GCN(6, 12, 4)
        self.gcn = GCN(8, 12, 4) # for mig


    def forward(self, x, graph):
        graph_state = self.gcn(graph)
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(torch.cat((x, graph_state), 0)) #joint tensor in dim 0
        x = self.act2(x)
        x = self.fc3(x)
        return x


class PiApprox(object):
    """
    n dimensional continous states
    m discret actions
    """
    def __init__(self, dimStates, numActs, alpha, network):
        """
        @brief approximate policy pi(. | st)
        @param dimStates: Number of dimensions of state space
        @param numActs: Number of the discret actions
        @param alpha: learning rate
        @param network: a pytorch model
        """
        self._dimStates = dimStates
        self._numActs = numActs
        self._alpha = alpha
        self._network = network(dimStates, numActs)
        #self._network.cuda()
        self._optimizer = torch.optim.Adam(self._network.parameters(), alpha, [0.9, 0.999])
        self.tau = 0.5 # temperature for gumbel_softmax

    def __call__(self, s, graph, phaseTrain=True): #s is input, such as state vector
        #print("run PiApprox")
        self._network.eval()
        #s = torch.from_numpy(s).float() #.cuda()
        out = self._network(s, graph)
        #interval = (out.max() - out.min()).data
        #out = (out - out.min().data) / interval
        #normal = self.normalizeLogits(out)
        #probs = F.gumbel_softmax(out, dim=-1, tau = self.tau, hard=True)
        probs = F.softmax(out, dim=-1)
        #w = list(self._network.parameters())
        """
        with open('log', 'a', 0) as outLog:
            line = "logits " + str(out) + "\n" + "action prob " + str(probs) + "\n" 
            outLog.write(line)
        """
        #line = "logits " + str(out) + "\n" + "action prob " + str(probs)
        #print(line)
        if phaseTrain:
            m = Categorical(probs)
            action = m.sample()

            #print("select action:", action.data.item())
            #action = torch.argmax(probs)
        else:
            action = torch.argmax(out)
            #action = torch.argmax(probs)

        return action.data.item()
#self._pi.update(state[0], state[1], action, self._gamma ** tIdx, delta)
    def update(self, s, graph, a, gammaT, delta):
        self._network.train() #switch to train mode
        prob = self._network(s, graph)#.cuda())
        #logProb = -F.gumbel_softmax(prob, dim=-1, tau = self.tau, hard=True)
        logProb = torch.log_softmax(prob, dim=-1)
        #print("logProb:",logProb)
        loss = -gammaT * delta * logProb
        """
        with open('log', 'a', 0) as outLog:
            line = "\n\n\nlogProb " + str(logProb) + '\n' 
            line += "prob " + str(prob) + '\n'
            line += "loss " +str(loss) + '\n'
            line += "action "+ str(a) + '\n'
            line += "gammaT " + str(gammaT) + '\n'
            line += "delta " + str(delta) + '\n'
            outLog.write(line)
        """
        self._optimizer.zero_grad()
        loss[a].backward()
        self._optimizer.step()

    def episode(self):
        #self._tau = self._tau * 0.98
        pass


'''
class Baseline(object):
    """
    The dumbest baseline: a constant for every state
    """
    def __init__(self, b):
        self.b = b

    def __call__(self, s):
        return self.b

    def update(self, s, G):
        pass
'''
class BaselineVApprox(object):
    """
    The baseline with approximation of state value V
    """
    def __init__(self, dimStates, alpha, network):
        """
        @brief approximate policy pi(. | st)
        @param dimStates: Number of dimensions of state space
        @param numActs: Number of the discret actions
        @param alpha: learning rate
        @param network: a pytorch model
        """
        self._dimStates = dimStates
        self._alpha = alpha
        self._network = network(dimStates, 1) #output is set to 1
        #self._network.cuda()
        self._optimizer = torch.optim.Adam(self._network.parameters(), alpha, [0.9, 0.999])
        """
        def initZeroWeights(m):
            if type(m) == nn.Linear:
                m.weight.data.fill_(0.0)
        self._network.apply(initZeroWeights)
        """
    def __call__(self, state):
        self._network.eval()
        return self.value(state).data
    def value(self, state):
        #state = torch.from_numpy(state).float()
        out = self._network(state)
        return out
    def update(self, state, G):
        self._network.train()
        vApprox = self.value(state)
        loss = (torch.tensor([G]) - vApprox[-1]) ** 2 / 2
        # in order to generate calculate graph

        '''with open('log', 'a', 1) as outLog:

            line = "loss" + str(loss) + "\n"
            line += "state " + str(state) + "\n"
            line += "approximate " + str(vApprox) + "\n"
            line += "G " + str(torch.tensor([G])) + "\n"
            outLog.write(line)'''

        self._optimizer.zero_grad()  # to avoid the accumulation of gradient(generally clear for each batch )
        loss.backward()  # to get the gradient form BP
        self._optimizer.step()  # update weight


class Trajectory(object):
    """
    @brief The experience of a trajectory
    """
    def __init__(self, states, rewards, actions, value, env_temp):
        self.states = states
        self.rewards = rewards
        self.actions = actions
        self.value = value
        self.env_temp = env_temp
    def __lt__(self, other):
        return self.value < other.value
class Trajectory_plus(object):
    """
    @brief The experience of a trajectory
    """
    def __init__(self, state0, state1, rewards, actions, value):
        self.state0 = state0
        self.state1 = state1
        self.rewards = rewards
        self.actions = actions
        self.value = value
    def __lt__(self, other):
        return self.value < other.value

def sub_gen(z):
    return genTrajectory_mp(z[0], z[1])
def genTrajectory_mp(pipe_env, pipe_pi):
    # , sync_data_pool, states_pool0, states_pool1
    print("generate Trajectory...")
    # pipe_env = env
    _env = pipe_env
    print("pip_env:", _env)
    _pi = pipe_pi
    print("pip_pi:", _pi)
    _env.reset()  # this is necessary, for the ABCgraph has run resyn2 two times
    state = _env.state()  # generate input state tensor, including state vector and ABC graph
    term = False
    states, rewards, actions = [], [0], []



    while not term:
        print("num of states:", len(states))
        action = _pi(state[0], state[1], True) #_pi is PiApprox actually
        term = _env.takeAction(action)


        nextState = _env.state()
        # return state[0] and state[1]
        nextReward = _env.reward()
        # self.statValue(self._lastStats) - self.statValue(self._curStats) - self._rewardBaseline

        # states[i] = state
        # rewards[i] = nextReward
        # actions[i] = action
        states.append(state)

        #states_pool0.append(state)
        #print(state[1])

        rewards.append(nextReward)
        actions.append(action)
        state = nextState
        #print("\n")

        #print(state)
        if len(states) >= length_of_command:
            term = True
    #print("Generated states:", states)
    #print("rewards:", rewards)
    # with open(log, 'a') as outLog:
    #     line ="\n"
    #     outLog.write(line)

    # print(states)
    # print(rewards)
    # print(actions)
    # print(pipe_env.curStatsValue())


    #state_test.append(state)
    #states_pool0.extend(state_test)
    #print("states:")
    #print(states[-1])
    #print("states_pool1:", states_pool1)
    #print(states1)
    #sync_data_pool.append(states)
    # print("actions:", actions)
    print("End of genTraj------------------------------------------------\n")
    return Trajectory(states, rewards, actions, _env.curStatsValue(), _env)
    # return 0

T_list = []
class Reinforce(object):
    def __init__(self, env, gamma, pi, baseline, ben, filename, envs, process):
        self._env = env
        self._envfile = filename
        self._envCopys = envs
        print(self._envCopys)
        self._gamma = gamma
        self._pi = pi
        self._baseline = baseline
        self.memTrajectory = []  # the memorized trajectories. sorted by value
        self.memLength = 4
        self.sumRewards = []
        self.ben = ben
        self.processes = process
    def genTrajectory(self, phaseTrain=True):
        print("generate Trajectory...")
        self._env.reset()
        pipe_env = self._env
        pipe_env.reset()  # this is necessary, for the ABCgraph has run resyn2 two times
        state = pipe_env.state()  # generate input state tensor, including state vector and ABC graph
        term = False
        states, rewards, actions = [], [0], []
        log = "./results/" + self.ben + "TestRecord-3.csv"

        while not term:
            print("num of states:", len(states))
            action = self._pi(state[0], state[1], phaseTrain) #_pi is PiApprox actually
            term = pipe_env.takeAction(action)


            nextState = pipe_env.state()
            # return state[0] and state[1]
            nextReward = pipe_env.reward()
            # self.statValue(self._lastStats) - self.statValue(self._curStats) - self._rewardBaseline
            states.append(state)
            rewards.append(nextReward)
            actions.append(action)

            state = nextState
            #print("\n")

            if len(states) >= length_of_command:
                term = True


        #print("Generated states:", states)
        #print("rewards:", rewards)
        # with open(log, 'a') as outLog:
        #     line ="\n"
        #     outLog.write(line)
        print(actions)
        print("End of genTraj------------------------------------------------\n")
        #trajectory.value = Trajectory(states, rewards, actions, self._env.curStatsValue())
        # self.memTrajectory.append(Trajectory(states, rewards, actions, pipe_env.curStatsValue()))
        # alist.append(Trajectory(states, rewards, actions, pipe_env.curStatsValue()))
        # T_list.append(Trajectory(states, rewards, actions, pipe_env.curStatsValue()))
        # return Trajectory(states, rewards, actions, self._env.curStatsValue())
    def episode(self, phaseTrain=True, epoch=0):
        #ctx = mp.get_context("spawn")
        #ctx = mp.get_context("fork")
        #trajectory = self.genTrajectory(phaseTrain=phaseTrain, epoch=epoch) # Generate a trajectory of episode of states, actions, rewards

        # for param in self.superpoint.parameters():
        #     param.requires_grad = False
        # self.superpoint.eval()
        #trajectory = mp.Value(Trajectory,)
        #pipe_dict = dict((i, (pipe1, pipe2)) for i in range(processes) for pipe1, pipe2 in (ctx.Pipe(),))

        #sync_data_pool = mp.managers.BaseManager.

        '''sync_data_pool = mp.Manager().list()
        states_pool0 = mp.Manager().list()
        states_pool1 = mp.Manager().list()
        child_process_list = []
        for i in range(processes):
            print('start subprocess:{}'.format(i))
            pro = mp.Process(target=genTrajectory_mp, args=(self._env, self._pi, sync_data_pool, states_pool0, states_pool1,))
            child_process_list.append(pro)

        [p.start() for p in child_process_list]
        [p.join() for p in child_process_list]'''

        #results = [q.get() for j in child_process_list]
        #print(results)
        #print("ok\n")
        #trajectory = q.get()

        self._env.reset()
        pool = Pool(processes=self.processes)

        args = []
        for i in range(self.processes):
            args.append((self._envCopys[i], self._pi))
        res = pool.map(sub_gen, args)
        # self.memTrajectory.append(res)
        pool.close()
        pool.join()
        print(res)
        # for i in res:
        #     print(i.actions)
        #     print(i.rewards)
        #     print(i.states)
        #     print(i.value)
        #     print(i.env_temp.returns)
        #     print(i.env_temp.statValue)
        # input()
        # self.memTrajectory.append(res[-1])
        # print(res)
        '''sum_reward = []
        for i in res:
            sum_reward.append(sum(i.rewards))
            print(i.actions)
            print("\n")
        print(sum_reward, "\n")'''

        # input()
        # print("Subprocess done")
        # print(results1)

        # for res in results1:
        #     print(res)
        #     trajectory = res.get()
        # trajectory = results1[-1]
        # print(trajectory)
        # print("666\n")
        # print(sync_data_pool)
        # #print(states_pool0)
        # print(states_pool0)
        # #print(states_pool0[-1])
        # #print(states_pool1[-1])
        # input()
        # self.memTrajectory = sync_data_pool
        # print(self.memTrajectory)
        # print(self.memTrajectory)
        sum_reward = []
        for i in res:
            sum_reward.append(sum(i.rewards))
        #print(self.memTrajectory[-1].actions)
        #print(sum_reward)
        # print("select worker:", sum_reward.index(max(sum_reward)))
        trajectory = res[sum_reward.index(max(sum_reward))]
        # trajectory = res[random.randint(0, 3)]
        command_sequence = ""
        for i in trajectory.actions:
            command_sequence += str(i)
            command_sequence += " "
        # self.memTrajectory.append(trajectory)
        self._env = trajectory.env_temp
        self.updateTrajectory(trajectory, phaseTrain)

        #print(self.memTrajectory.index(max(sum(self.memTrajectory.rewrads))))
        print("\n")
        # input()
        # self.memTrajectory.clear()
        # print(len(self.memTrajectory))

        '''for i in range(processes):
            print("select worker:", sum_reward.index(max(sum_reward)))
            trajectory = res[sum_reward.index(max(sum_reward))]
            #self._env = trajectory.env_temp
            self.updateTrajectory(trajectory, phaseTrain)
            self._pi.episode()'''



        # self.updateTrajectory(trajectory, phaseTrain)
        # print("Start self._pi.episode()")


        #what is this, seems useless
        #print("End self._pi.episode()")

        return self._env.returns(), command_sequence
        #that is return [self._curStats.numAnd , self._curStats.lev]
    def updateTrajectory(self, trajectory, phaseTrain=True):
        print("UpdateTraj")
        states = trajectory.states
        # print("states:")
        # print(states)
        rewards = trajectory.rewards
        # print("rewards:")
        # print(rewards)
        actions = trajectory.actions
        # print("actions:")
        # print(actions)
        # bisect.insort(self.memTrajectory, trajectory)  # memorize this trajectory
        self.lenSeq = len(states)  # Length of the episode
        for tIdx in range(self.lenSeq):
            G = sum(self._gamma ** (k - tIdx - 1) * rewards[k] for k in range(tIdx + 1, self.lenSeq + 1)) #it is a nice format
            #print("tIdx:", tIdx)
            #print("G:", G)
            state = states[tIdx]
            action = actions[tIdx]
            baseline = self._baseline(state[0])  # get an approximation with an FC model and combined tensor , using BaselineVApprox
            # print("baseline:", baseline)
            delta = G - baseline
            #print("delta:", delta)
            """
            with open('log', 'a', 0) as outLog:
                line = "update " + str(tIdx) + "\n"
                line += "G " + str(G) + "\n"
                line += "baseline " + str(baseline) + "\n"
                outLog.write(line)
            """
            self._baseline.update(state[0], G)
            self._pi.update(state[0], state[1], action, self._gamma ** tIdx, delta)
        sum_of_rewards = sum(rewards)
        self.sumRewards.append(sum_of_rewards)
        print("End of update Trajectory......\nsum_of_rewards:", sum_of_rewards, "\n")

    def replay(self):
        for idx in range(min(self.memLength, int(len(self.memTrajectory) / 10))):
            if len(self.memTrajectory) / 10 < 1:
                return
            upper = min(len(self.memTrajectory) / 10, 30)
            r1 = random.randint(0, upper)
            self.updateTrajectory(self.memTrajectory[idx])
