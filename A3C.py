import multiprocessing as mp
import torch
import configparser
import numpy as np

ctx = mp.get_context("spawn")
config = configparser.ConfigParser()
config.read("hyperConfig.conf", encoding="utf-8")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 超参数
BATCH_SIZE = config.getint('HYPERPARA','BATCH_SIZE')
LR = config.getfloat('HYPERPARA','LR')                  # learning rate
EPSILON = config.getfloat('HYPERPARA','EPSILON')              # 最优选择动作百分比
GAMMA = config.getfloat('HYPERPARA','GAMMA')                 # 奖励递减参数
TARGET_REPLACE_ITER = config.getfloat('HYPERPARA','TARGET_REPLACE_ITER')  # Q 现实网络的更新频率
MEMORY_CAPACITY = config.getint('HYPERPARA','MEMORY_CAPACITY')     # 记忆库大小


N_ACTIONS = 12  # 能做的动作
N_STATES = 9*3   # 能获取的环境信息数

nRow = 3
nCol = 3
colorSize=3

animationOn = False
animationfps=5

#CPU数
processes = 6

def choose_action_custom(x,limit,eval_):
        x = torch.unsqueeze(torch.FloatTensor(x).to(device), 0)
        # 这里只输入一个 sample
        if np.random.uniform() < EPSILON:   # 选最优动作
            actions_value = eval_.forward(x)[0].cpu().data.numpy()
            if(len(limit)>0):
                for index,__ in enumerate(actions_value):
                    if(not index in limit):
                        actions_value[index] = -1e9
            #选一个最大的动作
            action = actions_value.argmax()
        else:   # 选随机动作
            if(len(limit)>0):
                action = np.random.choice(limit,1,False)[0]
            else:
                action = np.random.randint(0, N_ACTIONS)
        return action



def processEpoch(pipe):
    totalreward=0
    combo = 0
    maxcomboget = 0
    board = Board(rowSize=nRow,colSize=nCol,colorSize=colorSize,limitsteps=10) # 定义版面
    #刷新版面
    board.initBoardnoDup(True)
    pos=np.random.randint(0,[nRow,nCol]).tolist()
    util = Util(nRow,nCol,colorSize)#定义util
    #转珠限制
    limit =util.getLimit(pos)
    #从主进程获取网络
    net = pipe.recv()
    eval_ = net
    #传输数据
    pipedata=[]
    while(True):
        s = board.board
        #平铺
        transS = util.boardTrans(s.reshape(1,-1)[0])
        transS = util.onehot(transS,colorsize=3)
        a = choose_action_custom(transS,limit,eval_)
        # 选动作, 得到环境反馈
        s_, r, done, combo,pos,limit = board.step(pos,a,combo)
        transS_ = util.boardTrans(s_.reshape(1,-1)[0])
        transS_ = util.onehot(transS_,colorsize=3)
        maxcomboget = max(maxcomboget,combo)
        # 传输记忆
        # dqn.store_transition(transS, a, r, transS_)
        totalreward += r

        pipedata.append([transS, a, r, transS_])

        if done:    # 如果回合结束, 进入下回合
            #发送数据
            pipe.send((pipedata,totalreward,maxcomboget))
            pipedata=[]
            #初始化
            totalreward=0
            combo = 0
            maxcomboget = 0
            board.initBoardnoDup(True)
            pos=np.random.randint(0,[nRow,nCol]).tolist()
            #更新网络 顺便同步 因为走到这一步会卡主，等待主进程发送数据
            net = pipe.recv()
            eval_ = net
        s = s_
    print('episode:{},total loss:{},maxcombo:{},totalreward:{},train started:{}'.format(i_episode,totalloss/board.steps,maxcomboget,totalreward,dqn.memory_counter > MEMORY_CAPACITY))

def main():
    print('start training...')
    dqn = DQN() # 定义 DQN 系统
    #启动进程
    pipe_dict = dict((i, (pipe1, pipe2)) for i in range(processes) for pipe1, pipe2 in (ctx.Pipe(),))
    child_process_list = []
    for i in range(processes):
        print('start process:{}'.format(i))
        pro = ctx.Process(target=processEpoch, args=(pipe_dict[i][1],))
        child_process_list.append(pro)
    # 发送第一波数据 启动进程探索
    [pipe_dict[i][0].send(dqn.eval_net) for i in range(processes)]
    [p.start() for p in child_process_list]

    # animation = padEnv() #定义动画
    # util = Util(nRow,nCol,colorSize)#定义util
    savefile = 'pazzuleparams_tiny'
    loadfile = 'pazzuleparams_tiny_last1.ckpt'
    if(os.path.isfile(loadfile)):
        print('loading weights....')
        dqn.target_net.load_state_dict(torch.load(loadfile))
        dqn.eval_net.load_state_dict(torch.load(loadfile))
    #启动动画
    #animation.gameStart(fps=0)
    #开始训练
    nowepoch = 0
    epochloss=0
    calreward = 0
    calloss = 0
    time1=datetime.datetime.now()
    for i_episode in range(1,10000000):
        for i in range(processes):
            receive = pipe_dict[i][0].recv()
            # transS, a, r, transS = receive[0]
            totalreward = receive[1]
            maxcomboget = receive[2]
            for transS, a, r, transS_ in receive[0]:
                dqn.store_transition(transS, a, r, transS_)
                if dqn.memory_counter > MEMORY_CAPACITY:
                    epochloss = dqn.learn() # 记忆库满了就进行学习
            nowepoch+=1
            calreward += totalreward
            calloss += epochloss
            print('process:{},episode:{},loss:{},maxcombo:{},totalreward:{},train started:{}'.format(i,nowepoch,epochloss,maxcomboget,totalreward,dqn.memory_counter > MEMORY_CAPACITY))
            if(nowepoch%1000==0):
                time2 = datetime.datetime.now()
                print('1000epochs执行所花费时间:{},平均得分:{},平均loss:{}'.format((time2-time1),calreward/1000,calloss/1000))
                time1 = time2
                calreward = 0
                calloss = 0
        [pipe_dict[i][0].send(dqn.eval_net) for i in range(processes)]

if __name__ == '__main__':
    main()