import torch
import numpy as np
from env import EnvGraph_mtl_xmg as Env
import sys, getopt

def test_action_runtime(inputfile, name_brief):

    end2end_target = 4
    env = Env(inputfile, 0, end2end_target)
    o_filename = "runtime4actions/" + name_brief + "_action_runtime.txt"
    output = sys.stdout
    outputfile = open(o_filename, 'w')
    sys.stdout = outputfile
    repeat = 5
    env.test_action_runtime_2(repeat)

    outputfile.close()
    sys.stdout = output
    

if __name__ == "__main__":
    argv_ = sys.argv[1:]
    inputfile = ''
    name = ''
    
    try:
        opts, args = getopt.getopt(argv_, "hi:n:p:t:", ["ifile=", "name="])
    except getopt.GetoptError:
        print('test_runtime_for_action.py -i <inputfile> -n <name>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('test_runtime_for_action.py -i <inputfile> -n <name>')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg
        elif opt in ("-n", "--name"):
            name = arg
        
    
    test_action_runtime(inputfile, name)
    print("done! test_action_runtime")