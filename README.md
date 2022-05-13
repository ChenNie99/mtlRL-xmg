The main framework is derived from :https://github.com/krzhu/abc_py

# Python environment

The project is tested in the environment:
```
Python 3.7.5
Pytorch 1.3
```

The project has other dependencies such as `numpy, six, etc.`
Please installing the dependencies correspondingly.


--------

# Usage

The current version can execute on combinational `.v` benchmarks.
To run the REINFORCE algorithm, please first preprocess the target verilog file:
```
yosys -p "read_verilog name.v; synth -auto-top; clean; write_verilog name_syn.v"
./converter -v name_syn.v
./mig name_syn_out.v 1 ""
```
And then execute `python3 testReinforce.py -i <path of file> -n <custom name> -p <num of process>`

Or use the verilog_to_xmg.sh to automately optimize a raw verilog file:

such as:

input file: epfl_max

processes: 4

--------

# mtlRL-xmg
