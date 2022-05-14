#!/bin/bash
read -p "Please Enter You file: " NAME
read -p "Please Enter You processes(1,4,16): " PROCESS
echo "Your fileName Is: $NAME"
ls
yosys -p "read_verilog ${NAME}.v; synth -auto-top; clean; write_verilog ${NAME}_syn.v"
./converter -v ${NAME}_syn.v
./mig ${NAME}_syn_out.v 1 "noop"
cd /home/abcRL2.0-4-24
python testReinforce.py -i "/home/MIG_project-main/${NAME}_syn_out_opt_1.v" -n "${NAME}" -p "${PROCESS}"
