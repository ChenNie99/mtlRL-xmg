#!/bin/bash
read -p "Please Enter You file:: " NAME
echo "Your fileName Is: $NAME"
ls
yosys -p "read_verilog ${NAME}.v; synth -auto-top; clean; write_verilog ${NAME}_syn.v"
./converter -v ${NAME}_syn.v
./mig ${NAME}_syn_out.v 1 ""
