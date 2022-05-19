
import math




def xmg_evaluation(file_name):
    
    # hardware
    array_size = 256
    mat_size = 4
    bank_size = 4 
    rank_size = 4 
    
    # clock
    clock = 1 # nS

    # cycle-wise operation energy cost
    cycle_energy_not_overwrite = 46.93 # fJ
    cycle_energy_overwrite = 54.61 # fJ
    cycle_energy_average = (cycle_energy_not_overwrite+cycle_energy_overwrite)/2

    # intra-mat (cross-array) cost 
    intra_mat_delay = 318.725 * 2 / 1000 # nS
    intra_mat_energy = (11.366+11.059) * 1000 / 256 # fJ/bit

    # intra-bank (cross-mat) cost 
    intra_bank_delay = 336.660 * 2 / 1000 # nS
    intra_bank_energy = (20.291+19.984) * 1000 / 256 # fJ/bit

    # intra-rank (cross-bank) cost 
    intra_rank_delay = 377.793 * 2 / 1000 # nS
    intra_rank_energy = (56.342+56.035) * 1000 / 256 # fJ/bit
    
    
    
    # get xmg 
    xmg = []
    inv_count = 0
    with open(file_name, 'r') as f:
        content = f.readlines()
    for line in content[1:-5]:
        inv_count += line.count("~")
        items = line[:-1].replace(" ","").replace("("," ").replace(")->"," ").replace(","," ").replace("~","").split(" ")
        if(items[0] == "AND"):
            xmg.append(["MAJ", int(items[1]), int(items[2]), -1, int(items[3])])
        elif(items[0] == "OR"):
            xmg.append(["MAJ", int(items[1]), int(items[2]), -1, int(items[3])])  
        elif(items[0] == "XOR"):
            xmg.append(["XOR3", int(items[1]), int(items[2]), 0, int(items[3])])  
        elif(items[0] in ["MAJ","XOR3"]):
            xmg.append([items[0], int(items[1]), int(items[2]), int(items[3]), int(items[4])])
        else:
            print("Not defined operation:", items)
            exit(0)
    
    gate_num = len(xmg)

            
            
    # evaluation
    row_used = 0
    
    cycle = 0
    latency = 0
    energy = 0 
    
    for gate in xmg:
        # basic cost
        cycle += 1
        latency += clock
        energy += cycle_energy_average
        # communication cost
        dst_row = gate[-1] + 1
        dst_array_id = math.ceil(dst_row/array_size)
        dst_mat_id = math.ceil(dst_row/array_size/mat_size)
        dst_bank_id = math.ceil(dst_row/array_size/mat_size/bank_size)
        dst_rank_id = math.ceil(dst_row/array_size/mat_size/bank_size/rank_size)
        row_used = max(row_used, dst_row)
        for src in gate[1:-1]:
            if src >= 0:
                src_row = src + 1
                src_array_id = math.ceil(src_row/array_size)
                row_used = max(row_used, src_row)
                if src_array_id != dst_array_id:
                    src_mat_id = math.ceil(src_row/array_size/mat_size)
                    
                    if src_mat_id != dst_mat_id:
                        src_bank_id = math.ceil(src_row/array_size/mat_size/bank_size)
                        
                        if src_bank_id != dst_bank_id:
                            src_rank_id = math.ceil(src_row/array_size/mat_size/bank_size/rank_size)
                            
                            if src_bank_id != dst_bank_id:
                                print("Too large memory occupation")
                                exit(1)
                            else:
                                latency += intra_rank_delay
                                energy += intra_rank_energy
                        else:
                            latency += intra_bank_delay
                            energy += intra_bank_energy
                    else:
                        latency += intra_mat_delay
                        energy += intra_mat_energy
                        
                        
    print("gate_num:", gate_num, "latency(uS):",latency/1e3, "energy(nJ):", energy/1e6, "inv_ratio", inv_count/gate_num, "row_usage", row_used)  
    
    return(gate_num, latency/1e3, energy/1e6, row_used, (latency/1e3)*energy/1e6)            
                    
                        
                
         


# gate_num, latency, energy, row_usage = xmg_evaluation('epfl_priority_opt_1_map.txt')
# print(gate_num, latency, energy, row_usage)
