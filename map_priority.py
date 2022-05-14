# PROFILE
from datetime import datetime
prof_start = datetime.now()

import os
import sys
import xmgmap

if __name__ != "__main__":
    raise Exception("Not executed directly")

if len(sys.argv) < 2 or len(sys.argv) > 3:
    raise Exception("Incorrect number of inputs")

# name of current script, without extension
script_name = os.path.splitext(__file__)[0]

print("Inverted gate inputs:", "ON" if xmgmap.INVERT_EDGES else "OFF", flush=True)
MAP = xmgmap.Mapping()
filename, x_values = sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else ""
MAP.read(filename, x_values)
MAP.build_level()

try:
    MAP.priority_map()
    if not xmgmap.INVERT_EDGES:
        MAP.invert_result()
    MAP.print_result()
except xmgmap.MemoryOverflow as e:
    print(type(e).__name__ + ":", e, file=sys.stderr)

# PROFILE
prof_dur = datetime.now() - prof_start
with open(script_name + "_prof.txt", "a") as prof_out:
    prof_out.write("Total runtime = " + str(prof_dur) + "\n")