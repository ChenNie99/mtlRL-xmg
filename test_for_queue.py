import multiprocessing
import random
import time
import multiprocessing
def worker(name, q):
    t = 0
    for i in range(10):
        print(name + " " +str(i))
        x = 1
        t += x
        time.sleep(x * 0.1)

    q.put([str(t), t])
q = multiprocessing.Queue()
jobs = []
for i in range(10):
    p = multiprocessing.Process(target=worker, args=(str(i), q))
    jobs.append(p)
    p.start()

for p in jobs:
    p.join()

results = [q.get() for j in jobs]
print(results)