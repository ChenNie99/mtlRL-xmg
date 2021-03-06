import multiprocessing as mp
import time

def job(v, num):

    for _ in range(10):
        time.sleep(0.1)
        v.value += num
        print(v.value)
    return 66
def multicore():

    v = mp.Value('i', 0)
    p1 = mp.Process(target=job, args=(v, 1))
    p2 = mp.Process(target=job, args=(v, 3))
    p1.start()
    p2.start()
    p1.join()
    p2.join()
    print(mp.Pool().apply_async(p1).get())
if __name__ == '__main__':
    multicore()

