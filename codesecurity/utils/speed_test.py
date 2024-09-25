import time

def time_cost(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        res = func(*args, **kwargs)
        print("time cost: ", time.time() - start)
        return res
    return wrapper

def compute_time_cost(func, *args, **kwargs):
    start = time.time()
    res = func(*args, **kwargs)
    print("time cost: ", time.time() - start)
    return res