import pynvml 
import numpy as np

def GPU():
    pynvml.nvmlInit()
    free_list = []
    for iter in range(pynvml.nvmlDeviceGetCount()):
        handle = pynvml.nvmlDeviceGetHandleByIndex(iter)
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        free_list.append(meminfo.free)
    return np.argmax(free_list)