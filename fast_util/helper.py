import torch
import resource
import time

class nvtxTimeTable(object):
    def __init__(self, allTimeStr=None, use_nvtx=False):
        '''
        During initialization, provide a string representing the total time, occupying the first position; by default, disable use_nvtx to avoid affecting performance
        '''
        self.timeDict = {}
        if allTimeStr is not None:
            self.timeDict[allTimeStr] = 0
        self.nvtxFlag = use_nvtx
        self.startTimeStack = []
        self.strStack = []
        self.strID = 0
        self.MultiEpochTime = {}

    def start(self):
        return time.perf_counter()

    def elapsed(self, start):
        return time.perf_counter() - start
        
    def nvtx_push(self,str="nvtxTag"):
        if self.nvtxFlag:
            torch.cuda.nvtx.range_push(str)

    def nvtx_pop(self):
        if self.nvtxFlag:
            torch.cuda.nvtx.range_pop()

    def nvtx_start(self):
        if self.nvtxFlag == 1: 
            torch.cuda.cudart().cudaProfilerStart()

    def nvtx_stop(self):
        self.strID = 0
        if self.nvtxFlag == 1: 
            torch.cuda.cudart().cudaProfilerStop()

    def time_push(self, keyStr=None):
        '''
        push the start time to the stack, work with nvtx_push
        '''
        # config str
        if keyStr == None:
            self.strID += 1
            keyStr = "t_" + str(self.strID)
        # push
        self.strStack.append(keyStr)
        self.startTimeStack.append(self.start())
        if self.nvtxFlag:
            self.nvtx_push(keyStr)
        if keyStr not in self.timeDict.keys():
            self.timeDict[keyStr] = 0
        return self.start()
    
    def time_pop(self, NoUseStr=None):
        '''
        pop the start time from the stack, work with nvtx_push
        '''
        if self.strStack:
            keyStr = self.strStack.pop()
        else:
            raise IndexError("pop from empty strStack")
        if self.startTimeStack:
            start = self.startTimeStack.pop()
        else:
            raise IndexError("pop from empty startTimeStack")
        if self.nvtxFlag:
            self.nvtx_pop()
        # save
        self.timeDict[keyStr] += self.elapsed(start)
        return self.elapsed(start)
    def save_print_time(self, pFlag=False):
        '''
        Called at the end of each round, it prints, saves and refreshes.
        '''
        result = "\t"
        result += ", ".join(self.timeDict.keys()) + ": "
        result += ", ".join(f"{v:.2f}" for v in self.timeDict.values())
        if pFlag: print(result)
        if not self.MultiEpochTime: # When saving for the first time, initialization occurs
            for k in self.timeDict.keys():
                self.MultiEpochTime[k] = []
        
        for k in self.timeDict.keys():
            self.MultiEpochTime[k].append(self.timeDict[k])
            self.timeDict[k] = 0
        
        return result

    def print_avg_stats(self, dropWarmNum=0):
        # calculate averages, skip N points because of warmup
        avg_stats = {}
        for k, v in self.MultiEpochTime.items():
            if len(v) > dropWarmNum:
                valid_data = v[dropWarmNum:]
                avg_stats[k] = sum(valid_data) / len(valid_data) if valid_data else 0
            else:
                avg_stats[k] = sum(v) / len(v) if v else 0
        
        # print(f'=== Epoch Num: {len(self.MultiEpochTime[k])}, Drop Warmup Epoch Num: {dropWarmNum}')
        print("=== AVG : ",end="")
        print(", ".join(avg_stats.keys()),end=": ")
        print(", ".join(f"{v:.2f}" for v in avg_stats.values()))

nsysForward = nvtxTimeTable("t_forward")
timeStack = nvtxTimeTable()

# initialize time stats dictionary
time_stats = {
    'allTime': [],
    'sample': [],
    'mfgs': [],
    'forward': [],
    'backward': [],
    'overHead': []
}

def append_time_stats(time_tot, time_sample, time_mfgs, 
                     time_forward, time_backward, overhead):
    # Append the provided time statistics to the respective lists
    time_stats['allTime'].append(time_tot)
    time_stats['sample'].append(time_sample)
    time_stats['mfgs'].append(time_mfgs)
    time_stats['forward'].append(time_forward)
    time_stats['backward'].append(time_backward)
    time_stats['overHead'].append(overhead)

def print_avg_stats(dropWarmNum=0):
    # calculate averages, skip N points because of warmup
    avg_stats = {}
    for k, v in time_stats.items():
        if len(v) > dropWarmNum:
            valid_data = v[dropWarmNum:]
            avg_stats[k] = sum(valid_data) / len(valid_data) if valid_data else 0
        else:
            avg_stats[k] = sum(v) / len(v) if v else 0
    
    print(f'=== Epoch Num: {len(time_stats["allTime"])}, Drop Warmup Epoch Num: {dropWarmNum}')
    print('=== allTime, sample, mfgs, forward, backward, overHead: '
          '{:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}'.format(
              avg_stats['allTime'],
              avg_stats['sample'],
              avg_stats['mfgs'],
              avg_stats['forward'],
              avg_stats['backward'],
              avg_stats['overHead']
          ))

def PRINT_PEAK_MEM(DEV_MEM=0):
    # Print peak memory usage for both host and device
    hostPeakMemMB = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024
    if DEV_MEM == 0:
        DEV_MEM = (torch.cuda.max_memory_allocated() / 1024**3) if torch.cuda.is_available() else 0

    print(f"=== Peak Host Memory, Peak Device Memory:(GB) { hostPeakMemMB :.2f}, {DEV_MEM:.2f}")
    return hostPeakMemMB, DEV_MEM
