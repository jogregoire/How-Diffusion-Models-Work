import torch
import logging as log

class GPUPerf:
    def __init__(self, gpu_enabled, device):
        self.gpu_enabled = gpu_enabled
        self.device = device
        device_name = torch.cuda.get_device_name(0)
        device_count = torch.cuda.device_count()
        self.mem_total = torch.cuda.get_device_properties(0).total_memory
        log.info(f"Device: {device_name}, Count: {device_count}, Memory: {self.__bytefmt(self.mem_total)} GB")
    
    def snapshot(self, event="None"):
        mem = torch.cuda.memory_stats(device=self.device)
        allocated = mem['active_bytes.all.allocated']
        freed = mem['active_bytes.all.freed']
        current = mem['active_bytes.all.current'] # = allocated - freed
        peak = mem['active_bytes.all.peak']
        available_at_peak = self.mem_total - peak

        # Now, use this function in your log message
        log.info(f"Event: {event} Current: {self.__bytefmt(current)}, Peak: {self.__bytefmt(peak)}, Available at Peak: {self.__bytefmt(available_at_peak)}, Allocated total: {self.__bytefmt(allocated)}, Freed: {self.__bytefmt(freed)}")

    def record_memory_history(self):
        # record memory usage: https://pytorch.org/docs/stable/torch_cuda_memory.html
        # no support on non-linux non-x86_64 platforms
        torch.cuda.memory._record_memory_history(True, device=self.device)
        
    def dump_memory_history(self):
        torch.cuda.memory._dump_snapshot("my_snapshot.pickle")

    def __bytefmt(self, byte_size):
        # Define the suffixes for each size
        suffixes = ['B', 'KB', 'MB', 'GB', 'TB', 'PB']

        # Find the index of the suffix we'll use
        i = 0
        while byte_size >= 1024 and i < len(suffixes)-1:
            byte_size /= 1024.
            i += 1
        f = ('%.2f' % byte_size).rstrip('0').rstrip('.')
        return f'{f} {suffixes[i]}'