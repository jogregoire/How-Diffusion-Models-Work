import torch
import logging as log

class GPUPerf:
    def __init__(self, gpu_enabled, device):
        self.gpu_enabled = gpu_enabled
        self.device = device
        device_name = torch.cuda.get_device_name(0)
        device_count = torch.cuda.device_count()
        self.mem_total = torch.cuda.get_device_properties(0).total_memory/(1024**2)
        log.info(f"Device: {device_name}, Count: {device_count}, Memory: {self.mem_total/1024:.2f} GB")
    
    def snapshot(self, event="None"):
        mem = torch.cuda.memory_stats(device=self.device)
        allocated = mem['active_bytes.all.allocated']/(1024**2)
        freed = mem['active_bytes.all.freed']/(1024**2)
        current = mem['active_bytes.all.current']/(1024**2) # = allocated - freed
        peak = mem['active_bytes.all.peak']/(1024**2)
        available_at_peak = self.mem_total - peak
        log.info(f"Event: {event} Allocated: {allocated:.2f} MB, Freed: {freed:.2f} MB, Current: {current:.2f} MB, Available at Peak: {available_at_peak:.2f} MB")

    def record_memory_history(self):
        # record memory usage: https://pytorch.org/docs/stable/torch_cuda_memory.html
        # no support on non-linux non-x86_64 platforms
        torch.cuda.memory._record_memory_history(True, device=self.device)
        
    def dump_memory_history(self):
        torch.cuda.memory._dump_snapshot("my_snapshot.pickle")
