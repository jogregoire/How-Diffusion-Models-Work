import torch
import pynvml
import datetime
import pandas as pd
import logging as log

class GPUPerf:
    def __init__(self, gpu_enabled, device):
        self.gpu_enabled = gpu_enabled
        self.device = device
        device_name = torch.cuda.get_device_name(0)
        device_count = torch.cuda.device_count()
        self.mem_total = torch.cuda.get_device_properties(0).total_memory
        
        pynvml.nvmlInit()
        deviceCount = pynvml.nvmlDeviceGetCount()
        for i in range(deviceCount):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            log.info(f"Device {i}: {pynvml.nvmlDeviceGetName(handle)}")

        # Get handle to the first GPU device
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        max_power = pynvml.nvmlDeviceGetPowerManagementLimit(self.handle)

        log.info(f"Device: {device_name}, Count: {device_count}, Memory: {self.__bytefmt(self.mem_total)} GB, Driver: {pynvml.nvmlSystemGetDriverVersion()}, Max Power: {max_power / 1000.0} W")

        self.df = pd.DataFrame(columns=['Time', 'Event', 'AllocatedMem', 'FreedMem', 'CurrentMem', 'PeakMem', 'Usage', 'Temperature', 'Power'])

    def snapshot(self, event="None"):
        if self.gpu_enabled:
            mem = torch.cuda.memory_stats(device=self.device)
            allocated = mem['active_bytes.all.allocated']
            freed = mem['active_bytes.all.freed']
            current = mem['active_bytes.all.current'] # = allocated - freed
            peak = mem['active_bytes.all.peak']

            # Now, use this function in your log message
            log.debug(f"Event: {event} Memory Current: {self.__bytefmt(current)}, Peak: {self.__bytefmt(peak)}, Allocated total: {self.__bytefmt(allocated)}, Freed: {self.__bytefmt(freed)}")

            usage = torch.cuda.utilization(device=self.device)
            power = pynvml.nvmlDeviceGetPowerUsage(self.handle)
            temperature = pynvml.nvmlDeviceGetTemperature(self.handle, pynvml.NVML_TEMPERATURE_GPU)

            log.debug(f"Event: {event} Utilization: {usage}, Temp: {temperature} C, Power {power / 1000.0} W")

            # Append a new row to the DataFrame
            new_row = {
                'Time': datetime.datetime.now(),
                'Event': event,
                'AllocatedMem': allocated,
                'FreedMem': freed,
                'CurrentMem': current,
                'PeakMem': peak,
                'Usage': usage,
                'Temperature': temperature,
                'Power': power / 1000.0
            }

            self.df.loc[len(self.df)] = new_row

    def save_snapshots(self, filename):
        self.df.to_csv(filename)

    def record_memory_history(self):
        # record memory usage: https://pytorch.org/docs/stable/torch_cuda_memory.html
        # no support on non-linux non-x86_64 platforms
        if self.gpu_enabled:
            torch.cuda.memory._record_memory_history(True, device=self.device)
        
    def dump_memory_history(self):
        if self.gpu_enabled:
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