import nvidia_smi
import psutil
import sched, time
nvidia_smi.nvmlInit()
handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)

s = sched.scheduler(time.time, time.sleep)
max_gpu = 0
max_gpu_mem = 0
curr_gpu = 0
curr_gpu_mem = 0
max_cpu = 0
max_cpu_mem = 0
curr_cpu = 0
curr_cpu_mem = 0

# res = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
# print(f'gpu: {res.gpu}%, gpu-mem: {res.memory}%')
def keep(sc):
	res = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
	info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
	# print("Device {}: {}, Memory : ({:.2f}% free): {}(total), {} (free), {} (used)".format(0, nvidia_smi.nvmlDeviceGetName(handle), 100*info.free/info.total, info.total, info.free, info.used))
	global curr_gpu
	global curr_gpu_mem
	global max_gpu
	global max_gpu_mem
	global curr_cpu
	global curr_cpu_mem
	global max_cpu
	global max_cpu_mem
	curr_gpu = res.gpu
	curr_gpu_mem = 100 - 100*info.free/info.total
	curr_cpu = psutil.cpu_percent()
	curr_cpu_mem = psutil.virtual_memory().percent
	if curr_gpu > max_gpu:
		max_gpu = curr_gpu
	if curr_gpu_mem > max_gpu_mem:
		max_gpu_mem = curr_gpu_mem
	if curr_cpu > max_cpu:
		max_cpu = curr_cpu
	if curr_cpu_mem > max_cpu_mem:
		max_cpu_mem = curr_cpu_mem
	s.enter(1, 1, keep, (sc,))

def log(sc):
	print("-*"*30)
	print(f'gpu: {curr_gpu}%, gpu-mem: {curr_gpu_mem}%')
	print(f'max gpu: {max_gpu}%, max gpu-mem: {max_gpu_mem}%')
	print(f'cpu: {curr_cpu}%, cpu-mem: {curr_cpu_mem}%')
	print(f'max cpu: {max_cpu}%, max cpu-mem: {max_cpu_mem}%')
	s.enter(30, 1, log, (sc,))

s.enter(1, 1, keep, (s,))
s.enter(5, 2, log, (s,))
s.run()
