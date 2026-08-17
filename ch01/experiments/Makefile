# ch01 實測程式
#
#   make            編譯全部
#   make run        編譯並依序執行全部
#   make clean      清掉執行檔
#
# ARCH 預設抓本機 GPU 的 compute capability，也可以自己指定：
#   make ARCH=sm_86

NVCC ?= nvcc

# sm_89 = Ada Lovelace，對應本機 RTX 4060 Laptop (compute capability 8.9)。
# 換機器時跑 ./01_device_query 看 compute capability，再改這裡或用 make ARCH=sm_xx
ARCH ?= sm_89

NVCCFLAGS := -arch=$(ARCH) -O2

BINS    := 01_device_query 02_warp_blocksize 03_context_compare 05_gpu_load
SCRIPTS := 04_nvcc_pipeline.sh 06_pstate_monitor.sh 07_nvidia_smi_query.sh

all: $(BINS)
	@chmod +x $(SCRIPTS)

01_device_query: 01_device_query.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

02_warp_blocksize: 02_warp_blocksize.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

# Driver API 在 libcuda（由顯示卡驅動安裝），要額外連結 -lcuda
03_context_compare: 03_context_compare.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $< -lcuda

05_gpu_load: 05_gpu_load.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

# 01~03 是問答一二題的實驗，跑得很快
run: all
	@for b in 01_device_query 02_warp_blocksize 03_context_compare; do \
		echo "=================================================="; \
		echo "  ./$$b"; \
		echo "=================================================="; \
		./$$b; \
		echo; \
	done

# 04/06/07 是問答三四五題的實驗。06 會加負載跑約一分鐘，故與 run 分開
run-tools: all
	@for s in $(SCRIPTS); do \
		echo "=================================================="; \
		echo "  ./$$s"; \
		echo "=================================================="; \
		./$$s; \
		echo; \
	done

run-all: run run-tools

clean:
	rm -f $(BINS)

.PHONY: all run run-tools run-all clean
