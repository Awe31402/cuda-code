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

BINS := 01_device_query 02_warp_blocksize 03_context_compare

all: $(BINS)

01_device_query: 01_device_query.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

02_warp_blocksize: 02_warp_blocksize.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

# Driver API 在 libcuda（由顯示卡驅動安裝），要額外連結 -lcuda
03_context_compare: 03_context_compare.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $< -lcuda

run: all
	@for b in $(BINS); do \
		echo "=================================================="; \
		echo "  ./$$b"; \
		echo "=================================================="; \
		./$$b; \
		echo; \
	done

clean:
	rm -f $(BINS)

.PHONY: all run clean
