# FPGA-Friendly SpinQuant (U280 / V80)

End-to-end FPGA implementation of an LLM inference pipeline optimized with **SpinQuant**.  
Includes a reusable **HLS kernel library (TAPA)** and a host to measure **prefilling**, **decoding**, and **end-to-end** latency on AMD/Xilinx FPGAs.

<p align="center">
  <img src="figs/SpinQuant_Ours.png" width="70%" alt="SpinQuant FPGA-friendly overview">
</p>

---

## Repository Layout

```
FPGA_Friendly_SpinQuant/
├─ src/                       # HLS kernel library + host sources and build files
│  └─ SpinQuant_Prefilling_Decoding_mem_opt_tb.cpp   # main host (edit seq lengths here)
├─ run/                       # Bitstreams and compiled host for quick tests
│  ├─ bitstreams/
│  │  ├─ SpinQuant_Prefilling_mem_opt_xilinx_u280_gen3x16_xdma_1_202211_1.xclbin
│  │  └─ SpinQuant_Decoding_mem_opt_xilinx_u280_gen3x16_xdma_1_202211_1.xclbin
│  └─ (copy the compiled host binary here)
├─ figs/                      # figures (optional)
└─ README.md
```

---
## Attention

There are two missing model parameter files due to super large size
```
FPGA_Friendly_SpinQuant/src/parameters/lm_head.bin
FPGA_Friendly_SpinQuant/src/parameters/model_embed_tokens_fp32.bin
```
Please download from 


## Requirements

- **OS:** Ubuntu 20.04 / 22.04 (or similar)
- **XRT** installed and sourced  
  ```bash
  source /opt/xilinx/xrt/setup.sh
  ```
- **Vitis 2022.2** if you plan to rebuild bitstreams
- **TAPA CLI** available on `PATH` (for `tapa g++`)
- A board/driver that matches your `.xclbin` platform (e.g., `xilinx_u280_gen3x16_xdma_1_202211_1`)

**Check device visibility**
```bash
xbutil examine
```

---

## Configure Prompt / Decode Sizes

Edit the two constants in the host's kernel invocations:
```
FPGA_Friendly_SpinQuant/src/SpinQuant_Prefilling_Decoding_mem_opt_tb.cpp
```
- `MAX_PRE_SEQ_LEN`  — prompt (prefill) size  
- `MAX_DEC_SEQ_LEN`  — decode size

Rebuild after any change.

---

## Build (Host Only)

From `FPGA_Friendly_SpinQuant/src`:
```bash
tapa g++ -- SpinQuant_Prefilling_Decoding_mem_opt_tb.cpp -o SpinQuant_Prefilling_Decoding_mem_opt
```
Copy the binary to `run/`:
```bash
cp SpinQuant_Prefilling_Decoding_mem_opt ../run/
```

---

## Run on U280

From `FPGA_Friendly_SpinQuant/run`:
```bash
./SpinQuant_Prefilling_Decoding_mem_opt \
  --bitstream_pref ./bitstreams/SpinQuant_Prefilling_mem_opt_xilinx_u280_gen3x16_xdma_1_202211_1.xclbin \
  --bitstream_dec  ./bitstreams/SpinQuant_Decoding_mem_opt_xilinx_u280_gen3x16_xdma_1_202211_1.xclbin
```

### Example Output (prompt=1024, decode=1024)

```
Prefilling parameters:
  Token Parallel: 8
  ...
Run 0 — kernel time: 1.66154 s
...
Run 0 — kernel time: 9.85676 s
...
Average prefilling kernel time over 3 runs: 1.66151 s
Average decoding kernel time over 3 runs:    9.85683 s
```

(Your log will also include buffer sizes and weight loading messages.)

---

## Notes for V80

V80 numbers in the paper are **estimated** from the U280 design (scaled to 300 MHz) and quick Vivado builds. On-board V80 support and bitstreams will be added when stable.

---

## Acknowledgments

We gratefully acknowledge the AMD team — **Fraser Nicholas** and **Blott Michaela** — for their guidance and support on model design and training.
