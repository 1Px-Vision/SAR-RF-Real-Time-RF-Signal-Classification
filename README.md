# SAR-RF-Real-Time-RF-Signal-Classification

![](https://github.com/1Px-Vision/SAR-RF-Real-Time-RF-Signal-Classification/blob/main/SAR_RF.jpg)

SAR-RF is an edge-ML stack for real-time RF signal classification designed for Search-and-Rescue (SAR) scenarios and future UAV/drone deployment. It ingests complex IQ samples from an SDR, classifies the modulation in real time, and exposes results to a lightweight UI and mapping layer. The project targets two hardware paths:

* DPU-FPGA (Vitis AI on Zynq/Zedboard-class devices) for low-power, on-board inference (INT8).

* GPU (CUDA-class devices / Jetson / desktop) for rapid prototyping and high-throughput lab runs.

In SAR operations, spectrum awareness helps detect, localize, and prioritize signals of interest (e.g., distress beacons, VHF/UHF comms) under tight size-weight-and-power (SWaP) constraints. SAR-RF brings robust RF classifiers to the edge so drones and field kits can act faster with limited bandwidth.

* Stream & preprocess IQ from RTL-SDR (or compatible front-ends): resampling, normalization, windowing.

* Classify common modulations (e.g., BPSK, QPSK, GMSK, FM, OOK, OQPSK, 8PSK, 16QAM, AM-DSB-WC/SC).

* Run anywhere: INT8 on DPU (Vitis AI) or FP32/FP16 on GPU with a shared model interface.

* Serve over Ethernet: FastAPI inference server on the device; Dash UI client on ground station.

* Visualize: live PSD/Waterfall, per-class confidence bars, and optional map overlays/GNSS.
