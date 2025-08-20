# SAR-RF-Real-Time-RF-Signal-Classification

SAR-RF is an edge-ML stack for real-time RF signal classification designed for Search-and-Rescue (SAR) scenarios and future UAV/drone deployment. It ingests complex IQ samples from an SDR, classifies the modulation in real time, and exposes results to a lightweight UI and mapping layer. The project targets two hardware paths:

* DPU-FPGA (Vitis AI on Zynq/Zedboard-class devices) for low-power, on-board inference (INT8).

* GPU (CUDA-class devices / Jetson / desktop) for rapid prototyping and high-throughput lab runs.
