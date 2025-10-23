# SAR-RF-Real-Time-RF-Signal-Classification

![](https://github.com/1Px-Vision/SAR-RF-Real-Time-RF-Signal-Classification/blob/main/SAR_RF.jpg)

SAR-RF is an edge-ML stack for real-time RF signal classification designed for Search-and-Rescue (SAR) scenarios and future UAV/drone deployment. It ingests complex IQ samples from an SDR, classifies the modulation in real-time, and exposes the results to a lightweight UI and mapping layer. The project targets two hardware paths:

* DPU-FPGA (Vitis AI on Zynq/Zedboard-class devices) for low-power, on-board inference (INT8).

* GPU (CUDA-class devices / FPGA / Jetson / desktop) for rapid prototyping and high-throughput lab runs.

In SAR operations, spectrum awareness helps detect, localize, and prioritize signals of interest (e.g., distress beacons, VHF/UHF communications) under tight size, weight, and power (SWaP) constraints. SAR-RF brings robust RF classifiers to the edge so drones and field kits can act faster with limited bandwidth.

* Stream & preprocess IQ from RTL-SDR (or compatible front-ends): resampling, normalization, windowing.

* Classify common modulations (e.g., BPSK, QPSK, GMSK, FM, OOK, OQPSK, 8PSK, 16QAM, AM-DSB-WC/SC).

* Run anywhere: INT8 on DPU (Vitis AI) or FP32/FP16 on GPU with a shared model interface.

* Serve over Ethernet: FastAPI inference server on the device; Dash UI client on the ground station.

* Visualize: live PSD/Waterfall, per-class confidence bars, and optional map overlays/GNSS.

![](https://github.com/1Px-Vision/SAR-RF-Real-Time-RF-Signal-Classification/blob/main/SAR_RF_lab.jpg)


## Dataset
Over-the-air signals inherently exhibit impairments and distortions—an essential part of any realistic dataset—including multipath fading, carrier-frequency offset, timing/phase errors, and additive white Gaussian noise (AWGN). These effects reduce the signal-to-noise ratio (SNR); lower SNR makes error-free reception increasingly difficult. For a fixed SNR, higher-order constellations (larger symbol alphabets) have denser decision regions, increasing the likelihood of detection errors. We utilize the [DeepSig dataset](https://www.deepsig.ai/datasets/), which encompasses both synthetic channel simulations and over-the-air recordings, spanning 27 analog and digital modulation types. 

![SNR](https://github.com/1Px-Vision/SAR-RF-Real-Time-RF-Signal-Classification/blob/main/ACC_SNR_RF.jpg)

[![Google Drive](https://img.shields.io/badge/Google%20Drive-Download-blue?logo=googledrive&logoColor=white)](https://drive.google.com/file/d/1TVBJpDYfoHrtdfIP-LWgFRK-X75OI91a/view?usp=sharing)


## Feature-Scale SE-Net for I/Q Signal Classification

A lightweight encoder–decoder CNN that fuses multi-scale features with a Feature-Scale (FS) block and channel attention (SE) for real-time classification of I/Q sequences. Default input: (1024, 1, 2).

![dpu_M](https://github.com/1Px-Vision/SAR-RF-Real-Time-RF-Signal-Classification/blob/main/Model_DPU.jpg)

### Encoder (5 stages)
Repeated Conv2D → BatchNorm → ReLU blocks (ConvBlock) with stride-2 downsamples. Feature maps:

* F1: (B, 16, H/2, W/2)

* F2: (B, 32, H/4, W/4)

* F3: (B, 64, H/8, W/8)

* F4: (B,128, H/16, W/16)

* F5: (B,256, H/32, W/32)

#### Squeeze-and-Excitation (SE). 
Adaptive 2D pooling → small MLP (or 1×1 convs) → sigmoid scale to recalibrate channels of F5, producing F5_out.

#### Multi-level fusion. 
Concatenate [F1, F2, F3, F4, F5_out] to expose both shallow and deep context to the decoder.

#### Feature-Scale (FS) block. Parallel convolutions at different receptive fields, then fuse:

* 1×1 (dilation = 1)

* 3×3 (dilation = 2)

* 3×3 (dilation = 3)

* Concatenate → Conv2D to mix scales

### Decoder

Concatenate the encoder features [F1, F2, F3, F4, F5_out] after spatially aligning them (to the smallest stride). This stacked tensor is the decoder’s input.

#### Feature-Scale (FS) block (context mixer).
Apply three parallel conv paths to the input, then fuse:

* 1×1, dilation=1

* 3×3, dilation=2

* 3×3, dilation=3
* Concatenate the three outputs channel-wise → Conv2D (1×1) to mix and reduce.

#### Upsample Stage 1.

* Upsample ×2 (nearest or bilinear), ConvBlock(192→128) → ConvBlock(128→64) (ConvBlock = Conv2D → BatchNorm → ReLU.)

#### Upsample Stage 2.

* Upsample ×2, ConvBlock(64→32) → ConvBlock(32→16)

#### Classification head.

AdaptivePool2D to (B, 16, 1, 1) (global pooling; input-size agnostic), Flatten → Linear(16 → Num_Classes) → logits (apply softmax only at evaluation time)

#### HIT: Vitis-AI container
```
docker pull xilinx/vitis-ai:2.5.0

sudo docker ps -a
sudo docker start -ai vitis-tf2

# Common places for PYNQ-DPU / TRD overlays
find / -name "arch.json" 2>/dev/null | grep -Ei "DPU|CZDX8G|KV260|KR260"
```

### 🧪 Training

**Task:** RF modulation classification (10 classes)  
**Input shape:** `(1024, 1, 2)` → (time × height × I/Q channels)  
**Framework:** TensorFlow 2.12 (Vitis-AI container)  
**Params:** **1,858,074** (trainable: 1,855,226)

**Model highlights**
- Stacked `Conv2D + BN + ReLU` downsampling path (16 → 256 channels).
- Lightweight channel attention (SE/ECA-style 1×1 gating).
- Inception-like triple branch at 64-channel scale, then concat → `192` channels.
- Two upsampling stages with `Conv2D` refinement.
- Global Average Pooling → `1×1 Conv` → `Flatten + Softmax` (10 classes).

<details>
<summary><b>Layer summary (click to expand)</b></summary>

<!-- Keep a blank line after <summary> for GitHub to render the table -->

| Layer (type)                     | Output Shape        | Param # | Connected to                         |
|----------------------------------|---------------------|--------:|--------------------------------------|
| rf_input (InputLayer)            | (None, 1024, 1, 2)  |       0 | —                                    |
| conv2d (Conv2D)                  | (None, 512, 1, 16)  |     288 | rf_input[0][0]                       |
| batch_normalization (BatchNorm)  | (None, 512, 1, 16)  |      64 | conv2d[0][0]                         |
| re_lu (ReLU)                     | (None, 512, 1, 16)  |       0 | batch_normalization[0][0]            |
| conv2d_1 (Conv2D)                | (None, 512, 1, 16)  |   2,304 | re_lu[0][0]                          |
| batch_normalization_1 (BatchNorm)| (None, 512, 1, 16)  |      64 | conv2d_1[0][0]                       |
| re_lu_1 (ReLU)                   | (None, 512, 1, 16)  |       0 | batch_normalization_1[0][0]          |
| conv2d_2 (Conv2D)                | (None, 256, 1, 32)  |   4,608 | re_lu_1[0][0]                        |
| batch_normalization_2 (BatchNorm)| (None, 256, 1, 32)  |     128 | conv2d_2[0][0]                       |
| re_lu_2 (ReLU)                   | (None, 256, 1, 32)  |       0 | batch_normalization_2[0][0]          |
| conv2d_3 (Conv2D)                | (None, 256, 1, 32)  |   9,216 | re_lu_2[0][0]                        |
| batch_normalization_3 (BatchNorm)| (None, 256, 1, 32)  |     128 | conv2d_3[0][0]                       |
| re_lu_3 (ReLU)                   | (None, 256, 1, 32)  |       0 | batch_normalization_3[0][0]          |
| conv2d_4 (Conv2D)                | (None, 128, 1, 64)  |  18,432 | re_lu_3[0][0]                        |
| batch_normalization_4 (BatchNorm)| (None, 128, 1, 64)  |     256 | conv2d_4[0][0]                       |
| re_lu_4 (ReLU)                   | (None, 128, 1, 64)  |       0 | batch_normalization_4[0][0]          |
| conv2d_5 (Conv2D)                | (None, 128, 1, 64)  |  36,864 | re_lu_4[0][0]                        |
| batch_normalization_5 (BatchNorm)| (None, 128, 1, 64)  |     256 | conv2d_5[0][0]                       |
| re_lu_5 (ReLU)                   | (None, 128, 1, 64)  |       0 | batch_normalization_5[0][0]          |
| conv2d_6 (Conv2D)                | (None, 64, 1, 128)  |  73,728 | re_lu_5[0][0]                        |
| batch_normalization_6 (BatchNorm)| (None, 64, 1, 128)  |     512 | conv2d_6[0][0]                       |
| re_lu_6 (ReLU)                   | (None, 64, 1, 128)  |       0 | batch_normalization_6[0][0]          |
| conv2d_7 (Conv2D)                | (None, 64, 1, 128)  | 147,456 | re_lu_6[0][0]                        |
| batch_normalization_7 (BatchNorm)| (None, 64, 1, 128)  |     512 | conv2d_7[0][0]                       |
| re_lu_7 (ReLU)                   | (None, 64, 1, 128)  |       0 | batch_normalization_7[0][0]          |
| conv2d_8 (Conv2D)                | (None, 32, 1, 256)  | 294,912 | re_lu_7[0][0]                        |
| batch_normalization_8 (BatchNorm)| (None, 32, 1, 256)  |   1,024 | conv2d_8[0][0]                       |
| re_lu_8 (ReLU)                   | (None, 32, 1, 256)  |       0 | batch_normalization_8[0][0]          |
| conv2d_9 (Conv2D)                | (None, 32, 1, 256)  | 589,824 | re_lu_8[0][0]                        |
| batch_normalization_9 (BatchNorm)| (None, 32, 1, 256)  |   1,024 | conv2d_9[0][0]                       |
| re_lu_9 (ReLU)                   | (None, 32, 1, 256)  |       0 | batch_normalization_9[0][0]          |
| global_average_pooling2d (GAP2D) | (None, 1, 1, 256)   |       0 | re_lu_9[0][0]                        |
| conv2d_10 (Conv2D)               | (None, 1, 1, 16)    |   4,112 | global_average_pooling2d[0][0]       |
| conv2d_11 (Conv2D)               | (None, 1, 1, 256)   |   4,352 | conv2d_10[0][0]                      |
| multiply (Multiply)              | (None, 32, 1, 256)  |       0 | re_lu_9[0][0]; conv2d_11[0][0]       |
| conv2d_12 (Conv2D)               | (None, 32, 1, 64)   |  16,384 | multiply[0][0]                       |
| conv2d_13 (Conv2D)               | (None, 32, 1, 64)   | 147,456 | multiply[0][0]                       |
| conv2d_14 (Conv2D)               | (None, 32, 1, 64)   | 147,456 | multiply[0][0]                       |
| batch_normalization_10 (BN)      | (None, 32, 1, 64)   |     256 | conv2d_12[0][0]                      |
| batch_normalization_11 (BN)      | (None, 32, 1, 64)   |     256 | conv2d_13[0][0]                      |
| batch_normalization_12 (BN)      | (None, 32, 1, 64)   |     256 | conv2d_14[0][0]                      |
| re_lu_10 (ReLU)                  | (None, 32, 1, 64)   |       0 | batch_normalization_10[0][0]         |
| re_lu_11 (ReLU)                  | (None, 32, 1, 64)   |       0 | batch_normalization_11[0][0]         |
| re_lu_12 (ReLU)                  | (None, 32, 1, 64)   |       0 | batch_normalization_12[0][0]         |
| concatenate (Concatenate)        | (None, 32, 1, 192)  |       0 | re_lu_10/11/12                       |
| conv2d_15 (Conv2D)               | (None, 32, 1, 192)  |  36,864 | concatenate[0][0]                    |
| up_sampling2d (UpSampling2D)     | (None, 64, 2, 192)  |       0 | conv2d_15[0][0]                      |
| conv2d_16 (Conv2D)               | (None, 64, 2, 128)  | 221,184 | up_sampling2d[0][0]                  |
| batch_normalization_13 (BN)      | (None, 64, 2, 128)  |     512 | conv2d_16[0][0]                      |
| re_lu_13 (ReLU)                  | (None, 64, 2, 128)  |       0 | batch_normalization_13[0][0]         |
| conv2d_17 (Conv2D)               | (None, 64, 2, 64)   |  73,728 | re_lu_13[0][0]                       |
| batch_normalization_14 (BN)      | (None, 64, 2, 64)   |     256 | conv2d_17[0][0]                      |
| re_lu_14 (ReLU)                  | (None, 64, 2, 64)   |       0 | batch_normalization_14[0][0]         |
| up_sampling2d_1 (UpSampling2D)   | (None, 128, 4, 64)  |       0 | re_lu_14[0][0]                       |
| conv2d_18 (Conv2D)               | (None, 128, 4, 32)  |  18,432 | up_sampling2d_1[0][0]                |
| batch_normalization_15 (BN)      | (None, 128, 4, 32)  |     128 | conv2d_18[0][0]                      |
| re_lu_15 (ReLU)                  | (None, 128, 4, 32)  |       0 | batch_normalization_15[0][0]         |
| conv2d_19 (Conv2D)               | (None, 128, 4, 16)  |   4,608 | re_lu_15[0][0]                       |
| batch_normalization_16 (BN)      | (None, 128, 4, 16)  |      64 | conv2d_19[0][0]                      |
| re_lu_16 (ReLU)                  | (None, 128, 4, 16)  |       0 | batch_normalization_16[0][0]         |
| average_pooling2d (AvgPool2D)    | (None, 1, 1, 16)    |       0 | re_lu_16[0][0]                       |
| conv2d_20 (Conv2D)               | (None, 1, 1, 10)    |     170 | average_pooling2d[0][0]              |
| flatten (Flatten)                | (None, 10)          |       0 | conv2d_20[0][0]                      |
| activation (Activation)          | (None, 10)          |       0 | flatten[0][0]                         |

**Total params:** 1,858,074  
**Trainable params:** 1,855,226  
**Non-trainable params:** 2,848

</details>


**Evaluation**
- Test **loss:** `0.1244` | Test **accuracy:** `0.9223`
- Samples: **31,949**

| Class        | Precision | Recall | F1-score | Support |
|--------------|-----------:|-------:|---------:|--------:|
| BPSK         | 0.97 | 0.97 | 0.97 | 19,798 |
| QPSK         | 0.91 | 1.00 | 0.95 | 1,345 |
| GMSK         | 0.98 | 0.71 | 0.82 | 1,348 |
| FM           | 1.00 | 0.98 | 0.99 | 1,397 |
| OOK          | 1.00 | 1.00 | 1.00 | 1,323 |
| OQPSK        | 1.00 | 1.00 | 1.00 | 1,449 |
| 8PSK         | 1.00 | 0.77 | 0.87 | 1,342 |
| 16QAM        | 0.66 | 0.96 | 0.79 | 1,332 |
| AM-SSB-WC    | 0.95 | 0.19 | 0.32 | 1,295 |
| AM-DSB-SC    | 0.55 | 0.99 | 0.71 | 1,320 |
| **Overall**  | —    | —    | **Accuracy 0.92** | **31,949** |
| **Macro avg**| 0.90 | 0.86 | 0.84 | — |
| **Weighted** | 0.94 | 0.92 | 0.92 | — |

**Notes & diagnostics**
- **Strong:** OOK, OQPSK, FM, BPSK are near-perfect.
- **Recall dips:** GMSK (0.71), 8PSK (0.77) → likely confusion among phase-modulated classes at lower SNRs.
- **16QAM imbalance:** High recall (0.96) but lower precision (0.66) → too many false positives as 16QAM.
- **AM-SSB-WC:** Very low recall (0.19) → consider more training samples, targeted augmentations, or class-specific loss reweighting.

**Next steps**
- Class-balanced sampling / focal loss for **AM-SSB-WC** and **16QAM**.
- SNR-aware augmentation (AWGN, CFO/SFO, multipath) targeted at **GMSK/8PSK**.
- Add a shallow residual head or cosine classifier to improve inter-class margins.
- (Optional, for DPU): keep kernel sizes {1,3,5}, avoid unsupported ops, and fold BN for quantization-aware training.

### DPU Board Support
Vitis AI DPU on Zynq UltraScale+ (DPUCZDX8G)

## DPU-FPGA Platform Support Matrix

| Platform            | DPU Arch | # Cores | Verified Status | Notes |
|---------------------|:-------:|:------:|:---------------:|-------|
| **KV260 SOM**       | B4096   |   1    | ✅ Official      | In AMD Vitis-AI Quick-Start list |
| **ZCU102**          | B4096   |   2    | ✅ Official      | Officially supported reference board |
| **ZCU104**          | B4096   |   2    | ✅ Official      | Officially supported reference board |
| **KR260 SOM**       | B4096   |   1    | ✅ Community     | Verified by AMD TRD / user builds |
| **Pynq-ZU**         | B4096   |   1    | ✅ Community     | Works via DPU-PYNQ overlay |
| **Ultra96 v2**      | B1600   |   1    | ✅ Community     | Runs B1600 via DPU-PYNQ |
| **ZCU111**          | B4096   |   2    | ⚠️ Community     | RFSoC board – custom platform required |
| **ZCU216**          | B4096   |   2    | ⚠️ Community     | RFSoC board – custom platform required |
| **ZCU208**          | B4096   |   2    | ⚠️ Community     | Reported to work with custom build |
| **RFSoC 2×2**       | B4096   |   2    | ⚠️ Community     | Custom Vitis/PetaLinux build needed |
| **RFSoC 4×2**       | B4096   |   2    | ⚠️ Community     | Custom Vitis/PetaLinux build needed |
| **ZCU106**          | B4096   |   2    | ⚠️ Community     | Not in quick-start; works via custom flow |
| **Genesys ZU-5EV**  | B4096   |   1    | ❓ Unverified     | No public confirmation yet |
| **T1 Telco RFSoC**  | B4096   |   2    | ❓ Unverified     | Likely feasible but not confirmed |
| **T1 Telco MPSoC**  | B4096   |   2    | ❓ Unverified     | Likely feasible but not confirmed |
| **TySOM-3A-ZU19EG** | B4096   |   2    | ❓ Unverified     | No public record of DPU build |
| **TySOM-3-ZU7EV**   | B4096   |   2    | ❓ Unverified     | No public record of DPU build |
| **Ultra96 v1**      | B1600   |   1    | ⚠️ Community     | Older DPU-PYNQ demos exist; not maintained |
| **UltraZed-EG**     | B4096   |   1    | ⚠️ Community     | Possible with custom design |
| **ZCU1285**         | B4096   |   2    | ❓ Unverified     | No confirmation found |
| **ZUBoard-1CG**     |  B800   |   1    | ❓ Unverified     | Very limited resources; no public DPU port |

> ✅ Official – in AMD Vitis-AI Quick-Start reference platforms  
> ✅ Community – successfully demonstrated by community / DPU-PYNQ / TRD  
> ⚠️ Community – reported to work but needs custom Vitis/PetaLinux build  
> ❓ Unverified – no confirmed deployment found yet


## DPU Testing: Kria Kr-260 

![kria_1](https://github.com/1Px-Vision/SAR-RF-Real-Time-RF-Signal-Classification/blob/main/DPU_Application.jpg)

### File extension

* .bit.bin: Binary file for bitstream. This is the .bin file that can be generated from Vivado/Vitis instead of .bit file.
* .bsp: Board support package
* .dtb: Device Tree Blob. A binary file containing binary data that describes hardware, compiled from .dtb and .dtbi files.
* .dtbo: Device Tree Blob Overlay. A binary file containing hardware that can be overlayed on top of the existing .dtb file.
* .dts: Device Tree Source. This is typically the top level (board level) device tree description.
* .dtsi: Device Tree Source Include. These files are typically used to describe hardware on a SoC and in this case, the PL designs as well.
* .elf: Executable and Linkable Format. Contains compiled software.
* .wic: The .wic image helps simplify the process of deploying a platform project image to test by including the required boot, rootfs, and related partitions in the image. As a result, all you need to do is copy the image to a storage device and use it to boot the hardware target device.
* .xdc: Xilinx Design Constraint file. Indicate pin mapping, and pin constraints in Vivado.
* .xml: The XML board file is a configuration file used by Vivado to create board related configuration.
* .xclbin: Device binary file, also known as an AXLF file. It is an extensible, future-proof container for both (bitstream/platform) hardware and software (MPSoC/MicroBlaze ELF files) design data. In the flows above, the .xclbin file has information about the address space of the PL design.
* .xsa: Xilinx Shell Archive. Vivado generates these files to contain the required hardware information for developing embedded software with Vitis and can only be opened with AMD tools.


## KR260 DPU-TRD Petalinux 2022.1 

### STEP 1: Hardware Platform Generation.
In the folder prj_kria_2022, container bitstream generation files necessary for generating image KR260.xpr, top.bit,top_wrapper.xsa.

### STEP 2: Petalinux 2022.1 build from BSP.

1. Download [xilinx-kr260-starterkit-v2022.1-05140151.bsp](https://adaptivesupport.amd.com/s/article/000034113?language=en_US)
2.  Create the petalinux project , import source settings.sh in the petalinux directory
   ```

   petalinux-create -t project -s ../inputs/xilinx-kr260-starterkit-v2022.1-05140151.bsp --name dpuOS

   ```
3. Import the hardware platform to the petalinux project
     ```
       petalinux-config --get-hw-description=.../prj_kria_2022/prj/ 
     ```
4. In the configuration screen make the following settings,
   * Enable FPGA MANAGER
   * Disable TFTPboot copy
   * Image package type INITRD, name as petalinux-initramfs-image
     
5. Run the Kernel Configuration
   ```
   petalinux-config -c kernel
   Device Drivers -->
      Misc devices -->
            <*> Xilinux Deep learning Processing Unit (DPU) Driver
   ```
6. Copy the necessary recipes to our petalinux project directory
  ```
      cp -r ..../project-spec/meta-user/recipes-kernel/ ./project-spec/meta-user/
      cp -r .../project-spec/meta-user/recipes-tools/ ./project-spec/meta-user/
      cp -r .../project-spec/meta-user/recipes-vitis-ai/ ./project-spec/meta-user/
      cp -r .../project-spec/meta-user/recipes-apps/ ./project-spec/meta-user/
  ```
7. Append the CONFIG_x lines below to ..../project-spec/meta-user/conf/user-rootfsconfig file   
   ```
      CONFIG_vitis-ai-library
      CONFIG_vitis-ai-library-dev
      CONFIG_vitis-ai-library-dbg
   ```
8. Update petalinuxbsp.conf with the following lines.
   ```
      IMAGE_INSTALL:append = " vitis-ai-library "
      IMAGE_INSTALL:append = " vitis-ai-library-dev "
      IMAGE_INSTALL:append = " dpu-sw-optimize "
      IMAGE_INSTALL:append = " resnet50 "
   ```

9. Run the rootfs configuration. Select the required packages, Don't select vitis-ai-library-dbg, including GNURADIO

  ```
   petalinux-config -c rootfs
  ```
10. Build the project, time estimate around 1-2 hours
```
   petalinux-build
```
Optional 
11. Create the WIC petalinux package
```
   petalinux-package --wic --images-dir images/linux/ --bootfiles "ramdisk.cpio.gz.u-boot, boot.scr, Image, system.dtb,system-zynqmp-sck-kv-g-revB.dtb" --disk-name "mmcblk1" --wic-extra-args "-c gzip" 
```
### STEP 3: Generating the Device Tree Overlay
1. Source XSCT on path ...PetaLinuxTool/tools/xsct/bin
```   
./xsct
```
2. Creating a device tree domain and generating the device tree.
```     
   xsct% createdts -hw .../prj/top_wrapper.xsa -zocl -platform-name KR260 -git-branch xlnx_rel_v2022.1 -overlay -compile -out .../projects/prj/KR260_dt
   xsct% exit
```
3. Compile the device tree
```
dtc -@ -O dtb -o ./kr260.dtbo ./kr260_dt/kr260/psu_cortexa53_0/device_tree_domain/bsp/pl.dtsi

``` 
4. Create shell.json
```   
echo '{ "shell_type" : "XRT_FLAT", "num_slots": "1" }' > shell.json
```
5. Make a copy of top_wrapper.bin in a different directory and rename it to kr260.bit.bin
6. At this time, you should have the following files ready
   * petalinux-sdimage.wic.gz
   * kr260.bit.bin
   * kr260.dtbo
   * shell.json

### STEP 4: Boot the KR260 with petalinux
   1. Using Balena Etcher flash the petalinux-sdimage.wic.gz onto a 16 GB SD Card.
   2. login with username petalinux and set a new password.
   3. Set up the ethernet connection and have an IP address for SFTP.
Petalinux wic image 
[![Google Drive](https://img.shields.io/badge/Google%20Drive-Download-blue?logo=googledrive&logoColor=white)](https://drive.google.com/file/d/1aIK3qEIjd82fbDY_gnRQyypUypVNO_Hz/view?usp=drive_link)

### STEP 5: Creating an Accelerated application
1. Make a directory in your user space i.e., /home/petalinux
```
sudo mkdir myAPP
```
2. Copy kr260.bit.bin, kr260.dtbo, shell.json to myApp directory.
3. List the apps, and you should see the following output.
```   
sudo xmutil listapps
```
4. Move the myApp directory
```
sudo mv myApp/ /lib/firmware/xilinx/
``` 
5. List the apps and you should see the following output
```
sudo xmutil listapps
```
6. Unload the current application
```
sudo xmutil unloadapp
```   
7. Load the myApp application
```
sudo xmutil loadapp myApp
```
8. Execute show_dpu or xdputil query command
```
sudo show_dpu
device_core_id=0 device= 0 core = 0 fingerprint = 0x101000016010407 batch = 1 full_cu_name=unknown:dpu0
```
```
sudo xdputil query
{
"DPU IP Spec":{
"DPU Core Count":1,
"IP version":"v4.0.0",
"enable softmax":"False"
},
"VAI Version":{
"libvart-runner.so":"Xilinx vart-runner Version: 2.5.0-c26eae36f034d5a2f9b2a7bfe816b8c43311a4f8  2023-01-22-01:10:05 ",
"libvitis_ai_library-dpu_task.so":"Xilinx vitis_ai_library dpu_task Version: 2.5.0-c26eae36f034d5a2f9b2a7bfe816b8c43311a4f8  2022-06-15 07:33:00 [UTC] ",
"libxir.so":"Xilinx xir Version: xir-c26eae36f034d5a2f9b2a7bfe816b8c43311a4f8 2023-01-22-01:08:11",
"target_factory":"target-factory.2.5.0 c26eae36f034d5a2f9b2a7bfe816b8c43311a4f8"
},
"kernels":[
{
"DPU Arch":"DPUCZDX8G_ISA1_B4096",
"DPU Frequency (MHz)":275,
"cu_idx":0,
"fingerprint":"0x101000016010407",
"is_vivado_flow":true,
"name":"DPU Core 0"
}
]
}
```
### Compilation

1. Create an arch.json file to configure xmodel with the DPU information, "fingerprint" the dpu.bit file.  
arch_b1152.json
```
{
    "fingerprint":"0x101000016010407"
}
```
⚠️ Hit: It is needed to match between model_fingerprint and  dpu_fingerprint.
```
    WARNING: Logging before InitGoogleLogging() is written to STDERR
    W1007 14:49:17.954731 23652 dpu_runner_base_imp.cpp:676] CHECK fingerprint fail ! model_fingerprint 0x101000056010407 dpu_fingerprint 0x101000016010407
    F1007 14:49:17.956219 23652 dpu_runner_base_imp.cpp:648] fingerprint check failure.
```

2. Source command, rf_model.xmodel files generated by our model. 
```
!vai_c_tensorflow2 \
  -m quantize_results/quantized_model.h5 \
  -a arch_b1152.json \
  -o vai_c_output \
  -n rf_model  \
  --options '{"input_shape":"1,1024,1,2"}'

```
##  Login with UART

1. If you want to connect a Windows PC with the target board with UART, you can use TeraTerm, for example via COM27 port (of course this will depend on your computer), as illustrated in Figure 2.

2. If you have a Linux host computer, you can connect to the target board via PuTTY utility, you have to launch these commands (note that you have to select the second port - ttyUSB1 - among the four available ttyUSBX with X=0,1,2,3):
```  
# from your host PC
# search for USB devices with "tty" string and
# look at the second of the list (ttyUSB1)
dmseg | grep tty

# call PuTTY on ttyUSB1 for KR260
sudo putty /dev/ttyUSB1 -serial -sercfg 115200,8,n,1,N
```
![Putty](https://github.com/1Px-Vision/SAR-RF-Real-Time-RF-Signal-Classification/blob/main/Hardware/Putty.jpg)

## Testing DPU
Virtual Environment 
echo 'alias pynqenv="source /etc/profile.d/pynq_venv.sh"' >> ~/.bashrc
source ~/.bashrc

sudo -E bash -lc 'source /usr/local/share/pynq-venv/bin/activate && python3 /home/ubuntu/pynq_jupyter_notebooks/pynq-dpu/Test_Kria_E.py'


## Vivado, Vitis, & PetaLinux 2024.2 Install on WLS-Ubuntu
FPGA design IDEs seem to sprint ahead with every release—yes, ‘accelerating’ very much intended—so installation tips never go out of style. I’m back with notes for AMD’s 2025.1 tools after running into a few new twists during setup (and, unsurprisingly, devoting yet more disk space to Vivado/Vitis).

The Vivado ([HW Developer](https://www.xilinx.com/support/download/index.html/content/xilinx/en/downloadNav/vivado-design-tools.html)) and Vitis ([SW Developer](https://www.xilinx.com/support/download/index.html/content/xilinx/en/downloadNav/vitis.html)) download links point to the same installer; during setup, you choose to install Vivado, Vitis, or both

### WLS-Ubuntu Alias

```
echo 'alias vitis-gui="source /tools/Xilinx/2025.1/Vitis/settings64.sh && DISPLAY=:0.0 vitis"' >> ~/.bashrc
# (re)load your shell config
source ~/.bashrc
# Use it
vitis-gui

echo 'alias vivado-gui="source /tools/Xilinx/2025.1/Vivado/settings64.sh && DISPLAY=:0.0 vivado"' >> ~/.bashrc
# (re)load your shell config
source ~/.bashrc

# Use it
vivado-gui
```

### PetaLinux Install

To complete the Linux setup for Yocto-based PetaLinux, you must install and configure a TFTP server. Manually create the TFTP root directory and apply the correct permissions. Begin by creating the TFTP service rules file:

#### Install the required host packages
```
# Core dev + Yocto build deps
sudo apt install -y \
  gawk wget git-core diffstat unzip texinfo \
  gcc-multilib build-essential chrpath socat xterm \
  autoconf automake libtool pkg-config zlib1g-dev \
  libssl-dev libncurses5-dev libncursesw5 libtinfo5 \
  libselinux1 libsdl2-dev libglib2.0-dev \
  python3 python3-pexpect python3-git python3-jinja2 \
  cpio rsync bc iproute2 net-tools iputils-ping \
  bison flex gnupg file curl tar gzip xz-utils zstd liblz4-tool
```
#### Prep for PetaLinux Install
```
sudo apt update
sudo apt install -y xinetd tftpd-hpa

sudo nano /etc/xinetd.d/tftp

service tftp
    {
    protocol = udp
    port = 69
    socket_type = dgram
    wait = yes
    user = nobody
    server = /usr/sbin/in.tftpd
    server_args = /tftpboot
    disable = no
    }
```
Save and close the TFTP rules file, then create the TFTP root directory with the correct ownership and permissions

```
~$ sudo mkdir /tftpboot
~$ sudo chmod -R 777 /tftpboot
~$ sudo chown -R nobody /tftpboot
```

Finally, restart the xinetd network service for the changes to take effect.
```
~$ sudo /etc/init.d/xinetd stop
~$ sudo /etc/init.d/xinetd start
```

Make the desired installation directory, apply the correct permissions, and install PetaLinux in the versioned path that matches your Vivado/Vitis release.
```
~$ sudo mkdir -p /tools/Xilinx/PetaLinux/2024.2/
~$ sudo chmod -R 755 /tools/Xilinx/PetaLinux/2024.2/
~$ sudo chown -R <user>:<user> /tools/Xilinx/PetaLinux/2024.2/
```

##### Option A 
```
# 1) Create just the PetaLinux path (as root)
sudo mkdir -p /tools/Xilinx/PetaLinux/2024.2

# 2) Give YOUR user ownership of that path
sudo chown -R "$USER:$USER" /tools/Xilinx/PetaLinux/2024.2

# (Alternative: keep root owner but grant ACL to you)
# sudo setfacl -R -m u:$USER:rwx /tools/Xilinx/PetaLinux
# sudo setfacl -d -m u:$USER:rwx /tools/Xilinx/PetaLinux

# 3) Sanity check
ls -ld /tools/Xilinx/PetaLinux/2024.2
test -w /tools/Xilinx/PetaLinux/2024.2 && echo "write OK" || echo "NOT writable"

# 4) Re-run installer (note: non-root user)
./petalinux-v2024.2-11062026-installer.run --dir /tools/Xilinx/PetaLinux/2024.2

```
## Vivado Hardware Design

PS/clock/reset/IRQ backbone and the standard AXI touchpoints (GP0 + HP0)

![](https://github.com/1Px-Vision/SAR-RF-Real-Time-RF-Signal-Classification/blob/main/models/DPU_Vivado_Radio.jpg)

## Build the PetaLinux and GNU Radio

### Creating the PetaLinux project

Create a PetaLinux project and configure the hardware with the XSA file created

```
# (env, once per shell)
source /tools/Xilinx/PetaLinux/202x.x/settings.sh

petalinux-create -t project --template zynq -n zedboard_202x_x-petalinux
cd zedboard_202x_x-petalinux
petalinux-config --get-hw-description=/tools/Xilinx/PetaLinux/202x.x/zedboard_202x_x-petalinux/zed_accel_x/zed_dpu_radio_wrapper.xsa

```
![PConf](https://github.com/1Px-Vision/SAR-RF-Real-Time-RF-Signal-Classification/blob/main/Hardware/Petalinux_Config.jpg)

### Modifying the Device Tree
We need to add some driver configurations for SD card. Modify ``` system-user.dtsi ``` file 
```
/include/ "system-conf.dtsi"
/ {
};

/* SD */
&sdhci1 {
        disable-wp;
        no-1-8-v;
};
```
### Config the Kernel

Run the following command
```
petalinux-config -c kernel
```

![conf2](https://github.com/1Px-Vision/SAR-RF-Real-Time-RF-Signal-Classification/blob/main/Hardware/Petalinux_Config_2.jpg)

#### Disable CPU IDLE in kernel config:

* CPU Power Management > CPU Idle > CPU idle PM support - set 'N' n the [ ] menu selection.
* CPU Power Management > CPU Frequency scaling > CPU Frequency scaling - set 'N'.
  
### GNU Radio software to RootFS file system

Yocto layer for the following SW to build in the system:

* **GNU Radio v3.8** - main program
* **gr-osmosdr v3.8** - module needed for SDR receivers
* **gr-fpga_ai v3.8** - module needed for accessing DPU on FPGA

#### Config the RootFS file system

```
grep -n 'LAYERSERIES_COMPAT' /tools/Xilinx/PetaLinux/202x.x/zedboard_202x_x-petalinux/project-spec/meta-sdr/conf/layer.conf

```

```
 nano project-spec/meta-user/conf/user-rootfsconfig file.

Note: Mention Each package in individual line
#These packages will get added into rootfs menu entry

CONFIG_gpio-demo
CONFIG_peekpoke

# Packages for base XRT support
CONFIG_xrt

# ackages for easy system management
CONFIG_dnf
CONFIG_e2fsprogs-resize2fs
CONFIG_parted
CONFIG_resize-part

# Packages for Vitis-AI dependencies support
CONFIG_packagegroup-petalinux-vitisai

# Optional Packages for natively building 
# Vitis AI applications on target board
CONFIG_packagegroup-petalinux-self-hosted
CONFIG_cmake
CONFIG_packagegroup-petalinux-vitisai-dev
CONFIG_xrt-dev
CONFIG_opencl-clhpp-dev
CONFIG_opencl-headers-dev
CONFIG_packagegroup-petalinux-opencv
CONFIG_packagegroup-petalinux-opencv-dev

# Optional Packages for running i
# Vitis-AI demo applications with GUI
CONFIG_mesa-megadriver
CONFIG_packagegroup-petalinux-x11
CONFIG_packagegroup-petalinux-v4lutils
CONFIG_packagegroup-petalinux-matchbox

# Packages for date and time settings
CONFIG_ntp
CONFIG_ntpdate
CONFIG_ntp-utils

# Some utils
CONFIG_git
CONFIG_zip
CONFIG_unzip

# Gnuradio and its modules
CONFIG_gnuradio
CONFIG_gr-osmosdr
CONFIG_gr-fpga-ai

```
Config the rootfs by running this command:

```
petalinux-config -c rootfs
```
* User Packages. Enable all, don't forget gnuradio, gr-osmosdr, and gr-fpga-ai.
* Enable OpenSSH and disable dropbear: Image Features-> Disable ssh-server-dropbear and enable ssh-server-openssh.
* Filesystem Packages -> misc -> packagegroup-core-ssh-dropbear and disable packagegroup-core-ssh-dropbear.
* Filesystem Packages -> console -> network -> openssh and enable openssh, openssh-sftp-server, openssh-sshd, openssh-scp.
*  Image Features and enable package-management and debug_tweaks.

### Image Feature Settings
SD card image ZCU104 for RF Modulation Recognition on DPU-FPGA with GNU Radio 

* **Build PetaLinux Images**
Run the command and wait for several hours to finish:
```
petalinux-build
```

Finally, verify that all images (rootfs, Image, u-boot, etc.) are built successfully in the image/linux directory. 
Check the time of file creation.
```
$ ls -lrt images/Linux
```

* **Build SDK**
Run the build --sdk command and wait for about an hour for the process to finish.
```
petalinux-build --sdk
```

## Vitis Platform with DPU for AI Inference

### Prepare Files for Platform Packaging
Created "platform" directory under your project main directory, or create it if doesn’t exist."
```
pwd
/tools/workspace/zcu104
mkdir platform
ls
hardware  platform  software
cd platform
```
New directories for storing some previously created files.
```
mkdir -p pfm/boot
mkdir -p pfm/sd_dir
```

Copy the generated Linux software boot components to pfm/boot directory.
```
$ cp ../xxx/images/linux/zynqmp_fsbl.elf pfm/boot/
$ cp ../xxx/images/linux/pmufw.elf pfm/boot/
$ cp ../xxx/images/linux/bl31.elf pfm/boot/
$ cp ../xxx/images/linux/system.dtb pfm/boot/
$ cp ../xxx/images/linux/u-boot-dtb.elf pfm/boot/u-boot.elf

$ ls pfm/boot/
bl31.elf  pmufw.elf  system.dtb  u-boot.elf  zynqmp_fsbl.elf
```
Copy the boot.scr and system.dtb to pfm/sd_dir folder.
```
$ cp ../xxx/images/linux/boot.scr pfm/sd_dir/
$ cp ../xxx/images/linux/system.dtb pfm/sd_dir/

$ ls pfm/sd_dir/
boot.scr  system.dtb
```
Install sysroot into ``` pfm ``` folder. Before installing sysroot, you must unset the system variable ``` LD_LIBRARY_PATH ``` . Install the sysroot from the previously generated ```sdk.sh``` script.

```
 ../xxx/images/linux/sdk.sh -d pfm

```

##  Performance Test
Run the benchmark from the target board’s serial or SSH terminal.

```
Usage: python3 test_performance.py <threads> <model.xmodel> <num_frames>
```

* RF frames are read from rf_input.npy; each frame contains 1024 I/Q samples.

* The script prints the measured performance.

* The sample results shown below were obtained on a ZCU104 with two 4096-MAC DPU cores running at 300 MHz (~2.45 peak INT8 TOPS).

## Accuracy Test

On the target board (serial console or SSH), run:

```
Usage: python3 test_accuracy.py rfClassification.xmodel
```

The script reads evaluation data from:

* rf_input.npy – RF frames

* rf_snr.npy* – per-frame SNR values

* rf_classes.npy – ground-truth modulation classes

## Hardware/Software

* Edge: Zedboard/Ultra96-class DPU (Vitis AI) or NVIDIA GPU/Jetson.

* I/O: RTL-SDR (default), pluggable front-ends.

* Runtime: Python, FastAPI (server), Dash/Plotly (UI).

* Models: CNN/ResNet-style; logits-only export for DPU; Vitis-AI quantization (INT8).

## Roadmap

* On-drone deployment (companion computer), power budgeting, and thermal envelopes.

* Bearing/DOA fusion and geotagged detections.

* Dataset expansion, domain adaptation, and semi-supervised updates.

* Optional on-device recording & replay for after-action review.

## Datasets & model shape

* Input tensor: (batch, 1024, 1, 2) (real/imag channels).

* Label space: default 10-class modulation set (easily extendable).

* Training: cross-entropy from logits (no in-graph softmax), weighted for class imbalance; robust augmentation across SNR (e.g., −20 dB to +30 dB), frequency offset, IQ imbalance.

## DPU-FPGA deployment (overview)

1. Train TF/Keras model → output logits.

2. Quantize with Vitis-AI (calibration dataset) → INT8 Keras model.

3. Compile with vai_c_tensorflow2 for your arch.json → produce .xmodel.

4. Run with VART on the target (Zynq/Kria), expose an Ethernet API (/infer) for the drone/ground UI.

5. Do softmax on host (DPU outputs logits/INT8 scores).


## Real-time constraints & target metrics

* **End-to-end latency (1×1024 window):** ≤10 ms (goal, DPU) / ≤20 ms (GPU mobile).

* **Throughput:** ≥100 FPS per stream on embedded; scalable via batching on GPU.

* **Power:** <6–8 W (DPU SoC node), 15–30 W (Jetson class).

* **Robustness:** stable classification from −10 dB SNR upward with temporal smoothing.

## Drone integration plan

* **Form factors:** SDR + DPU SoC or SDR + Jetson on companion computer.

* **Interfaces:** Ethernet to flight computer, optional MAVLink status, GNSS (NMEA) for geo-tagging.

* **Operational modes:** fixed-channel monitoring, band scanning, triggered capture, geo-fence alerts.

* **Data products:** time-stamped detections, confidence, SNR, (future) coarse bearings.

# Single-board SDR platform.
Built around the Xilinx Zynq-7020 SoC and an AD936x RF transceiver (AD9363/AD9361/AD9364), this board integrates 512 MB DDR3L, 16 MB flash, and MicroSD for boot/configuration. I/O includes USB OTG, USB-JTAG/UART, and Ethernet. The Zynq architecture combines an ARM Cortex-A9 processing system with programmable logic, delivering a highly flexible platform where workloads can run on the ARM cores, the FPGA fabric, or a combination of both.

![HW_SDR](https://github.com/1Px-Vision/SAR-RF-Real-Time-RF-Signal-Classification/blob/main/Hardware/HW_SDR.jpg)

