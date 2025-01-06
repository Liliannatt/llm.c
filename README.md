# llm.c

# LLM Computation Acceleration with Tensor Core

## Project Overview

We are accelerating Large Language Model (LLM) computation using NVIDIA Tensor Cores, focusing on mixed precision techniques via cuBLAS or the WMMA API. We'll use [Karpathy's llm.c](https://github.com/karpathy/llm.c) as our base implementation.

## Team Members and Responsibilities

### Member A: Background Research and Problem Analysis

- **Tasks:**
  - Research Tensor Core technology and mixed precision computation.
  - Identify performance bottlenecks using profiling tools (e.g., NVIDIA Nsight).
  - Write the **Introduction**, **Results** and **References** sections of the report.

### Member B: Optimization Design and Implementation

- **Tasks:**
  - Design optimization strategies using cuBLAS or WMMA API.
  - Implement mixed precision computation in the code.
  - Write the **Methodology** and **Implementation** sections of the report.

### Member C: Experimentation and Results Analysis

- **Tasks:**
  - Set up the GPU environment and necessary software.
  - Analyze results and compare with the original implementation.
  - Write the **Experimental Setup**, **Results**, and **Discussion and Conclusion** sections of the report.



---

# LLM计算加速与Tensor Core

## 项目概述

我们将使用NVIDIA Tensor Core加速大型语言模型（LLM）的计算，重点关注通过cuBLAS或WMMA API实现的混合精度技术。我们将使用[Karpathy的llm.c](https://github.com/karpathy/llm.c)作为基础实现。

## 组员及职责分工

### 成员A：背景研究和问题分析

- **任务：**
  - 研究Tensor Core技术和混合精度计算。
  - 使用Profiling工具（如NVIDIA Nsight）识别性能瓶颈。
  - 撰写报告的**引言**和**参考文献**部分。

### 成员B：优化设计与实现

- **任务：**
  - 设计使用cuBLAS或WMMA API的优化策略。
  - 在代码中实现混合精度计算。
  - 撰写报告的**方法**和**实现**部分。

### 成员C：实验与结果分析

- **任务：**
  - 搭建GPU环境和必要的软件。
  - 分析结果并与原始实现进行比较。
  - 撰写报告的**实验设置**、**结果**和**讨论与结论**部分。



## quick start (1 GPU, BF16 only)

```bash
chmod u+x ./dev/download_starter_pack.sh
./dev/download_starter_pack.sh
make train_gpt2cu PRECISION=BF16 NO_MULTI_GPU=1
./train_gpt2cu -i dev/data/tinyshakespeare/tiny_shakespeare_train.bin \
               -j dev/data/tinyshakespeare/tiny_shakespeare_val.bin \
               -v 250 -s 250 -g 144 -f shakespeare.log -b 4
```
