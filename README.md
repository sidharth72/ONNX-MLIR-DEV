# ONNX-MLIR Docker Workspace

This project provides a Docker-based environment for experimenting with [ONNX-MLIR](https://github.com/onnx/onnx-mlir), a compiler that lowers ONNX models to optimized machine code using MLIR (Multi-Level Intermediate Representation). The workspace includes scripts and pre-built models (e.g., GPT-2) to help you get started with ONNX-MLIR compilation and inference.

---

## What is ONNX-MLIR?

**ONNX-MLIR** is an open-source compiler that translates [ONNX](https://onnx.ai/) (Open Neural Network Exchange) models into highly optimized code for various hardware backends. It leverages the [MLIR](https://mlir.llvm.org/) compiler infrastructure to enable modular, extensible, and efficient compilation pipelines for machine learning models.

**Key Features:**
- Converts ONNX models to native code (e.g., shared libraries, executables).
- Supports multiple hardware targets (CPU, GPU, etc.).
- Enables advanced optimizations via MLIR's intermediate representations.

---

## How to Run

### 1. Build the Docker Image

From the project root, build the Docker image:

```sh
docker build -t onnx-mlir-dev-env .
```

### 2. Start the Docker Container

Mount your workspace and start an interactive shell:

```sh
docker run --rm -it -v $(pwd)/workspace:/workdir/workspace onnx-mlir-dev-env /bin/bash
```

### 3. Compile and Run Models

- Place your ONNX models in the `workspace/models/` directory.
- Use the provided Python scripts (e.g., `run_inference.py`) or ONNX-MLIR CLI tools inside the container to compile and run inference.

Example (inside the container):

```sh
cd /workdir/workspace
python3 run_inference.py --model gpt2_124M.onnx
```

---

## ONNX-MLIR Lowering Stages

ONNX-MLIR uses MLIR to progressively lower ONNX models through several intermediate representations (dialects), each enabling specific optimizations and transformations:

### 1. ONNX Dialect

- Represents the original ONNX operations in MLIR.
- Closely matches the ONNX standard.

### 2. Krnl Dialect

- A custom dialect for ONNX-MLIR.
- Provides higher-level abstractions for loops, memory, and computations.
- Enables optimizations before lowering to lower-level dialects.

### 3. Affine/Standard Dialects

- Lower-level MLIR dialects.
- Affine: Used for loop transformations and memory optimizations.
- Standard: Represents generic computations and control flow.

### 4. LLVM Dialect

- The final MLIR dialect before code generation.
- Maps directly to LLVM IR, enabling native code generation for CPUs and other targets.

---

## Key Learnings

- **ONNX-MLIR bridges the gap** between portable ONNX models and efficient, hardware-specific code.
- **MLIR's modular design** allows for extensible compilation pipelines and custom optimizations.
- **Lowering stages** (ONNX → Krnl → Affine/Standard → LLVM) enable progressive optimization and transformation.
- **Dockerization** ensures a reproducible and isolated environment for development and experimentation.

---

## Project Structure

```
DockerOnnxMlir/
├── Dockerfile
├── README.md
└── workspace/
    ├── PyCompileAndRuntime.py
    ├── PyRuntime.py
    ├── run_inference.py
    └── models/
        ├── gpt2_124M.onnx
        └── ...
```

---

## References

- [ONNX-MLIR GitHub](https://github.com/onnx/onnx-mlir)
- [MLIR Documentation](https://mlir.llvm.org/)
- [ONNX Model Zoo](https://github.com/onnx/models)

```
