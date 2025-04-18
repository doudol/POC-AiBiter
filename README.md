# AiBiter: Packaging and Loading Pre-Quantized LLMs (Early Proof-of-Concept)

## Abstract

This document details an early Proof-of-Concept (PoC) investigating methods for packaging and loading pre-quantized Large Language Models (LLMs) for direct inference. The PoC successfully demonstrated that a `state_dict` saved from a `bitsandbytes` INT8-quantized `gpt2` model, when packaged alongside its tokenizer and metadata into a preliminary `.aibit` archive, can be subsequently loaded into a natively configured 8-bit `transformers` model instance for inference without runtime weight dequantization. This validates the basic workflow of serializing and deserializing quantized model state within a custom package.

## 1. Introduction

Deploying large LLMs involves challenges related to model size, memory requirements, and initialization latency. While quantization reduces model size, the process of loading and preparing these models can still be complex. This work explores AiBiter, a concept for an integrated format designed to package all necessary components (quantized weights, tokenizer data, configuration, potentially optimized graph elements) for streamlined loading and direct execution. This PoC focuses on the foundational step: packaging and reloading pre-quantized weights.

## 2. Proof-of-Concept Implementation

The PoC utilized Python, PyTorch, `transformers`, and `bitsandbytes`:

1.  **Conversion & Packaging:**
    *   A standard `gpt2` model was loaded using `transformers`.
    *   It was quantized to INT8 using `bitsandbytes` via the `load_in_8bit=True` argument in `from_pretrained`. This step inherently reduces the memory footprint of the weights compared to FP16.
    *   The resulting `state_dict` (containing INT8 weights and quantization state) was saved using `torch.save()`.
    *   This `state_dict`, the standard tokenizer files (`tokenizer.save_pretrained()`), and basic metadata (model type, quantization details) were packaged into a ZIP archive (representing the preliminary `.aibit` format).
2.  **Loading & Execution:**
    *   A separate script extracted the contents of the `.aibit` archive.
    *   It instantiated the `gpt2` model structure *natively configured for 8-bit execution* using `AutoModelForCausalLM.from_pretrained(..., quantization_config=BitsAndBytesConfig(load_in_8bit=True), ...)`. This ensures the model structure contains the necessary `bitsandbytes` layers (e.g., `Linear8bitLt`).
    *   The `state_dict` previously saved to the archive was loaded from disk (`torch.load()`).
    *   This loaded `state_dict` was then successfully applied to the natively initialized 8-bit model instance (`model.load_state_dict(..., strict=False)`).
    *   Inference was performed using `model.generate()`. The underlying computation utilized the loaded INT8 weights via the `bitsandbytes` layers without requiring a separate weight dequantization step during the loading phase.

## 3. Key Result & Significance

The primary outcome of this PoC is the **successful validation of the package-load-execute workflow for pre-quantized weights**. It demonstrates that:
a) A `state_dict` derived from a `bitsandbytes` 8-bit quantized model can be serialized.
b) This serialized state, along with other necessary artifacts like the tokenizer, can be packaged.
c) The packaged `state_dict` can be successfully loaded back into a correctly, natively configured 8-bit model instance in a separate environment, enabling direct inference using the pre-quantized weights.

This confirms the basic feasibility of distributing and reusing pre-quantized model states via a custom package format. It serves as a necessary foundation before implementing AiBiter's *intended* optimizations. **This PoC itself does not demonstrate performance or compression advantages over existing optimized formats like GGUF**, which employ techniques like memory mapping (`mmap`) for efficient loading. The significance here is purely the validation of the described serialization and reloading pipeline for quantized weights within our packaging concept.

## 4. Future Work

Building upon this validated foundation, planned developments for the AiBiter format include:

*   **Actual Compression & Optimization:** Integrating techniques beyond basic INT8, such as INT4/NF4 quantization, structured pruning, and weight clustering, directly within the format specification.
*   **Tokenizer Optimization:** Implementing and packaging vocabulary subsetting/remapping.
*   **Graph Optimization:** Investigating storage of pre-compiled/optimized inference sub-graphs.
*   **Custom Binary Format:** Replacing the PoC's ZIP archive with a format optimized for loading efficiency, potentially leveraging `mmap`.
*   **Benchmarking:** Quantitatively comparing the performance (load time, inference speed, memory usage) and compression ratios of future AiBiter versions against established formats like GGUF.

## 5. Status

AiBiter remains highly experimental. This PoC focused solely on validating the core workflow of packaging and reloading a pre-quantized INT8 state dict into a compatible runtime model. Significant development and benchmarking are required to implement and evaluate the planned optimizations.
