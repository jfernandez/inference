# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Rust-based HTTP inference server for LLM text generation using the Qwen2-0.5B model. The server provides a REST API endpoint for generating text responses from prompts using GPU-accelerated inference via CUDA.

## Architecture

- **Async HTTP Server**: Uses Axum framework with CORS support on port 3000
- **Model Worker**: Dedicated async task handling inference requests via mpsc channels
- **Request Flow**: HTTP requests → channel → model worker → response channel → HTTP response
- **KV Cache**: Implements key-value caching for efficient autoregressive generation
- **GPU Acceleration**: Uses CUDA through Candle framework for tensor operations

## Development Commands

- **Build**: `cargo build`
- **Run**: `cargo run`
- **Test**: `cargo test`
- **Lint**: `cargo clippy`
- **Check**: `cargo check`

## Dependencies & Candle Fork

This project uses a custom Candle fork:
- **Fork Location**: `~/Code/candle` (local development)
- **Remote**: `github.com/jfernandez/candle.git`
- **Branch**: `cudarc-0.17.3`
- **Cargo Dependencies**: All candle crates point to this fork

The fork is used for custom CUDA implementations and compatibility fixes.

## Current Known Issues

### RMS Normalization CUDA Error
- **Error**: "Forward pass failed at step 0: no cuda implementation for rms-norm"
- **Location**: Occurs during model forward pass in `process_inference` function
- **Impact**: Prevents inference from completing
- **Context**: Qwen2 model uses RmsNorm layers that require CUDA kernel implementations

## Hardware Requirements

- **GPU**: NVIDIA GPU with CUDA support (tested with RTX 5080, compute capability 12.0)
- **CUDA**: Compatible CUDA installation required for GPU acceleration

## Model Details

- **Model**: Qwen/Qwen2-0.5B from HuggingFace
- **Loading**: Uses safetensors format with mmap for memory efficiency
- **Tokenizer**: HuggingFace tokenizer with EOS token 151643
- **Generation**: Greedy decoding (argmax) with configurable max tokens (50 default)

## API Usage

```bash
curl -X POST http://localhost:3000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What year were you trained?"}'
```