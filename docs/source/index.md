# Welcome to LLM compressor

:::{figure} ./assets/logos/llm-compressor-logo.png
:align: center
:alt: LLM compressor
:class: no-scaled-link
:width: 60%
:::

:::{raw} html
<p style="text-align:center">
<strong>Easy, fast, and cheap LLM serving for everyone
</strong>
</p>

<p style="text-align:center">
<script async defer src="https://buttons.github.io/buttons.js"></script>
<a class="github-button" href="https://github.com/vllm-project/llm-compressor" data-show-count="true" data-size="large" aria-label="Star">Star</a>
<a class="github-button" href="https://github.com/vllm-project/llm-compressor/subscription" data-icon="octicon-eye" data-size="large" aria-label="Watch">Watch</a>
<a class="github-button" href="https://github.com/vllm-project/llm-compressor/fork" data-icon="octicon-repo-forked" data-size="large" aria-label="Fork">Fork</a>
</p>
:::

`llmcompressor` is an easy-to-use library for optimizing models for deployment with `vllm`, including:

* Comprehensive set of quantization algorithms for weight-only and activation quantization
* Seamless integration with Hugging Face models and repositories
* `safetensors`-based file format compatible with `vllm`
* Large model support via `accelerate`

<p align="center">
   <img alt="LLM Compressor Flow" src="https://github.com/user-attachments/assets/adf07594-6487-48ae-af62-d9555046d51b" width="80%" />
</p>

## Supported Formats
* Activation Quantization: W8A8 (int8 and fp8)
* Mixed Precision: W4A16, W8A16
* 2:4 Semi-structured and Unstructured Sparsity

## Supported Algorithms
* Simple PTQ
* GPTQ
* SmoothQuant
* SparseGPT

## When to Use Which Optimization

Please refer to our [schemes document](./getting_started/schemes.md) for detailed information about available optimization schemes and their use cases.


## Installation

```bash
pip install llmcompressor
```

## Get Started

### End-to-End Examples

Applying quantization with `llmcompressor`:
* [Activation quantization to `int8`](getting_started/examples/quantization_w8a8_int8.md)
* [Activation quantization to `fp8`](getting_started/examples/quantization_w8a8_fp8.md)
* [Weight only quantization to `int4`](getting_started/examples/quantization_w4a16.md)
* [Quantizing MoE LLMs](getting_started/examples/quantizing_moe.md)
* [Quantizing Vision-Language Models](getting_started/examples/multimodal_vision.md)
* [Quantizing Audio-Language Models](getting_started/examples/multimodal_audio.md)

### User Guides
Deep dives into advanced usage of `llmcompressor`:
* [Quantizing with large models with the help of `accelerate`](getting_started/examples/big_models_with_accelerate.md)


## Quick Tour
Let's quantize `TinyLlama` with 8 bit weights and activations using the `GPTQ` and `SmoothQuant` algorithms.

Note that the model can be swapped for a local or remote HF-compatible checkpoint and the `recipe` may be changed to target different quantization algorithms or formats.

### Apply Quantization
Quantization is applied by selecting an algorithm and calling the `oneshot` API.

```python
from llmcompressor.modifiers.smoothquant import SmoothQuantModifier
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor import oneshot

# Select quantization algorithm. In this case, we:
#   * apply SmoothQuant to make the activations easier to quantize
#   * quantize the weights to int8 with GPTQ (static per channel)
#   * quantize the activations to int8 (dynamic per token)
recipe = [
    SmoothQuantModifier(smoothing_strength=0.8),
    GPTQModifier(scheme="W8A8", targets="Linear", ignore=["lm_head"]),
]

# Apply quantization using the built in open_platypus dataset.
#   * See examples for demos showing how to pass a custom calibration set
oneshot(
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    dataset="open_platypus",
    recipe=recipe,
    output_dir="TinyLlama-1.1B-Chat-v1.0-INT8",
    max_seq_length=2048,
    num_calibration_samples=512,
)
```

### Inference with vLLM

The checkpoints created by `llmcompressor` can be loaded and run in `vllm`:

Install:

```bash
pip install vllm
```

Run:

```python
from vllm import LLM
model = LLM("TinyLlama-1.1B-Chat-v1.0-INT8")
output = model.generate("My name is")
```

## Questions / Contribution

- If you have any questions or requests open an [issue](https://github.com/vllm-project/llm-compressor/issues) and we will add an example or documentation.
- We appreciate contributions to the code, examples, integrations, and documentation as well as bug reports and feature requests! [Learn how here](./CONTRIBUTING.md).

## Where to go next

Start with the Getting Started guide.

:::{toctree}
:caption: Getting Started
:maxdepth: 2

getting_started/examples/big_models_with_accelerate.md
getting_started/examples/examples_index.md
getting_started/examples/finetuning.md
getting_started/examples/multimodal_audio.md
getting_started/examples/multimodal_vision.md
getting_started/examples/quantization_2of4_sparse_w4a16.md
getting_started/examples/quantization_kv_cache.md
getting_started/examples/quantization_w4a16.md
getting_started/examples/quantization_w8a8_fp8.md
getting_started/examples/quantization_w8a8_int8.md
getting_started/examples/quantizing_moe.md
getting_started/examples/sparse_2of4_quantization_fp8.md
getting_started/examples/trl_mixin.md
:::