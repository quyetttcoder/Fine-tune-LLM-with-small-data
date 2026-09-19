# Vietnamese University Admissions LLM Fine-Tuning and Deployment Pipeline

This repository outlines the pipeline for fine-tuning, quantizing, serving, and exposing two open-source Large Language Models—**Qwen3-8B** and **Llama-3.1-8B**—for Vietnamese university admissions consulting. 

The project is structured as a collection of modular **Google Colab notebooks** covering each phase independently.

---

## Project Overview

1. **Base Models:** Qwen3-8B and Llama-3.1-8B.
2. **Dataset:** A specialized dataset containing approximately 1,117 question-answer pairs focused on university admissions in Vietnam.
3. **Fine-Tuning:** Performed using **Unsloth** for memory-efficient and rapid parameter-efficient fine-tuning (PEFT/LoRA).
4. **Quantization:** The fine-tuned adapters/models are quantized to optimize memory footprint and ensure full compatibility with vLLM inference engines.
5. **Serving & Tunneling:** Models are served locally via **vLLM** (compatible with OpenAI's API specs) and exposed to the public internet using **ngrok**, allowing seamless integration into downstream applications via OpenAI client SDKs.

---

 **Note:** All parameters, configurations, and fine-tuned model paths in this project are provided for reference purposes only and should be customized according to the specific requirements and directory structure of your target project.

## Pipeline Workflow & Google Colab Notebooks

Follow these sequential steps by executing the corresponding Google Colab notebooks.

### Step 1: Fine-Tuning with Unsloth
Fine-tune the base models using your admissions dataset (1,117 samples).

* **Notebook Focus:** Data preparation, formatting into Q/A templates, setting up Unsloth configurations, and running the training loop.
* **Outputs:** Saved LoRA adapters or merged model weights.
* **Models to run:** 
  * `fine_tune_qwen3-8b_v3.ipynb`
  * `Fine_tune_Llama31_8B.ipynb`

### Step 2: Model Quantization
Quantize the fine-tuned model to prepare it for high-performance deployment.

* **Notebook Focus:** Loading the fine-tuned model weights and applying quantization techniques to ensure compatibility with vLLM serving optimization.
* **Outputs:** Quantized model directory ready for inference.

### Step 3: Serving with vLLM and Exposing via ngrok
Serve the quantized model using vLLM to provide an OpenAI-compatible API endpoint, then tunnel it with ngrok for external access.

* **Notebook Focus:** Launching the vLLM server via Python subprocesses or CLI, checking health endpoints, and establishing an ngrok tunnel.

You can run vLLM via the command line with custom parallelization and resource constraints:

```bash
vllm serve "path/to/your/quantized/model" \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 1536
```

---

## Integration with OpenAI API

Because vLLM exposes an OpenAI-compliant REST API, you can seamlessly query your fine-tuned university admissions model using the standard OpenAI Python client by changing the `base_url`:

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://<your-ngrok-subdomain>.ngrok-free.app/v1",
    api_key="not-needed-for-local-vllm"
)

response = client.chat.completions.create(
    model="path/to/your/quantized/model",
    messages=[
        {"role": "system", "content": "You are an expert Vietnamese university admissions consultant."},
        {"role": "user", "content": "Điều kiện xét tuyển học bạ của Đại học Bách Khoa Hà Nội năm nay thế nào?"}
    ],
    temperature=0.7,
)

print(response.choices[0].message.content)
