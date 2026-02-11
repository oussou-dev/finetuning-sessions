# Tech Stack

~~~
An overview of the tools, frameworks and services
~~~

<br>

- [Unsloth](https://unsloth.ai/)  
is a high-performance library designed to make finetuning large language models faster and more memory-efficient. 
Not need to install Unsloth locally. we'll run it:
    - inside Google Colab notebooks, and 
    - Inside Hugging Face Jobs  

![](https://substackcdn.com/image/fetch/$s_!hPtg!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F5481d5e6-5196-461c-b877-f0a7b2fa401b_1200x600.png)

<br>

- [Comet ecosystem](https://www.comet.com/site/)  
provides the tooling we'll use to track, analyze, and evaluate our finetuning experiments.
We'll use Comet to: 
    - Create experiments when launching finetuning jobs
    - Log training metrics and losses
    - Compare multiple runs side by side
    - Identify which hyperparameters perform best for a given use case
<br>

    [Opik](https://github.com/comet-ml/opik), the LLM-focused layer of the Comet ecosystem, will be used to evaluate the LLM systems we build throughout the course. Specifically, we'll use Opik to:
        - Run batch evaluations on model outputs
        - Perform real-time evaluations during inference
        - Analyze and compare different model versions as systems evolve

Together, Comet and Opik give us visibility into both how models are trained and how they behave in practice, which is essential for building reliable LLM applications.

![](https://substackcdn.com/image/fetch/$s_!f4RL!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F02276c6b-9b20-43c3-806a-b13950c32e53_1200x427.png)

<br>


- [Hugging Face Jobs](https://huggingface.co/docs/huggingface_hub/guides/jobs)
to run our finetuning workloads on Hugging Face–managed infrastructure. Jobs execute commands from a Docker image on managed compute, with access to GPUs.

![](https://substackcdn.com/image/fetch/$s_!rlEC!,w_1456,c_limit,f_webp,q_auto:good,fl_lossy/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F51834c50-d2fd-42c4-9b89-e0aee93e83ba_1280x314.gif)


<br>


- [Hugging Face Inference Endpoints](https://huggingface.co/docs/inference-endpoints/index)
let us deploy models quickly without worrying about infrastructure setup, server management, or deployment complexity. 

![](https://substackcdn.com/image/fetch/$s_!phec!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F50282ff4-1f5b-469a-86f0-b89c74a6308a_2645x957.png)


<br>

### Scripts
to validate that everything is working as expected.

- inference_sample.py
- main.py

<br>

**Hugging Face Job**
```
hf jobs uv run --flavor a10g-small main.py --input_text "'The answer is 42'"
```

**Hugging Face Endpoint**
```
python inference_sample.py "The capital of France is" --model "Qwen/Qwen3-0.6B-Base" --max_tokens 1024
```

<br>

### **Quick Run Instructions**

- **HF login (preferred):**

  ```bash
  hf login
  ```

- **Run `main.py` with `hf` from repo root:**

  ```bash
  cd /finetuning_sessions
  hf jobs uv run --flavor a10g-small 0_lab-0_tech-stack/main.py --input_text "'The answer is 42'"
  ```

- **Or run from the script directory:**

  ```bash
  cd 0_lab-0_tech-stack
  hf jobs uv run --flavor a10g-small main.py --input_text "'The answer is 42'"
  ```

- **If using `inference_sample.py` locally you must set these env vars:**

  ```bash
  export HF_API_TOKEN="hf_YOUR_TOKEN"
  export HF_ENDPOINT_URL="https://your-hf-endpoint.example.com/v1/"
  ```

- **Run `inference_sample.py` (local test):**

  ```bash
  python3 inference_sample.py --prompt "The answer is 42" --model "your-model-id"
  ```

- **Or pass URL/key directly:**

  ```bash
  python3 inference_sample.py --prompt "Hello" --model "your-model-id" \
    --api_url "https://your-hf-endpoint.example.com/v1/" --api_key "hf_YOUR_TOKEN"
  ```

- **Notes:**
  - `inference_sample.py` expects `HF_ENDPOINT_URL` that ends with `/v1/` (the script will append it if missing).
  - If `hf` reports FileNotFoundError, check your current working directory or pass the correct relative path to the script.
