**Quick Run Instructions**

- **HF login (preferred):**

  ```bash
  hf login
  ```

- **Run `main.py` with `hf` from repo root:**

  ```bash
  cd /home/nextai/Documents/AI_Agents/theneuralmaze.substack.com/finetuning_sessions
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
