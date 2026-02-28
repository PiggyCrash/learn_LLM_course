import os
import time
import subprocess
from pathlib import Path
from dotenv import load_dotenv
import requests

ROOT_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = ROOT_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH)

CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_NAME = os.getenv("LLM_MODEL", "llama3.1")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", 300))
TEMPERATURE = float(os.getenv("TEMPERATURE", 0.7))
CEREBRAS_MODEL = os.getenv("CEREBRAS_MODEL", "llama3.1")

SYSTEM_PROMPT = """You are an autonomous AI Agent. Provide structured reasoning and final answer."""

def measure_memory():
    import psutil
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 2)  # MB

def print_metrics(mode, latency, memory, output):
    print(f"--- {mode.upper()} ---")
    print(f"Latency: {latency:.2f}s | Memory: {memory:.2f} MB | Output length: {len(output)} chars")
    print(f"Output:\n{output}\n{'='*50}\n")

_tokenizer_cache = {}
_model_cache = {}

def get_library_model(model_name):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError
    import torch
    
    full_model_path = "meta-llama/Llama-3.1-8B-Instruct" if model_name == "llama3.1" else model_name
    
    if full_model_path not in _model_cache:
        print(f"--- Loading library model: {full_model_path} (this may take a few minutes) ---")
        if full_model_path == "meta-llama/Llama-3.1-8B-Instruct":
            print("--- WARNING: Loading Llama 3.1 8B in Library Mode requires ~16GB+ RAM ---")
            
        try:
            _tokenizer_cache[full_model_path] = AutoTokenizer.from_pretrained(
                full_model_path, 
                token=HF_TOKEN
            )
            _model_cache[full_model_path] = AutoModelForCausalLM.from_pretrained(
                full_model_path, 
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                token=HF_TOKEN
            )
            print(f"--- Model {full_model_path} loaded successfully ---")
        except GatedRepoError:
            print(f"--- Error: Model {full_model_path} is gated. Please provide a valid HF_TOKEN in .env and ensure you have access. ---")
            raise Exception("Gated Model Access Required")
        except RepositoryNotFoundError:
            print(f"--- Error: Model {full_model_path} not found on Hugging Face hub. ---")
            raise Exception("Model Not Found")
        except Exception as e:
            print(f"--- Error loading model {full_model_path}: {e} ---")
            raise e
            
    return _tokenizer_cache[full_model_path], _model_cache[full_model_path]

def run_library(prompt):
    try:
        import torch
        print(f"Mode: Library | Model: {MODEL_NAME} | Status: Loading/Checking...")
        tokenizer, model = get_library_model(MODEL_NAME)
        
        full_prompt = f"{SYSTEM_PROMPT}\n\nUser:\n{prompt}"
        start_time = time.time()
        mem_before = measure_memory()
        
        print("Mode: Library | Status: Generating...")
        inputs = tokenizer(full_prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
            
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                do_sample=True
            )
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        latency = time.time() - start_time
        mem_after = measure_memory()
        return output_text, latency, mem_after - mem_before
    except Exception as e:
        return f"Skipped: {e}", 0, 0

def run_api(prompt):
    try:
        api_key = CEREBRAS_API_KEY.strip() if CEREBRAS_API_KEY else None
        if not api_key:
            return "Skipped: no API key", 0, 0
            
        print(f"Mode: API | Model: {CEREBRAS_MODEL} | Status: Calling...")
        full_prompt = f"{SYSTEM_PROMPT}\n\nUser:\n{prompt}"
        start_time = time.time()
        mem_before = measure_memory()
        
        url = "https://api.cerebras.ai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": CEREBRAS_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": MAX_TOKENS,
            "temperature": TEMPERATURE
        }
        
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        
        if response.status_code == 404:
            url = "https://api.cerebras.ai/v1/generate"
            payload = {
                "model": CEREBRAS_MODEL,
                "prompt": full_prompt,
                "max_tokens": MAX_TOKENS,
                "temperature": TEMPERATURE
            }
            response = requests.post(url, headers=headers, json=payload, timeout=60)

        response.raise_for_status()
        
        json_data = response.json()
        output_text = ""
        if "choices" in json_data:
            output_text = json_data["choices"][0]["message"]["content"]
        elif "completion" in json_data:
            output_text = json_data["completion"]
            
        latency = time.time() - start_time
        mem_after = measure_memory()
        return output_text, latency, mem_after - mem_before
    except requests.Timeout:
        return "Skipped: API request timed out", 0, 0
    except Exception as e:
        return f"Skipped: {e}", 0, 0

def run_local(prompt):
    try:
        print(f"Mode: Local | Model: {MODEL_NAME} | Status: Calling Ollama...")
        full_prompt = f"{SYSTEM_PROMPT}\n\nUser:\n{prompt}"
        start_time = time.time()
        mem_before = measure_memory()
        result = subprocess.run(
            ["ollama", "run", MODEL_NAME, f"{full_prompt}"],
            capture_output=True, text=True, timeout=120
        )
        
        output = result.stdout.strip()
        if not output and result.stderr:
            output = f"Ollama Error: {result.stderr.strip()}"
            
        latency = time.time() - start_time
        mem_after = measure_memory()
        return output, latency, mem_after - mem_before
    except FileNotFoundError:
        return "Skipped: ollama not installed or in PATH", 0, 0
    except subprocess.TimeoutExpired:
        return "Skipped: local process timed out", 0, 0
    except Exception as e:
        return f"Skipped: {e}", 0, 0

if __name__ == "__main__":
    try:
        print("Chose Method :")
        print("1. Local LLM (Installing model in local device)")
        print("2. Import library")
        print("3. Using API / CLI")
        
        choice = input("\nUser input (Chose method) : ").strip()
        user_prompt = input("User Input (Insert Prompt) : ").strip()
        
        if not user_prompt:
            user_prompt = "Explain the Transformer architecture in 3 sentences."

        print(f"\nRunning benchmark for prompt:\n{user_prompt}\n{'='*50}\n")

        if choice == "1":
            output, latency, mem = run_local(user_prompt)
            print_metrics("local", latency, mem, output)
        elif choice == "2":
            output, latency, mem = run_library(user_prompt)
            print_metrics("library", latency, mem, output)
        elif choice == "3":
            output, latency, mem = run_api(user_prompt)
            print_metrics("api", latency, mem, output)
        else:
            print("Invalid choice. Running all methods...")
            for mode, func in [("api", run_api), ("local", run_local), ("library", run_library)]:
                output, latency, mem = func(user_prompt)
                print_metrics(mode, latency, mem, output)

    except KeyboardInterrupt:
        print("\nBenchmark cancelled by user.")
