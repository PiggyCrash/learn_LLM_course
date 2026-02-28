from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "mistral-7b-chat"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
