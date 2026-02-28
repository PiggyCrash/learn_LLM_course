print("--- Initializing sentiment-analysis pipeline (may download files) ---")
from transformers import pipeline

classifier = pipeline("sentiment-analysis")

print("--- Running classification ---")
result = classifier("I've been waiting for a HuggingFace course my whole life.")
print(f"Result: {result}")