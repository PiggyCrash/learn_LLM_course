## Tech Progressive Overview
DL -> LSTM (Only remember backward/sequentially, ignoring whole data, in-reality : finding context can be anywhere/poor generalization) -> Transformer Model (Layer Becomes Block, Introducing Self Attention, creating relationship with whole data, mapped 1-by-1, based on score/attention-score/correlation-score, creating better context/generalization) -> LLM (Pre-trained Transformer with Billion parameter)

## Transformer Architecture : Encoder vs Decoder vs Hybrid
See : [transformer_architecture.py](src/transformer_architecture.py) | Still poor output because random embedding/FF + no training sequence

The Flow : Token Embedding -> Positional Encoding (Assign value for order) -> Self-Attention (Computes score based on relevation) -> Feed Forward Network (FFN) (Vector input is passed betwen NN) -> Residual Connections + LayerNorm (stabilizes gradients and preserves original info.) -> Encoder/Decoder blocks (Repeated until layers become block, deeper block = higher level of understanding) -> Output Head

Note : Things to be look on input/output
1. The diimension of the token, if different -> padding (batch_size, sequence_length, hidden_size)
2. The position of the token
3. The attention mask (tells which token to ignore)
4. The activation function
5. The length of token, each model has different MAX 

Note : This will be different with other purposes, different transformer architecture for audio, image, text, suumarization, translation, etc

## Difference of LLM implementation
1. Local LLM (Installing model in local device)
2. Import library 
3. Using API / CLI 

## Choosing the right model ? Different Usecase -> Diffferent Model
BERT - Text, Mask2Former - Image, ConvNeXT - Image (Probably discuss/research first which model is best for specific context)

## Training transformer Model : MLM vs CLM
MLM (Masked) -> Look all the other words in the sentence, before & after
CLM (Causal) -> Guess the next word, based on previous

## Inferece of LLMs : How token predict next word/phrase/sentence ?
Inference -> running the Transformer forward repeatedly to predict one token at a time using frozen learned knowledge. (Weight is not updated / differ from training process)

Token -> random scaler (try some combination) -> Neighbor Filter (Create penalty to prevent repetition/ Stop sequences to control generated length)-> Generated : Selected word 

## Auto Model : Architecture-model routing concept in pretrained model and fine-tuning
Given a model name and config, then route and construct the correct architecture class automatically

## What can be optimized on Inference Deployment ?
1. Memory Management Issue ? try change Normal Attention -> Flash Attention
2. Deployment on Local vs Production issue -> try to use TGI 
3. KV cache management issue ? try to use vLLM
4. Limited hardware ? try llama.cpp

## Flash Attention : Improving memory management for normal attention
Normal attention (instead of compute and store QKT + stored full matrix)-> Larger matrix = larger GPU usage -> Compute attention in chunks + accumulate results + discard intermediate large matrices + Never stored full matrix

## TGI : Server ready to deploy LLM on prod
Scaling from local to prod = problem, introducing framework LLM prod-ready, that handle -> inferences, streaming + latency, GPU management

Process Flow = Request -> Tokenizer -> Batch Queue -> GPU Forward Pas -> Sampling -> Output token -> Response

Pros : 
- 'Continious Batch' as merging incoming request instead of full batch, higher throughput, lower token cost
- Cache management, for token, through all user -> No memory break
- Tensor parallelism, heavy ? TGI handles split GPU
- Stream output, like chat GPT typing, whereas user can view how the token generated
- Quantization, reducing precision of model weight (just a bit) ->lesser memory

## vLLM : solve GPU memory fragmentation & KV cache & poor batch
Why PaggedAttention ? Introducting KVcache 

Why vLLM ? KV cache stored key value & vector -> lots of user (lots of token generated -> memory full + scaling issue)

Prev Flow -> Req A & Req B, each 1 memory block
Flow with vLLM -> Instead of allocating resource all upfront -> Split into pages -> Forward Pass -> Generate Next Token -> Append KV New page

## llama.cpp : Optimizing memory, quantization, CPU for low-end device
For accesibilty : Reducing size of overall spec in local device for accesibility

## Fine-tuning vs Hyper Parameter Tuning
Hyperparameter -> Adjust from sctratch the data & learning-curve & parameter & architecture (layer, nodes, function, etc)
Fine-tune -> Adjust pre-trained model (LLM & transformer) to your benefit, WO changing the whole infrastructure / stay identical

The core concept of changing : 
1. Architecture = stay identic, what changes -> the weight inside the metrix
2. Your data use case aren't becoming the ground truth (the data isn't stored) -> but your data adjust the loss function -> gradient -> weight 
3. Sequence/your data are input -> not going to be stored (different from DL)

Note : There's seems to be a different of fine-tune pretrained vs LLM, maybe want to check later, something like this : 

Pretrained Model Fine-Tuning Methods
1. Full fine-tune (large dataset + different from pretrained + GPU -> update all weight/ no new layer)
2. Feature extraction (freeze base) (small dataset, freeze initial layer + train last layer)
3. Partial layer fine-tune (freeze early layer, fine-tune deeper layer)
4. Adapters (Insert new layer)

Other notable methods : 
1. Trainer API, basically library for fine-tune instead of scratching all the code

Process flow of how they train data (WO pretrained model): 
Raw text + labels → tokenize → convert to input_ids/attention_mask/token_type_ids → batch & pad → numeric tensors ready for model → feed into model for training → compute loss → backprop → update weights

Process Flow of train data (pretrained model): (PS : Also has library called accelerate)
1. Data cleaning/pre-processing [clean raw text → remove noise, missing data]
2. Tokenize (+ AttentionMask) [convert text → model-readable IDs & mark padding]
3. Feature Engineering [optional, add extra info → improve model input]
4. Create DataLoader (Handles Batch, Shuffle, Padding) [efficient batch iteration, randomization, uniform length]
5. Import AutoModel [load pretrained architecture + weights]
6. Move Model to Device (CPU → GPU) [ensure model & batch on same device → GPU for speed, memory]
7. Pass Batch to Model (Forward → Loss → Backward → Optimizer → Learning Rate Scheduler → Update Weights) [compute predictions, calculate error, adjust weights to minimize loss]
8. Model Set to Training Mode (Random Dropout + BatchNorm/FF) [enable stochastic regularization → prevent overfit, allow proper batch normalization]
9. Outer Loop (Epoch) [iterate over dataset multiple times → better convergence]
10. Inner Loop (Batch) [process batch-by-batch → efficient memory usage]
11. Model Set to Evaluation Mode (Disable Dropout + Freeze BatchNorm) [disable stochastic behavior → deterministic inference]
12. Move Batch to GPU [same device → avoid crash]
13. Disable Gradient Computation [no backprop during evaluation → save memory & speed]
14. Logits Extracted [raw predictions → before softmax / probability]
15. Prediction [argmax / label selection → convert logits → class]
16. Evaluate Metric [measure performance → guide fine-tuning decisions]

# LLM Token 
Usage of Token = all previous messages + current input + output

Reduce the usage : 
1. Use structured input/output
2. Use summarized version
3. Set max token
4. File input are token expensive, so try to make it structured first
5. Context management (Sliding windows, compress history chat to summarized, RAG, structured state)
6. Early design LLM / Engineering side
7. Keep reasoning off for simple question

## Fine-tune LLM Model 
Method : 
1. Full-fine tune (Update 100% of model weights + Needs GPU + large dataset)
2. Partial FT (Freeze some layers/ only fine tune classification output layer / deeper layer)
3. Parameter-Efficient Fine-Tuning (PEFT) [No weight is updated here] -> Adapters, LoRA, QLoRA, Prefix Tune, Prompt Tune
4. Instruction based FT -> SFT, RLHF, DPO
5. RL -> DPO, PPO, RLHF
6. Prompting (which doesn't include to fine-tuning method)

Note : Method 1, 2, 3 is how difference the architectural model is changed | While 3 4 5 6 is telling how to optimized the change (can be either in changing weight or not)

## SFT : Instruction/Answer Dataset
Core Idea = take a pretrained LLM and train it on paired examples of input → output. (Model weights updated on instruction-response dataset)

Process Flow = AutoModel -> Dataset -> Tokenize -> Forward pass → compute cross-entropy loss → backprop → update weights

Use when = Permanent behaviour (updated weight) + large data + large compute resource

Library : ```from trl import SFTConfig, SFTTrainer```

Data example : 
```
[
  {
    "instruction": "Explain photosynthesis to a 12-year-old.",
    "response": "Photosynthesis is how plants make their own food using sunlight, water, and air. Think of it like plants cooking!"
  },
  {
    "instruction": "What is the capital of France?",
    "response": "The capital of France is Paris."
  },
  {
    "instruction": "Summarize: The water cycle involves evaporation, condensation, and precipitation.",
    "response": "The water cycle is how water moves around the Earth. It evaporates, forms clouds, and falls as rain."
  }
]
```
Different with Prompt-Tune = Basically same, the data the process flow, but with updated Weight (while PT is not)

# Prompting
Basically just prompt for LLM to learn ur context and generate output without changing any weight
How is differ with SFT and PT = Model sees instruction and generates output; no weights updated. except, we export the historical chat and import it into our model

## LoRA & QLoRA
LoRA core idea -> Instead of updating a huge weight matrix W, we freeze it and learn a low-rank update ΔW.

Mental idea : 
1. What if we the model has 4096 x 4096 hidden size, which approx = 17 M parameter
2. Insteaad of updated all, we chose rank (r) = 8, so ->  (4096×8)+(8×4096)=32768+32768 = 65536 parameter

Still big parameter -> try QLoRA -> Quantize the base moedl to 4-bit precision + Keep LoRA adapters in 16-bit + Train LoRA while base model stays quantized

