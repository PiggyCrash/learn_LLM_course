import math
import numpy as np

vocab = ["The", "cat", "sat", "on", "the", "mat", "[CLS]"]
vocab_size = len(vocab)
embedding_dim = 4

np.random.seed(42)
embeddings = {word: np.random.rand(embedding_dim) for word in vocab}

sentence = ["[CLS]", "The", "cat", "sat", "on", "the", "mat"]
input_vectors = np.array([embeddings[w] for w in sentence])

def positional_encoding(seq_len, dim):
    pos_enc = np.zeros((seq_len, dim))
    for pos in range(seq_len):
        for i in range(0, dim, 2):
            pos_enc[pos, i] = math.sin(pos / (10000 ** (i / dim)))
            if i+1 < dim:
                pos_enc[pos, i+1] = math.cos(pos / (10000 ** (i / dim)))
    return pos_enc

pos_enc = positional_encoding(len(sentence), embedding_dim)
x = input_vectors + pos_enc

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=-1, keepdims=True)

def self_attention(x):
    Q = x
    K = x
    V = x
    
    scores = np.dot(Q, K.T) / math.sqrt(embedding_dim)
    weights = softmax(scores)
    
    out = np.dot(weights, V)
    return out

attn_output = self_attention(x)

def relu(x):
    return np.maximum(0, x)

def feed_forward(x):
    W1 = np.random.rand(embedding_dim, embedding_dim)
    W2 = np.random.rand(embedding_dim, embedding_dim)
    out = relu(np.dot(x, W1))
    out = np.dot(out, W2)
    return out

ffn_output = feed_forward(attn_output)

def layer_norm(x):
    mean = x.mean(axis=-1, keepdims=True)
    std = x.std(axis=-1, keepdims=True) + 1e-6
    return (x - mean) / std

residual_output = layer_norm(ffn_output + x)
cls_vector = residual_output[0]

W_cls = np.random.rand(embedding_dim, 2)
logits = np.dot(cls_vector, W_cls)
probs = softmax(logits)

print("Sentence:", " ".join(sentence))
print("Predicted probabilities [Positive, Negative]:", probs)
