import os
import math

from dataclasses import dataclass
import torch
import torch.nn as nn 
from torch.nn import functional as F 

device = "cpu" # declare as string
if torch.cuda.is_available():
    device = "cuda" #Nvidia GPU
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps" # backend for apple silicon, which has a fairly capable GPU 
print(f"using device: {device}")

class CausalSelfAttention(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # mask (not bias), used OpenAI/huggingface naming through
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)
                                                ).view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionalit (n_embd)
        # calculate query, key, values for all heads in batch and move head forard to be the batch
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT2 (124Mill), n_head= 12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim = 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        
        # attention, materializes the large (T, T) matrix for all the queries and keys
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v # (B, nh, T, T) * (B, nh, T, hs) --> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) #re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc   = nn.Linear(config.n_embd, 4 * config.n_embd)
        # Why use Gelu instead of Relu? Read aobut it here: https://arxiv.org/abs/1606.08415
        # tl;dr: it's smoother and has some benefits making it better
        self.gelu   = nn.GELU(approximate='tanh') # smoother Relu using approx https://pytorch.org/docs/stable/generated/torch.nn.GELU.html
        # Gelu always creates a change and approximation making it work more effectively and adapt better than relu.
        # This is empirically demonstrated in numerous scenarios.
        # Why use the apprximate Gelu function instead of exact? https://github.com/pytorch/pytorch/issues/39853
        # tl;dr: erf function was pretty slow in tensorflow awhile back, just legacy reason
        # So why __still__ use it? Because we're being historically true to form.
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
    # forward pass for what this block actually computes
    # Clean residual pathway is very desirable from a pathway optimization perspective
    def forward(self, x):
        # remember: attention is an aggregation function, a pooling function. I call it a weighted sum function.
        # Another way to describe this: the attention is the reduce (as in map reduce)
        x = x + self.attn(self.ln_1(x)) # x first goes through layer normalization, then the attention
        # mlp is happening each time to each token
        # Another way to describe this: mlp is the map itself that you're reducing
        x = x + self.mlp(self.ln_2(x)) # then x goes back out through the next layer 
        # normalization into the multila perceptron (fnn - feed forward network)
        return x
    
@dataclass
class GPTConfig:
    block_size: int = 1024 # max sequence length
    # note: "endoftext" is a special delimiter token
    vocab_size: int = 50257 # num of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    
    n_layer: int = 12 # num of layers
    n_head: int = 12 # num of heads
    n_embd: int = 768 # embedding dimension
    
class GPT(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # setup allows you to index into modules using string like a dictionary
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd), 
            # embeddings are glorified wrappers around the tensor allowing you to 
            # index into the tensors rows
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            # additional final layer norm in here
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        # Final classifier, the language model head, that projects from 768 
        # to vocab size 50257
        # pe = positional encodings
        # dot h is all the blocks
        # lm_head is this linear part within figure 1
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False) # Why doesn't GPT2 use a bias? Explain.
    
    def forward(self, idx):
        # idx is of shape (B, T)
        B, T = idx.size() 
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        #forward the token and position embeddings
        pos = torch.arange(0, T, dtype = torch.long, device = idx.device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        return logits
        
        
        
        
    # reminder: class method is the same as a constructor
    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel 
        print("loading weights from pretrained gpt: %s" % model_type)
        
        # n_layer, n_head, and n_embd are determined from model_type
        config_args = {
            'gpt2': dict(n_layer = 12, n_head = 12, n_embd = 768), # 124M params
            'gpt2-medium': dict(n_layer = 24, n_head = 16, n_embd = 1024), # 350M params
            'gpt2-large': dict(n_layer = 36, n_head = 20, n_embd = 1280), # 774M params
            'gpt2-xl': dict(n_layer = 48, n_head = 25, n_embd = 1600) # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] #discard this mask / buffer
        
        # initialize a huggingface/transformer model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()
        
        #copy while ensuring all of the params are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same as above line
        # manually hardcoded the weights that needs to be transposed, a cludge solution but whatever
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # openai checkpoints use a "Conv1D module, but we only want to use a vanille one"
        # This means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treastment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # copy over the other params
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        return model
num_return_sequences = 5
max_length = 30

import tiktoken # good visual representation of this tokenization https://tiktokenizer.vercel.app/
enc = tiktoken.get_encoding('gpt2')
with open('input.txt', 'r') as f:
    text = f.read()
text = text[:1000]
tokens = enc.encode(text)
B, T = 4, 32 # note for non-pythoners, this assigns both things in order B = 4, T = 32
buf = torch.tensor(tokens[:B*T + 1]).to(device) # added one as first tensor
x = buf[:-1].view(B, T)
y = buf[1:].view(B, T)

# get logits
# model = GPT.from_pretrained('gpt2')
# below is the random model initialization
model = GPT(GPTConfig()) # uses 124M parameter model by default
model.to(device) # moving model to CUDA, better for parallel processing
logits = model(x)
# print("didn't crash, woohoo!")

tokens = enc.encode("Hello, I'm a language model,")
# added .to(device) to fix issue of mismatch CPU and GPU running
tokens = torch.tensor(tokens, dtype = torch.long).to(device) # (8,) 
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1) # (5, 8)
x = tokens.to('cuda')

print(logits.shape)
import sys; sys.exit(0)

# prefix tokens
model.eval()
num_return_sequences = 5

# generate! right now x is (B, T) where B = 5, T = 8
# set the seed to 42
torch.manual_seed(42)
torch.cuda.manual_seed(42)
while x.size(1) < max_length:
    # fwd the model to get the logits
    with torch.no_grad(): # not calling .backward(), no need to cache this stuff (space saver and possibly time)
        logits = model(x) # (B, T, vocab_size)
        # take the logits at the last position
        logits = logits[:, -1, :] # (B, vocab_size)
        #get the probabilities
        probs = F.softmax(logits, dim = -1)
        # do top-k sampling of 50 (huggingface pipeline default)
        # topk_probs here become (5, 50), topk_indices is (5, 50), only keep the top 50 probabilities
        # therefore, we are sampling only the most likely tokens (top 50). Never the unlikely tokens.
            # Think of this as a bumper to keep the model "on the rails" of likely tokens
        # More info: https://huggingface.co/docs/transformers/v4.48.0/en/internal/generation_utils#transformers.TopKLogitsWarper
        topk_probs, topk_indices = torch.topk(probs, 50, dim = -1)
        # select a token from the top-k probabilities
        ix = torch.multinomial(topk_probs, 1) # (B, 1)
        # gather the corresponding indices
        xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
        # append to the sequence
        x = torch.cat((x, xcol), dim = 1)

# print the generated text
for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(">", decoded)