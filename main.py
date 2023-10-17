# ## !wget https://raw.githubusercontent.com/karpathy/char-nn/master/data/tinyshakespeare/input.txt

# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# import time
# import math

# torch.manual_seed(372)

# ## Hyperparameters
# BATCH_SIZE = 32         ## Number of sequence that will process in parallel
# BLOCK_SIZE = 256        ## Maximum context Length for prediction
# MAX_ITER = 10000
# EVAL_INTERVAL = 500
# LEARNING_RATE = 5e-3
# EVAL_ITERS = 100
# N_EMBED = 564
# N_HEAD = 8
# N_LAYER = 8
# dropout = 0.2
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# #____________________________________________________________________________________________________________


# with open('Dataset/input.txt', 'r') as f:
#     text = f.read()

# chars = sorted(list(set(text)))
# vocab_size = len(chars)

# # Tokenize input text: CHaracterize level (subject to change for better results)
# # create a mapping for character to integer, and create a encpder and a decoder


# stoi = {ch: i for i, ch, in enumerate(chars)}
# itos = {i: ch for i, ch, in enumerate(chars)}

# encode = lambda s: [stoi[ch] for ch in s]
# # encoder: Take a string input, output a list of integer
# decode = lambda l: ''.join([itos[ch] for ch in l] )
# # decoder: Takes a list of integer as input and outputs a string


# # Split the text into training and validation set
# data = torch.tensor(encode(text), dtype=torch.long)
# n = int(0.9 * len(data))

# train_data = data[:n]
# val_data = data[n:]

# # Load the data
# def get_batch(split):
#     # Generate a small batch of data of inputs X and targets Y
#     data = train_data if split == 'train' else val_data
#     ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
#     x = torch.stack([data[i:i+BLOCK_SIZE] for i in ix])
#     y = torch.stack([data[i+1:i+BLOCK_SIZE+1] for i in ix])

#     return x,y


# @torch.no_grad()
# def estimate_loss():
#     out = {}
#     model.eval()
#     for split in ['train', 'val']:
#         losses = torch.zeros(EVAL_ITERS)
#         for k in range(EVAL_ITERS):
#             X, Y = get_batch(split)
#             logits, loss =  model(X,Y)
#             losses[k] = loss.item()
        
#         out[split] = losses.mean()
#     model.train()
#     return out
        

# class Head(nn.Module):
#     ''' One head of self-attention'''
    
#     def __init__(self, head_size):
#         super().__init__()
#         ## Linear Projections
#         self.key = nn.Linear(N_EMBED, head_size, bias=False)
#         self.query = nn.Linear(N_EMBED, head_size, bias=False)
#         self.value = nn.Linear(N_EMBED, head_size, bias=False)
#         self.register_buffer('tril', torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, x):
#         B,T,C = x.shape
#         k = self.key(x)     ## (B,T,C)
#         q = self.query(x)   ## (B,T,C)

#         ## Compute attention scores ("affinities")
#         wei = q @ k.transpose(-2,-1) * C**-0.5  ## (B,T,C) @ (B,C,T) --> (B,T,T)
#         wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf')) ## (B,T,T)
#         wei = F.softmax(wei, dim=-1) ## (B,T,T)
#         wei = self.dropout(wei)

#         #Perform weighted sum of values
#         val = self.value(x) ## (B,T,C)
#         out = wei @ val ## (B,T,T) @ (B,T,C) --> (B,T,C)
#         return out
    

# class MultiHeadAttention(nn.Module):
#     ''' Multiple head of self-attention, process in parallel '''

#     def __init__(self, num_head, head_size):
#         super().__init__()
#         self.heads = nn.ModuleList([Head(head_size=head_size) for _ in range(num_head)])
#         ## Residual Projection
#         self.projection = nn.Linear(N_EMBED, N_EMBED)
#         self.dropout = nn.Dropout(dropout)
        
#     def forward(self, x):
#         out = torch.cat([h(x) for h in self.heads], dim=-1)
#         out = self.dropout(self.projection(out))
#         return out


# class FeedForward(nn.Module):
#     ''' A simple Linear layer followed by a non-linearity'''

#     def __init__(self, n_embed):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(n_embed, 4*n_embed),
#             nn.ReLU(),
#             nn.Linear(4*n_embed, n_embed),
#             nn.Dropout(dropout)
#         )

#     def forward(self, x):
#             return self.net(x)


# class Block(nn.Module):
#     ''' Transformer Block: Communication followed by the computation '''

#     def __init__(self, n_embed, n_head):
#         ## n_embed: Embedding dimension
#         ## n_head: Number of head we'd like
#         super().__init__()
#         head_size = n_embed // n_head
#         self.sa = MultiHeadAttention(n_head, head_size)
#         self.ffwd = FeedForward(n_embed)
#         self.ln1 = nn.LayerNorm(n_embed)
#         self.ln2 = nn.LayerNorm(n_embed)

#     def forward(self, x):
#         x = x + self.sa(self.ln1(x))
#         x = x + self.ffwd(self.ln2(x))
#         return x

# class GPTLanguageModel(nn.Module):

#     def __init__(self):
#         super().__init__()

#         # Each token directly reads off the logits for the next token from a lookup table
#         self.token_embedding_table = nn.Embedding(vocab_size, N_EMBED)
#         self.position_embedding_table = nn.Embedding(BLOCK_SIZE, N_EMBED)
#         self.blocks = nn.Sequential(*[Block(N_EMBED, n_head=N_HEAD) for _ in range(N_LAYER)])
#         self.ln_f = nn.LayerNorm(N_EMBED) ## Final layer Norm
#         self.lm_head = nn.Linear(N_EMBED, vocab_size)

#         self.apply(self._init_weights)

#     def _init_weights(self, module):
#         if isinstance(module, nn.Linear):
#             torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
#             if module.bias is not None:
#                 torch.nn.init.zeros_(module.bias)

#         elif isinstance(module, nn.Embedding):
#             torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


#     def forward(self, idx, targets=None):
        
#         B,T = idx.shape

#         # idx and token are both (B,T) tensor of integers
#         token_embedding = self.token_embedding_table(idx)  ##(B,T,C)    
#         position_embedding = self.position_embedding_table(torch.arange(T, device=device)) ##(B,T,C)
#         x = token_embedding + position_embedding ##(B,T,C)
#         x = self.blocks(x)  ##(B,T,C)
#         x = self.ln_f(x)  ##(B,T,C)

#         logits = self.lm_head(x) ##(B,T,vocab_size)

#         ## Loss Function
#         '''
#             As cross entropy function require the paramenter in multi-dimensional input
#             Channels is require to be 2nd Parapeter .
#             Therefore we need to reshape the logits and target variable accordingly
#         '''
#         if targets == None:
#             loss = None
#         else:
#             B, T, C = logits.shape
#             logits = logits.view(B*T, C)
#             targets = targets.view(B*T)

#             loss = F.cross_entropy(logits, targets)

#         return logits, loss

#     def generate(self, idx, max_new_tokens):
        
#         # idx is (B,T) array of indices in the current context
#         for _ in range(max_new_tokens):
            
#             # Crop idx to the last block_size tokens
#             idx_cond = idx[:, -BLOCK_SIZE:]
#             # Get the predictions
#             logits, loss = self(idx_cond)

#             # Focus only on last step --> logits becomes (B,C)
#             logits = logits[:,-1,:]
#             probs = F.softmax(logits, dim=-1) # Apply softmax to get probabilites -- (B,C)

#             # Sample from the distribution
#             idx_next = torch.multinomial(probs, num_samples=1) ## (B,1)

#             # Append sampled index to the running sequence
#             idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)

#         return idx
    
# model = GPTLanguageModel()
# model = model.to(device)

# optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# for iter in range(MAX_ITER):

#     ## Evaluate the loss on train and val set, every once in a while
#     if iter % EVAL_INTERVAL == 0 or iter == MAX_ITER-1:
#         losses = estimate_loss()
#         print(f"step{iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

#     ## Sample a batch of data
#     xb, yb = get_batch('train')


#     ##  Evaluate the loss
#     logits, loss = model(xb, yb)
#     optimizer.zero_grad(set_to_none=True)
#     loss.backward()
#     optimizer.step()

# ## Generate from the model
# context = torch.zeros((1,1), dtype=torch.long, device=device)
# print(decode(model.generate(context, max_new_tokens=1000)[0].tolist()))

import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
# ------------

torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('Dataset/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T,C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

        # better init, not covered in the original GPT video, but important, will cover in followup video
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = GPTLanguageModel()
m = model.to(device)
# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))