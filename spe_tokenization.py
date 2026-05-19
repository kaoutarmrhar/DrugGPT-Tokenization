import logging
import sys

# --- DÉBUT DU CODE À AJOUTER POUR LE LOGGING ---

# Configuration du système de journalisation (logging)
logging.basicConfig(
    level=logging.INFO, # Enregistre les messages INFO et ERROR
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        # Enregistre la sortie dans ce fichier
        logging.FileHandler("crash_debug.log"), 
        # Affiche aussi la sortie dans la console (terminal)
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger()
logger.info("Démarrage du script Python.")

# --- FIN DU CODE À AJOUTER ---

# Le reste de vos imports (import pandas as pd, import torch, etc.) est ici
# ...
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import logging
import torch
import torch.nn as nn
from torch.nn import functional as F
import unidecode
import random
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
import time
import os
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen
import warnings
from collections import Counter, defaultdict
import re
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

class SPETokenizer:
    """
    SMILES Pair Encoding (SPE) Tokenizer

    True SPE implementation:
    1. Chemistry-aware initial tokenization (Br, Cl, [atoms], @@ stay together)
    2. Proper BPE: iteratively find most frequent pair and merge
    3. Each merge creates new tokens that can form new pairs
    """

    def __init__(self, num_merges=3000):
        self.num_merges = num_merges
        self.vocab = {}
        self.merges = []  # Ordered list of merge operations

        # SMILES-aware regex pattern
        self.smiles_pattern = re.compile(
            r'(\[[^\]]+\]|'      # Bracket atoms: [NH2], [O-], etc.
            r'Br|Cl|Si|Se|se|As|Te|te|Sb|'  # Two-letter atoms
            r'@@|@|'             # Stereochemistry
            r'%[0-9]{2}|'        # Ring numbers >9
            r'.)'                # Any single character
        )

    def _tokenize_smiles(self, text):
        """Chemistry-aware tokenization - KEY SPE FEATURE"""
        return self.smiles_pattern.findall(text)

    def _get_pair_stats(self, tokens):
        """Count frequency of adjacent token pairs"""
        pairs = defaultdict(int)
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])
            pairs[pair] += 1
        return pairs

    def _merge_pair(self, tokens, pair, new_token):
        """Merge all occurrences of a pair into new_token"""
        new_tokens = []
        i = 0
        while i < len(tokens):
            # Check if current position matches the pair
            if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                new_tokens.append(new_token)
                i += 2  # Skip both tokens in the pair
            else:
                new_tokens.append(tokens[i])
                i += 1
        return new_tokens

    def train(self, text, verbose=True):
        """
        Train SPE tokenizer using proper BPE algorithm.

        Args:
            text: String containing all molecules (with < and >)
            verbose: Print progress
        """
        if verbose:
            print("="*70)
            print("🔬 Training SMILES Pair Encoding (SPE) Tokenizer")
            print("="*70)

        start_time = time.time()

        # Step 1: Chemistry-aware initial tokenization
        if verbose:
            print("\n[1/3] Chemistry-aware tokenization...")
        tokens = self._tokenize_smiles(text)
        initial_token_count = len(tokens)

        if verbose:
            print(f"   Initial tokens: {len(tokens):,}")
            print(f"   Example: {tokens[:30]}")

        # Step 2: Build initial vocabulary
        if verbose:
            print("\n[2/3] Building initial vocabulary...")
        unique_tokens = sorted(set(tokens))
        self.vocab = {'<PAD>': 0, '<UNK>': 1}
        for token in unique_tokens:
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)

        initial_vocab_size = len(self.vocab)
        if verbose:
            print(f"   Initial vocab size: {initial_vocab_size:,}")

        # Step 3: PROPER BPE - iterative merging
        if verbose:
            print(f"\n[3/3] BPE merging (target: {self.num_merges:,} merges)...")

        for merge_idx in range(self.num_merges):
            # Count all pairs in current token sequence
            pair_stats = self._get_pair_stats(tokens)

            if not pair_stats:
                if verbose:
                    print(f"   ⚠️ No more pairs at merge {merge_idx}")
                break

            # Find most frequent pair
            best_pair = max(pair_stats.items(), key=lambda x: x[1])
            pair, freq = best_pair

            # Don't merge across molecule boundaries
            if '<' in pair or '>' in pair:
                del pair_stats[pair]
                if not pair_stats:
                    if verbose:
                        print(f"   ⚠️ Only boundary pairs left at merge {merge_idx}")
                    break
                best_pair = max(pair_stats.items(), key=lambda x: x[1])
                pair, freq = best_pair

            # Stop if frequency too low
            if freq < 2:
                if verbose:
                    print(f"   ⚠️ Max frequency < 2 at merge {merge_idx}")
                break

            # Create new merged token
            new_token = ''.join(pair)

            # Store merge operation (ORDER MATTERS for encoding!)
            self.merges.append((pair, new_token))

            # Add to vocabulary
            if new_token not in self.vocab:
                self.vocab[new_token] = len(self.vocab)

            # THIS IS THE KEY: Apply merge to create new token sequence
            # This allows NEW pairs to form in next iteration
            tokens = self._merge_pair(tokens, pair, new_token)

            # Progress
            if verbose and (merge_idx + 1) % 500 == 0:
                compression = initial_token_count / len(tokens)
                print(f"   Merge {merge_idx + 1:,}/{self.num_merges:,}: "
                      f"('{pair[0]}', '{pair[1]}') → '{new_token}' "
                      f"[freq: {freq:,}, tokens: {len(tokens):,}, compression: {compression:.2f}x]")

        training_time = time.time() - start_time
        final_compression = initial_token_count / len(tokens)

        if verbose:
            print("\n" + "="*70)
            print("✅ Training Complete!")
            print("="*70)
            print(f"   Initial vocab size: {initial_vocab_size:,}")
            print(f"   Final vocab size: {len(self.vocab):,}")
            print(f"   Vocab growth: +{len(self.vocab) - initial_vocab_size:,} tokens")
            print(f"   Merges performed: {len(self.merges):,}")
            print(f"   Initial tokens: {initial_token_count:,}")
            print(f"   Final tokens: {len(tokens):,}")
            print(f"   Compression ratio: {final_compression:.2f}x")
            print(f"   Training time: {training_time:.2f}s")

        return self

    def encode(self, text):
        """
        Encode SMILES string to token IDs.
        MUST apply merges in the same order as training!
        """
        # Step 1: Chemistry-aware tokenization
        tokens = self._tokenize_smiles(text)

        # Step 2: Apply merges IN ORDER (critical!)
        for pair, new_token in self.merges:
            tokens = self._merge_pair(tokens, pair, new_token)

        # Step 3: Convert to IDs
        ids = [self.vocab.get(token, self.vocab['<UNK>']) for token in tokens]

        return ids

    def decode(self, ids):
        """Decode token IDs back to SMILES string"""
        id_to_token = {idx: token for token, idx in self.vocab.items()}
        tokens = [id_to_token.get(idx, '<UNK>') for idx in ids]
        # Filter out special tokens
        tokens = [t for t in tokens if t not in ['<PAD>', '<UNK>']]
        return ''.join(tokens)

    def save(self, filepath):
        """Save tokenizer"""
        torch.save({
            'vocab': self.vocab,
            'merges': self.merges,  # Save ordered merges!
            'num_merges': self.num_merges
        }, filepath)
        print(f"💾 Tokenizer saved to {filepath}")

    def load(self, filepath):
        """Load tokenizer"""
        checkpoint = torch.load(filepath)
        self.vocab = checkpoint['vocab']
        self.merges = checkpoint['merges']
        self.num_merges = checkpoint['num_merges']
        print(f"📂 Tokenizer loaded from {filepath}")
        return self

if __name__ == "__main__":
    
    # Load dataset
    print("Loading dataset...")
    try:
        with open('data_for_generation_mol.txt', 'r') as f:
            text = f.read()
        print(f"✅ Dataset loaded")
        print(f"   Characters: {len(text):,}")
        num_molecules = text.count('<')
        print(f"   Molecules: {num_molecules:,}")

        # Recommended merges
        if num_molecules > 1_000_000:
            recommended = 4000
        elif num_molecules > 500_000:
            recommended = 3000
        else:
            recommended = 2000

        print(f"\n💡 Recommended merges: {recommended:,}")

    except FileNotFoundError:
        print("⚠️ File not found, using sample data...")
        text = "<CCO><CC(=O)O><CCBr><c1ccccc1><CC[NH2]>" * 100
        recommended = 1000
try:
    logger.info("Démarrage de la boucle d'entraînement.")
# Train SPE
    print(f"\nTraining SPE with {recommended:,} merges...")
    spe = SPETokenizer(num_merges=1500)
    spe.train(text, verbose=True)
    logger.info("Fin de l'entraînement. Sauvegarde du modèle...")
# Test
    print("\n" + "="*70)
    print("🔍 Testing Encoding/Decoding")
    print("="*70)

    test_smiles = text.split('>')[0] + '>'
    if '<' in test_smiles:
        print(f"\nOriginal: {test_smiles}")
        encoded = spe.encode(test_smiles)
        decoded = spe.decode(encoded)
        print(f"Encoded:  {len(encoded)} tokens")
        print(f"Decoded:  {decoded}")
        print(f"Match:    {'✓' if decoded == test_smiles else '✗'}")

# Show learned merges
    print("\n" + "="*70)
    print("📊 Sample Learned Merges")
    print("="*70)
    print("   First 10 merges:")
    for i, (pair, merged) in enumerate(spe.merges[:10], 1):
        print(f"      {i}. ('{pair[0]}', '{pair[1]}') → '{merged}'")

    if len(spe.merges) > 10:
        print(f"\n   Last 10 merges:")
        for i, (pair, merged) in enumerate(spe.merges[-10:], len(spe.merges)-9):
            print(f"      {i}. ('{pair[0]}', '{pair[1]}') → '{merged}'")

    logger.info("Démarrage de la boucle d'entraînement.")
   # Encode entire dataset
    print("\n" + "="*70)
    print("⚙️ Encoding Entire Dataset")
    print("="*70)
    all_ids = spe.encode(text)
    data = torch.tensor(all_ids, dtype=torch.long)
    vocab_size = len(spe.vocab)
    print(f"   Total tokens: {len(all_ids):,}")
    print(f"   Avg tokens/molecule: {len(all_ids)/text.count('<'):.2f}")
    print(f"   Tensor shape: {data.shape}")
    print(f"   Vocab size: {vocab_size}")
    logger.info("Fin de l'entraînement. Sauvegarde du modèle...")

except Exception as e:
    # Ceci n'est exécuté que si une erreur Python (IndexError, etc.) survient
    logger.error(f"UNE EXCEPTION PYTHON S'EST PRODUITE : {e}", exc_info=True)
# Save
    spe.save('spe_tokenizer.pth')

    print("\n✅ Done! SPE tokenizer ready for your generation model.")
    print("\nUsage:")
    print("   vocab_size = len(spe.vocab)")
    print("   encoded = spe.encode(smiles_string)")
    print("   decoded = spe.decode(token_ids)")

"""## Data Splitting and Batch Generation"""

# Improved hyperparameters
batch_size = 32        # Increased batch size for better gradients
block_size = 256       # Increased context length for better SMILES learning
learning_rate = 1e-4  # Better learning rate
n_head = 12             # Reduced for efficiency
n_embd = n_head * 32    # Reasonable embedding size (384)
ff_size_factor = 4     # Standard transformer ratio
n_layer = 12            # Sufficient depth
dropout = 0.15          # Regularization

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"🖥️ Using device: {device}")

# Split data into train/validation
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]
print(f"📊 Train data: {len(train_data):,} tokens")
print(f"📊 Val data: {len(val_data):,} tokens")

def get_batch(split):
    """Generate a batch of training data"""
    data_split = train_data if split == 'train' else val_data
    ix = torch.randint(len(data_split) - block_size, (batch_size,))
    x = torch.stack([data_split[i:i+block_size] for i in ix])
    y = torch.stack([data_split[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

# Test batch generation
xb, yb = get_batch('train')
print(f"\n🔢 Batch shapes:")
print(f"Input (x): {xb.shape}")
print(f"Target (y): {yb.shape}")

# Show example of training sequences
print(f"\n📝 Sample training sequences:")
for i in range(min(3, batch_size)):
    sample_x = spe.decode(xb[i][:20].cpu().tolist())
    sample_y = spe.decode(yb[i][:20].cpu().tolist())
    print(f"Sequence {i+1}:")
    print(f"  Input:  '{sample_x}'")
    print(f"  Target: '{sample_y}'")

# Model Architecture

class Head(nn.Module):
    """Single head of self-attention with improvements"""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
        self.head_size = head_size

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        # Scaled dot-product attention with proper scaling
        wei = q @ k.transpose(-2, -1) * (self.head_size ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    """Multi-head attention with residual connection"""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    """Feed-forward network with GELU activation"""

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, ff_size_factor * n_embd),
            nn.GELU(),
            nn.Linear(ff_size_factor * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """Transformer block with pre-layer normalization"""

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # Pre-layer normalization (better for training stability)
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class DrugGPT(nn.Module):
    """Improved DrugGPT model for SMILES generation"""

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights for better training"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens, eos_token, temperature=1.0, top_k=None):
        """Improved generation with temperature and top-k sampling"""
        self.eval()  # Set to evaluation mode

        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Crop context to block_size
                idx_cond = idx[:, -block_size:]

                # Get predictions
                logits, _ = self(idx_cond)
                logits = logits[:, -1, :] / temperature

                # Apply top-k filtering if specified
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')

                # Sample from the distribution
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)

                # Check for end-of-sequence token
                if idx_next.item() == eos_token:
                    break

                # Append to sequence
                idx = torch.cat((idx, idx_next), dim=1)

        self.train()  # Back to training mode
        return idx

# Create and initialize model
model = DrugGPT()
model = model.to(device)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"🔧 Model created successfully!")
print(f"📊 Total parameters: {total_params/1e6:.2f}M")
print(f"🏋️ Trainable parameters: {trainable_params/1e6:.2f}M")
print(f"💾 Model size: ~{total_params * 4 / 1e6:.1f}MB")

# Training Setup and Loss Estimation

# Training hyperparameters
max_iters = 10000
eval_interval = 200
eval_iters = 100
save_interval = 500
save_path = "spe/improved_drugGPT.pt"

# Optimizer with weight decay for regularization
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iters, eta_min=learning_rate*0.1)

@torch.no_grad()
def estimate_loss():
    """Estimate loss on train and validation sets"""
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

def validate_smiles(smiles_list):
    """Validate SMILES strings using RDKit"""
    valid_smiles = []
    for smi in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                # Additional validation: check if it can be converted back
                canonical_smi = Chem.MolToSmiles(mol)
                if canonical_smi:
                    valid_smiles.append(canonical_smi)
        except:
            continue
    return valid_smiles

def generate_smiles(model, n_samples=100, temperature=0.8, max_length=100):
    """Generate SMILES strings"""
    sos_token = spe.encode('<')[0]
    eos_token = spe.encode('>')[0]

    generated_smiles = []
    context = torch.tensor([[sos_token]], dtype=torch.long, device=device)

    for i in range(n_samples):
        try:
            # Generate sequence
            generated = model.generate(
                context,
                max_new_tokens=max_length,
                eos_token=eos_token,
                temperature=temperature,
                top_k=40  # Add top-k sampling for better diversity
            )

            # Decode and clean
            smi = spe.decode(generated[0].tolist())
            smi = smi.replace('<', '').replace('>', '').strip()

            if smi:  # Only add non-empty strings
                generated_smiles.append(smi)

        except Exception as e:
            print(f"⚠️ Generation error for sample {i}: {e}")
            continue

    return generated_smiles

print("✅ Training functions ready!")

# Training Loop

# Check for existing checkpoint
resume_training = os.path.exists(save_path)
start_iter = 0

if resume_training:
    print(f"🔄 Resuming from checkpoint: {save_path}")
    try:
        checkpoint = torch.load(save_path, map_location=device)
        if isinstance(checkpoint, dict):
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_iter = checkpoint.get('iteration', 0)
            print(f"📍 Resuming from iteration {start_iter}")
        else:
            # Legacy: whole model saved
            model = checkpoint
            model.to(device)
            print("📍 Loaded legacy checkpoint")
    except Exception as e:
        print(f"⚠️ Error loading checkpoint: {e}")
        print("🔄 Starting fresh training...")
        resume_training = False

# Training variables
losses = []
train_losses = []
val_losses = []
start_time = time.time()

print(f"🚀 Starting training for {max_iters} iterations...")
print(f"📊 Batch size: {batch_size}, Block size: {block_size}")
print(f"🧠 Model: {trainable_params/1e6:.1f}M parameters")


# Training loop
for iter_num in range(start_iter, max_iters):
    # Evaluate losses periodically
    if iter_num % eval_interval == 0 or iter_num == max_iters - 1:
        losses_dict = estimate_loss()
        train_loss = losses_dict['train']
        val_loss = losses_dict['val']

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        elapsed_time = time.time() - start_time
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Step {iter_num:4d} | "
              f"Train: {train_loss:.4f} | "
              f"Val: {val_loss:.4f} | "
              f"LR: {current_lr:.2e} | "
              f"Time: {elapsed_time:.1f}s")

    # Training step
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()

    # Gradient clipping for stability
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    optimizer.step()
    scheduler.step()

    losses.append(loss.item())

    # Save checkpoint
    if iter_num % save_interval == 0 or iter_num == max_iters - 1:
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'iteration': iter_num,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'config': {
                'vocab_size': vocab_size,
                'block_size': block_size,
                'n_embd': n_embd,
                'n_head': n_head,
                'n_layer': n_layer,
                'dropout': dropout
            }
        }
        torch.save(checkpoint, save_path)
        # torch.save(model.state_dict(), save_path)
        print(f"💾 Checkpoint saved at iteration {iter_num}")
   
total_time = time.time() - start_time
print(f"\n✅ Training completed!")
print(f"⏱️ Total training time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
print(f"🚀 Average time per iteration: {total_time/max_iters:.3f}s")

# Plot training curves
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(losses, alpha=0.7, label='Training Loss (per step)')
plt.title('Training Loss Over Time')
plt.xlabel('Training Step')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
eval_steps = range(0, len(train_losses) * eval_interval, eval_interval)
plt.plot(eval_steps, train_losses, 'o-', label='Train Loss', linewidth=2)
plt.plot(eval_steps, val_losses, 's-', label='Validation Loss', linewidth=2)
plt.title('Train vs Validation Loss')
plt.xlabel('Training Step')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"📊 Final Training Loss: {train_losses[-1]:.4f}")
print(f"📊 Final Validation Loss: {val_losses[-1]:.4f}")

# Quick generation test
print(f"\n🧪 Testing generation...")
test_samples = generate_smiles(model, n_samples=10, temperature=0.8)
print(f"Generated samples: {test_samples[:5]}")
logger.info("Fin de l'entraînement. Sauvegarde du modèle...")


# # Set path to utils_one.py
# utils_path = 'utils_first.py'

# # Add its directory to Python path
# import sys
# import os
# sys.path.append(os.path.dirname(utils_path))

# Import utility functions from the module utils_one
# from utils_first import
from utils import (
    n_chars,
    check_validity,
    check_novelty,
    strsmis2listsmis,
    list2txt,
    get_props,
)

# ==========================================
# FIXED BENCHMARKING
# ==========================================
import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
import random

# ==========================================
# FIRST FIX: PROPER NOVELTY CHECKING
# ==========================================

def load_training_smiles(file_path):
    """Load and parse training SMILES from the dataset"""
    try:
        with open(file_path, 'r') as f:
            text = f.read()

        # Extract SMILES from the training data
        # Assuming your data has format: <SMILES>
        training_smiles = []
        current_smi = ""
        in_smiles = False

        for char in text:
            if char == '<':
                in_smiles = True
                current_smi = ""
            elif char == '>':
                if in_smiles and current_smi:
                    training_smiles.append(current_smi)
                in_smiles = False
                current_smi = ""
            elif in_smiles:
                current_smi += char

        print(f"Loaded {len(training_smiles)} training SMILES")
        return list(set(training_smiles))  # Remove duplicates

    except Exception as e:
        print(f"Error loading training data: {e}")
        return []

# Load your training data
training_data_path = 'data_for_generation_mol.txt'
training_smiles = load_training_smiles(training_data_path)

# ==========================================
# IMPROVED GENERATION WITH BETTER SAMPLING
# ==========================================

def generate_diverse_smiles(model, n_samples=500, temperature_range=(0.7, 1.2), max_length=100):
    """Generate SMILES with varying temperatures for diversity"""
    model.eval()
    sos_token = spe.encode('<')[0]
    eos_token = spe.encode('>')[0]

    generated_smiles = []

    for i in range(n_samples):
        # Vary temperature for diversity
        temp = random.uniform(temperature_range[0], temperature_range[1])

        try:
            context = torch.tensor([[sos_token]], dtype=torch.long, device=device)

            generated = model.generate(
                context,
                max_new_tokens=max_length,
                eos_token=eos_token,
                temperature=temp,
                top_k=random.randint(30, 100)  # Vary top_k too
            )

            smi = spe.decode(generated[0].tolist())
            smi = smi.replace('<', '').replace('>', '').strip()

            if smi and len(smi) > 5:  # Basic length filter
                generated_smiles.append(smi)

        except Exception as e:
            continue

        if i % 50 == 0:
            print(f"Generated {i}/{n_samples} molecules...")

    model.train()
    return generated_smiles

# ==========================================
# CORRECTED BENCHMARK FUNCTIONS
# ==========================================

def check_validity_batch(smiles_list):
    """Check validity of SMILES batch"""
    valid_smiles = []
    for smi in smiles_list:
        if check_validity(smi):
            valid_smiles.append(smi)
    return valid_smiles

def check_novelty_correct(generated_smiles, training_smiles):
    """Check novelty against training set (CORRECTED)"""
    novel_smiles = []
    training_set = set(training_smiles)  # Convert to set for faster lookup

    for smi in generated_smiles:
        if smi not in training_set:
            novel_smiles.append(smi)

    return novel_smiles

def check_uniqueness_correct(smiles_list):
    """Remove duplicates"""
    return list(set(smiles_list))

def calculate_diversity_tanimoto(smiles_list, sample_size=1000):
    """Calculate internal diversity using Tanimoto similarity"""
    from rdkit.Chem import AllChem, DataStructs

    if len(smiles_list) < 2:
        return 0.0

    # Sample if too many molecules
    if len(smiles_list) > sample_size:
        smiles_sample = random.sample(smiles_list, sample_size)
    else:
        smiles_sample = smiles_list

    # Convert to molecules and fingerprints
    mols = []
    fps = []

    for smi in smiles_sample:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            mols.append(mol)
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=2048)
            fps.append(fp)

    if len(fps) < 2:
        return 0.0

    # Calculate pairwise similarities
    similarities = []
    for i in range(len(fps)):
        for j in range(i+1, len(fps)):
            sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
            similarities.append(sim)

    # Diversity = 1 - average similarity
    avg_similarity = np.mean(similarities)
    return 1 - avg_similarity

# ==========================================
# COMPREHENSIVE BENCHMARKING
# ==========================================

def run_comprehensive_benchmark(model, n_samples=1000):
    """Run complete benchmark with all metrics"""
    print("="*50)
    print("COMPREHENSIVE BENCHMARK")
    print("="*50)

    # Generate molecules
    print(f"Generating {n_samples} molecules...")
    generated_smiles = generate_diverse_smiles(model, n_samples=n_samples)
    print(f"Generated {len(generated_smiles)} raw molecules")

    # 1. Validity
    print("\n1. Checking validity...")
    valid_smiles = check_validity_batch(generated_smiles)
    validity = len(valid_smiles) / len(generated_smiles) if generated_smiles else 0
    print(f"Valid molecules: {len(valid_smiles)}/{len(generated_smiles)} ({validity:.2%})")

    if not valid_smiles:
        print("No valid molecules generated!")
        return

    # 2. Uniqueness
    print("\n2. Checking uniqueness...")
    unique_smiles = check_uniqueness_correct(valid_smiles)
    uniqueness = len(unique_smiles) / len(valid_smiles) if valid_smiles else 0
    print(f"Unique molecules: {len(unique_smiles)}/{len(valid_smiles)} ({uniqueness:.2%})")

    # 3. Novelty (CORRECTED)
    print("\n3. Checking novelty...")
    if training_smiles:
        novel_smiles = check_novelty_correct(unique_smiles, training_smiles)
        novelty = len(novel_smiles) / len(unique_smiles) if unique_smiles else 0
        print(f"Novel molecules: {len(novel_smiles)}/{len(unique_smiles)} ({novelty:.2%})")
    else:
        print("Warning: No training data loaded for novelty check")
        novel_smiles = unique_smiles
        novelty = 1.0

    # 4. Diversity
    print("\n4. Calculating diversity...")
    if len(novel_smiles) > 1:
        diversity = calculate_diversity_tanimoto(novel_smiles)
        print(f"Internal diversity: {diversity:.4f}")
    else:
        diversity = 0.0
        print("Not enough molecules for diversity calculation")

    # 5. Drug-likeness (Lipinski's Rule of Five)
    print("\n5. Checking drug-likeness...")
    drug_like_count = 0
    for smi in novel_smiles:
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                mw = Descriptors.MolWt(mol)
                logp = Descriptors.MolLogP(mol)
                hbd = Descriptors.NumHDonors(mol)
                hba = Descriptors.NumHAcceptors(mol)

                # Lipinski's Rule of Five
                if mw <= 500 and logp <= 5 and hbd <= 5 and hba <= 10:
                    drug_like_count += 1
        except:
            continue

    drug_likeness = drug_like_count / len(novel_smiles) if novel_smiles else 0
    print(f"Drug-like molecules: {drug_like_count}/{len(novel_smiles)} ({drug_likeness:.2%})")

    # Summary
    print("\n" + "="*50)
    print("BENCHMARK SUMMARY")
    print("="*50)
    print(f"Validity:      {validity:.2%}")
    print(f"Uniqueness:    {uniqueness:.2%}")
    print(f"Novelty:       {novelty:.2%}")
    print(f"Diversity:     {diversity:.4f}")
    print(f"Drug-likeness: {drug_likeness:.2%}")
    print("="*50)

    return {
        'validity': validity,
        'uniqueness': uniqueness,
        'novelty': novelty,
        'diversity': diversity,
        'drug_likeness': drug_likeness,
        'valid_smiles': valid_smiles,
        'unique_smiles': unique_smiles,
        'novel_smiles': novel_smiles
    }



# # Run the corrected benchmark
# print("Running corrected benchmark...")
# results = run_comprehensive_benchmark(model, n_samples=10)



"""### generate with untrained model"""

model = DrugGPT()
model = model.to(device)



generated_smiles = []
sos_token = spe.encode('<')[0]
eos_token = spe.encode('>')[0]
for i in range(10):
        # Vary temperature for diversity
        temp = random.uniform(0.65)

        try:
            context = torch.tensor([[sos_token]], dtype=torch.long, device=device)

            generated = model.generate(
                context,
                max_new_tokens=100,
                eos_token=eos_token,
                temperature=temp,
                top_k=random.randint(30, 100)  # Vary top_k too
            )

            smi = spe.decode(generated[0].tolist())
            smi = smi.replace('<', '').replace('>', '').strip()

            if smi and len(smi) > 5:  # Basic length filter
                generated_smiles.append(smi)

        except Exception as e:
            continue

        # if i % 10 == 0:
        print(f"Generated {i}/{10} molecules...")

generated_smiles



"""------------------------------------------------------------------

### Load the trained model and generate
"""

def load_model(checkpoint_path, device=device):
    # Initialize model without hyperparameters, relying on global definitions
    model = DrugGPT()

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint.state_dict()

    # Load state dict with strict=False to ignore unexpected keys
    model.load_state_dict(state_dict, strict=False)

    model = model.to(device)
    model.eval()
    return model





model = load_model("spe/improved_drugGPT.pt")

# model





generated_smiles = []
sos_token = spe.encode('<')[0]
eos_token = spe.encode('>')[0]
for i in range(10):
        # Vary temperature for diversity
        temp = random.uniform(0.7, 1.2)

        try:
            context = torch.tensor([[sos_token]], dtype=torch.long, device=device)

            generated = model.generate(
                context,
                max_new_tokens=100,
                eos_token=eos_token,
                temperature=temp,
                top_k=random.randint(30, 100)  # Vary top_k too
            )

            smi = spe.decode(generated[0].tolist())
            smi = smi.replace('<', '').replace('>', '').strip()

            if smi and len(smi) > 5:  # Basic length filter
                generated_smiles.append(smi)

        except Exception as e:
            continue

        # if i % 10 == 0:
        print(f"Generated {i}/{10} molecules...")

generated_smiles





## check validity :

vals = []

for s in generated_smiles:
    v = check_validity(s)
    if v:
        vals.append(s)

print((len(vals) / len(generated_smiles) ) *100, "%")







print("\n1. Checking validity...")
valid_smiles = check_validity_batch(generated_smiles)
validity = len(valid_smiles) / len(generated_smiles) if generated_smiles else 0
print(f"Valid molecules: {len(valid_smiles)}/{len(generated_smiles)} ({validity:.2%})")

print("\n2. Checking uniqueness...")
unique_smiles = check_uniqueness_correct(valid_smiles)
uniqueness = len(unique_smiles) / len(valid_smiles) if valid_smiles else 0
print(f"Unique molecules: {len(unique_smiles)}/{len(valid_smiles)} ({uniqueness:.2%})")

# 3. Novelty (CORRECTED)
print("\n3. Checking novelty...")
if training_smiles:
        novel_smiles = check_novelty_correct(unique_smiles, training_smiles)
        novelty = len(novel_smiles) / len(unique_smiles) if unique_smiles else 0
        print(f"Novel molecules: {len(novel_smiles)}/{len(unique_smiles)} ({novelty:.2%})")
else:
        print("Warning: No training data loaded for novelty check")
        novel_smiles = unique_smiles
        novelty = 1.0

# 4. Diversity
print("\n4. Calculating diversity...")
if len(novel_smiles) > 1:
        diversity = calculate_diversity_tanimoto(novel_smiles)
        print(f"Internal diversity: {diversity:.4f}")
else:
        diversity = 0.0
        print("Not enough molecules for diversity calculation")

# 5. Drug-likeness (Lipinski's Rule of Five)
print("\n5. Checking drug-likeness...")
drug_like_count = 0
for smi in novel_smiles:
     try:
        mol = Chem.MolFromSmiles(smi)
        if mol:
           mw = Descriptors.MolWt(mol)
           logp = Descriptors.MolLogP(mol)
           hbd = Descriptors.NumHDonors(mol)
           hba = Descriptors.NumHAcceptors(mol)

           # Lipinski's Rule of Five
           if mw <= 500 and logp <= 5 and hbd <= 5 and hba <= 10:
              drug_like_count += 1
     except:
            continue

drug_likeness = drug_like_count / len(novel_smiles) if novel_smiles else 0
print(f"Drug-like molecules: {drug_like_count}/{len(novel_smiles)} ({drug_likeness:.2%})")

# Summary
print("\n" + "="*50)
print("BENCHMARK SUMMARY")
print("="*50)
print(f"Validity:      {validity:.2%}")
print(f"Uniqueness:    {uniqueness:.2%}")
print(f"Novelty:       {novelty:.2%}")
print(f"Diversity:     {diversity:.4f}")
print(f"Drug-likeness: {drug_likeness:.2%}")
print("="*50)





