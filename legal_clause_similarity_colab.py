"""
Legal Clause Similarity Detection System - Google Colab Version
================================================================
This script implements multiple baseline NLP architectures for identifying semantic 
similarity between legal clauses without using pretrained transformers.

GOOGLE COLAB COMPATIBLE - Reads data from Google Drive

Architectures Implemented:
1. BiLSTM-based Siamese Network
2. Attention-based Encoder Network

Author: Deep Learning Assignment 2

USAGE IN COLAB:
1. Upload the 'archive' folder to your Google Drive
2. Run this script - it will automatically mount Google Drive
3. Update DRIVE_ARCHIVE_PATH if your archive folder is in a different location
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Deep Learning imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

# NLP imports
import re
import string
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, roc_curve, auc, confusion_matrix,
    classification_report
)

# ============================================================================
# GOOGLE COLAB SETUP
# ============================================================================

def setup_colab():
    """Setup Google Colab environment and mount Google Drive."""
    print("="*60)
    print("GOOGLE COLAB SETUP")
    print("="*60)
    
    try:
        # Check if running in Colab
        import google.colab
        IN_COLAB = True
        print("✓ Running in Google Colab")
        
        # Mount Google Drive
        from google.colab import drive
        print("\nMounting Google Drive...")
        drive.mount('/content/drive', force_remount=False)
        print("✓ Google Drive mounted successfully")
        
        return True, '/content/drive/MyDrive'
        
    except ImportError:
        print("⚠ Not running in Google Colab")
        print("Using local file system")
        return False, '.'

# Check Colab and mount drive
IN_COLAB, DRIVE_ROOT = setup_colab()

# ============================================================================
# CONFIGURATION - MODIFY THIS PATH IF NEEDED
# ============================================================================

# Default path - modify this if your archive folder is in a different location
# Example paths:
# - '/content/drive/MyDrive/archive'                    (archive in root of Drive)
# - '/content/drive/MyDrive/DL/archive'                 (archive in DL folder)
# - '/content/drive/MyDrive/FAST/DL/archive'           (archive in FAST/DL folder)

if IN_COLAB:
    # MODIFY THIS PATH to match your Google Drive structure
    DRIVE_ARCHIVE_PATH = os.path.join(DRIVE_ROOT, 'archive')
    
    print(f"\n{'='*60}")
    print(f"Looking for archive folder at:")
    print(f"  {DRIVE_ARCHIVE_PATH}")
    print(f"{'='*60}")
    
    # Check if the path exists
    if not os.path.exists(DRIVE_ARCHIVE_PATH):
        print("\n⚠ WARNING: Archive folder not found at the specified path!")
        print("\nPlease update DRIVE_ARCHIVE_PATH in the script to match your folder location.")
        print("\nYour Google Drive structure:")
        
        # Show what's in the Drive root
        try:
            items = os.listdir(DRIVE_ROOT)
            print(f"\nContents of {DRIVE_ROOT}:")
            for item in items[:20]:  # Show first 20 items
                item_path = os.path.join(DRIVE_ROOT, item)
                if os.path.isdir(item_path):
                    print(f"  📁 {item}/")
                else:
                    print(f"  📄 {item}")
            if len(items) > 20:
                print(f"  ... and {len(items) - 20} more items")
        except:
            pass
        
        print("\nExample paths you might need:")
        print("  DRIVE_ARCHIVE_PATH = os.path.join(DRIVE_ROOT, 'archive')")
        print("  DRIVE_ARCHIVE_PATH = os.path.join(DRIVE_ROOT, 'DL', 'archive')")
        print("  DRIVE_ARCHIVE_PATH = os.path.join(DRIVE_ROOT, 'FAST', 'DL', 'archive')")
        
        # Try to find archive folder
        print("\nSearching for 'archive' folder...")
        for root, dirs, files in os.walk(DRIVE_ROOT):
            if 'archive' in dirs:
                found_path = os.path.join(root, 'archive')
                # Count CSV files
                csv_count = len(list(Path(found_path).glob('*.csv')))
                if csv_count > 0:
                    print(f"  ✓ Found: {found_path} ({csv_count} CSV files)")
        
        raise FileNotFoundError(
            f"Archive folder not found at {DRIVE_ARCHIVE_PATH}. "
            "Please update DRIVE_ARCHIVE_PATH variable in the script."
        )
    else:
        # Count CSV files
        csv_files = list(Path(DRIVE_ARCHIVE_PATH).glob('*.csv'))
        print(f"✓ Archive folder found with {len(csv_files)} CSV files")
        
    DATA_DIR = DRIVE_ARCHIVE_PATH
else:
    # Local mode (not in Colab)
    DATA_DIR = 'archive'

# Set random seeds for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nUsing device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")


# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

class LegalClauseDataLoader:
    """Loads and processes legal clause data from CSV files."""
    
    def __init__(self, data_dir=DATA_DIR):
        self.data_dir = data_dir
        self.clauses = []
        self.clause_types = []
        
    def load_data(self, max_files=None):
        """Load all CSV files from the archive directory."""
        print(f"\n{'='*60}")
        print("LOADING LEGAL CLAUSE DATASET")
        print(f"{'='*60}")
        print(f"Data directory: {self.data_dir}")
        
        csv_files = list(Path(self.data_dir).glob('*.csv'))
        if max_files:
            csv_files = csv_files[:max_files]
        
        print(f"Found {len(csv_files)} CSV files")
        
        all_clauses = []
        all_types = []
        
        # Progress tracking
        from tqdm import tqdm
        
        for csv_file in tqdm(csv_files, desc="Loading CSV files"):
            try:
                df = pd.read_csv(csv_file, encoding='utf-8')
                if 'clause_text' in df.columns and 'clause_type' in df.columns:
                    clauses = df['clause_text'].dropna().tolist()
                    types = df['clause_type'].dropna().tolist()
                    
                    # Ensure equal length
                    min_len = min(len(clauses), len(types))
                    all_clauses.extend(clauses[:min_len])
                    all_types.extend(types[:min_len])
            except Exception as e:
                print(f"Error loading {csv_file}: {e}")
                continue
        
        self.clauses = all_clauses
        self.clause_types = all_types
        
        print(f"\n✓ Loaded {len(self.clauses)} clauses")
        print(f"✓ Number of unique clause types: {len(set(self.clause_types))}")
        
        return self.clauses, self.clause_types


class TextPreprocessor:
    """Handles text cleaning and preprocessing for legal clauses."""
    
    @staticmethod
    def clean_text(text):
        """Clean and normalize legal clause text."""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep sentence structure
        text = re.sub(r'[^\w\s\.\,\;\:\(\)\-]', '', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    @staticmethod
    def tokenize(text):
        """Simple word tokenization."""
        return text.split()


class Vocabulary:
    """Builds and manages vocabulary for text data."""
    
    def __init__(self, max_vocab_size=50000, min_freq=2):
        self.max_vocab_size = max_vocab_size
        self.min_freq = min_freq
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx2word = {0: '<PAD>', 1: '<UNK>'}
        self.word_freq = Counter()
        
    def build_vocab(self, texts):
        """Build vocabulary from list of texts."""
        print("\nBuilding vocabulary...")
        
        preprocessor = TextPreprocessor()
        
        # Count word frequencies
        for text in texts:
            cleaned = preprocessor.clean_text(text)
            tokens = preprocessor.tokenize(cleaned)
            self.word_freq.update(tokens)
        
        # Add words to vocabulary based on frequency
        vocab_words = [word for word, freq in self.word_freq.most_common(self.max_vocab_size) 
                       if freq >= self.min_freq]
        
        for idx, word in enumerate(vocab_words, start=2):
            self.word2idx[word] = idx
            self.idx2word[idx] = word
        
        print(f"✓ Vocabulary size: {len(self.word2idx)}")
        print(f"  Most common words: {[w for w, _ in self.word_freq.most_common(10)]}")
        
        return self
    
    def encode(self, text):
        """Convert text to list of indices."""
        preprocessor = TextPreprocessor()
        cleaned = preprocessor.clean_text(text)
        tokens = preprocessor.tokenize(cleaned)
        return [self.word2idx.get(token, self.word2idx['<UNK>']) for token in tokens]
    
    def __len__(self):
        return len(self.word2idx)


class ClausePairGenerator:
    """Generates positive and negative clause pairs for similarity learning."""
    
    def __init__(self, clauses, clause_types):
        self.clauses = clauses
        self.clause_types = clause_types
        self.type_to_clauses = self._group_by_type()
        
    def _group_by_type(self):
        """Group clauses by their type."""
        type_dict = {}
        for clause, ctype in zip(self.clauses, self.clause_types):
            if ctype not in type_dict:
                type_dict[ctype] = []
            type_dict[ctype].append(clause)
        return type_dict
    
    def generate_pairs(self, num_pairs=10000):
        """
        Generate clause pairs with labels.
        Label 1: Similar (same clause type)
        Label 0: Dissimilar (different clause type)
        """
        print(f"\n{'='*60}")
        print("GENERATING CLAUSE PAIRS")
        print(f"{'='*60}")
        
        pairs = []
        labels = []
        
        # Generate positive pairs (similar clauses)
        num_positive = num_pairs // 2
        types_list = list(self.type_to_clauses.keys())
        
        for _ in range(num_positive):
            # Select a random clause type
            ctype = np.random.choice(types_list)
            clauses_of_type = self.type_to_clauses[ctype]
            
            # If there are at least 2 clauses of this type, create a pair
            if len(clauses_of_type) >= 2:
                clause1, clause2 = np.random.choice(clauses_of_type, size=2, replace=False)
                pairs.append((clause1, clause2))
                labels.append(1)  # Similar
        
        # Generate negative pairs (dissimilar clauses)
        num_negative = num_pairs - len(pairs)
        
        for _ in range(num_negative):
            # Select two different clause types
            type1, type2 = np.random.choice(types_list, size=2, replace=False)
            clause1 = np.random.choice(self.type_to_clauses[type1])
            clause2 = np.random.choice(self.type_to_clauses[type2])
            pairs.append((clause1, clause2))
            labels.append(0)  # Dissimilar
        
        print(f"✓ Generated {len(pairs)} clause pairs")
        print(f"  Positive pairs (similar): {sum(labels)}")
        print(f"  Negative pairs (dissimilar): {len(labels) - sum(labels)}")
        
        return pairs, labels


class ClausePairDataset(Dataset):
    """PyTorch Dataset for clause pairs."""
    
    def __init__(self, pairs, labels, vocabulary, max_length=100):
        self.pairs = pairs
        self.labels = labels
        self.vocabulary = vocabulary
        self.max_length = max_length
        
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        clause1, clause2 = self.pairs[idx]
        label = self.labels[idx]
        
        # Encode clauses
        encoded1 = self.vocabulary.encode(clause1)[:self.max_length]
        encoded2 = self.vocabulary.encode(clause2)[:self.max_length]
        
        # Convert to tensors
        tensor1 = torch.LongTensor(encoded1)
        tensor2 = torch.LongTensor(encoded2)
        label_tensor = torch.FloatTensor([label])
        
        return tensor1, tensor2, label_tensor


def collate_fn(batch):
    """Custom collate function to pad sequences in a batch."""
    clause1_list, clause2_list, labels = [], [], []
    
    for clause1, clause2, label in batch:
        clause1_list.append(clause1)
        clause2_list.append(clause2)
        labels.append(label)
    
    # Pad sequences
    clause1_padded = pad_sequence(clause1_list, batch_first=True, padding_value=0)
    clause2_padded = pad_sequence(clause2_list, batch_first=True, padding_value=0)
    labels = torch.stack(labels)
    
    # Get lengths for packed sequences
    lengths1 = torch.LongTensor([len(c) for c in clause1_list])
    lengths2 = torch.LongTensor([len(c) for c in clause2_list])
    
    return clause1_padded, clause2_padded, lengths1, lengths2, labels


# ============================================================================
# MODEL ARCHITECTURES
# ============================================================================

class BiLSTMSiameseNetwork(nn.Module):
    """
    Architecture 1: BiLSTM-based Siamese Network
    
    This architecture uses a shared BiLSTM encoder for both clauses,
    then computes similarity based on the encoded representations.
    """
    
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, num_layers=2, dropout=0.3):
        super(BiLSTMSiameseNetwork, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        
        # Similarity computation layers
        lstm_output_dim = hidden_dim * 2  # Bidirectional
        self.fc1 = nn.Linear(lstm_output_dim * 4, 256)  # Concatenated features
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)
        
        self.batch_norm1 = nn.BatchNorm1d(256)
        self.batch_norm2 = nn.BatchNorm1d(128)
        
    def encode_clause(self, clause, lengths):
        """Encode a single clause using BiLSTM."""
        # Embedding
        embedded = self.embedding(clause)
        embedded = self.dropout(embedded)
        
        # Pack padded sequence
        packed = pack_padded_sequence(
            embedded, 
            lengths.cpu(), 
            batch_first=True, 
            enforce_sorted=False
        )
        
        # BiLSTM
        packed_output, (hidden, cell) = self.lstm(packed)
        
        # Unpack
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        
        # Use last hidden state (concatenate forward and backward)
        # hidden shape: (num_layers * 2, batch, hidden_dim)
        forward_hidden = hidden[-2, :, :]
        backward_hidden = hidden[-1, :, :]
        encoded = torch.cat([forward_hidden, backward_hidden], dim=1)
        
        return encoded
    
    def forward(self, clause1, clause2, lengths1, lengths2):
        """Forward pass for clause pair."""
        # Encode both clauses
        encoded1 = self.encode_clause(clause1, lengths1)
        encoded2 = self.encode_clause(clause2, lengths2)
        
        # Compute similarity features
        # Concatenate: [encoded1, encoded2, abs(encoded1-encoded2), encoded1*encoded2]
        diff = torch.abs(encoded1 - encoded2)
        prod = encoded1 * encoded2
        combined = torch.cat([encoded1, encoded2, diff, prod], dim=1)
        
        # Feed through fully connected layers
        x = self.fc1(combined)
        x = self.batch_norm1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.batch_norm2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc3(x)
        output = torch.sigmoid(x)
        
        return output


class AttentionEncoder(nn.Module):
    """
    Architecture 2: Attention-based Encoder Network
    
    Uses self-attention mechanism to capture important words in clauses,
    then computes similarity based on attended representations.
    """
    
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, num_heads=4, dropout=0.3):
        super(AttentionEncoder, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.embedding_dim = embedding_dim
        
        # Self-attention layers
        self.attention = nn.MultiheadAttention(
            embedding_dim, 
            num_heads, 
            dropout=dropout,
            batch_first=True
        )
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim)
        )
        
        self.layer_norm1 = nn.LayerNorm(embedding_dim)
        self.layer_norm2 = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Similarity computation
        self.fc1 = nn.Linear(embedding_dim * 4, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)
        
        self.batch_norm1 = nn.BatchNorm1d(256)
        self.batch_norm2 = nn.BatchNorm1d(128)
        
    def create_attention_mask(self, lengths, max_len):
        """Create attention mask for padding."""
        batch_size = lengths.size(0)
        mask = torch.arange(max_len).expand(batch_size, max_len).to(lengths.device)
        mask = mask >= lengths.unsqueeze(1)
        return mask
    
    def encode_clause(self, clause, lengths):
        """Encode clause using self-attention."""
        # Embedding
        embedded = self.embedding(clause)
        embedded = self.dropout(embedded)
        
        # Create attention mask
        max_len = clause.size(1)
        attn_mask = self.create_attention_mask(lengths, max_len)
        
        # Self-attention with residual connection
        attended, _ = self.attention(
            embedded, embedded, embedded,
            key_padding_mask=attn_mask
        )
        attended = self.layer_norm1(embedded + self.dropout(attended))
        
        # Feed-forward with residual connection
        ffn_output = self.ffn(attended)
        output = self.layer_norm2(attended + self.dropout(ffn_output))
        
        # Mean pooling (excluding padding)
        mask = (~attn_mask).float().unsqueeze(-1)
        masked_output = output * mask
        summed = masked_output.sum(dim=1)
        lengths_expanded = lengths.unsqueeze(-1).float()
        encoded = summed / lengths_expanded
        
        return encoded
    
    def forward(self, clause1, clause2, lengths1, lengths2):
        """Forward pass for clause pair."""
        # Encode both clauses
        encoded1 = self.encode_clause(clause1, lengths1)
        encoded2 = self.encode_clause(clause2, lengths2)
        
        # Compute similarity features
        diff = torch.abs(encoded1 - encoded2)
        prod = encoded1 * encoded2
        combined = torch.cat([encoded1, encoded2, diff, prod], dim=1)
        
        # Similarity scoring
        x = self.fc1(combined)
        x = self.batch_norm1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.batch_norm2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc3(x)
        output = torch.sigmoid(x)
        
        return output


class CNNBiLSTMHybrid(nn.Module):
    """
    Architecture 3: CNN-BiLSTM Hybrid Network
    
    Combines CNN for local feature extraction with BiLSTM for
    sequential modeling, providing a comprehensive representation.
    """
    
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, 
                 num_filters=128, filter_sizes=[3, 4, 5], num_layers=2, dropout=0.3):
        super(CNNBiLSTMHybrid, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # CNN layers with multiple filter sizes
        self.convs = nn.ModuleList([
            nn.Conv1d(embedding_dim, num_filters, kernel_size=fs)
            for fs in filter_sizes
        ])
        
        # BiLSTM layer
        cnn_output_dim = num_filters * len(filter_sizes)
        self.lstm = nn.LSTM(
            cnn_output_dim,
            hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # Similarity computation
        lstm_output_dim = hidden_dim * 2  # Bidirectional
        self.fc1 = nn.Linear(lstm_output_dim * 4, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)
        
        self.batch_norm1 = nn.BatchNorm1d(256)
        self.batch_norm2 = nn.BatchNorm1d(128)
        
    def encode_clause(self, clause, lengths):
        """Encode clause using CNN-BiLSTM hybrid."""
        # Embedding
        embedded = self.embedding(clause)  # (batch, seq_len, embedding_dim)
        embedded = self.dropout(embedded)
        
        # CNN expects (batch, embedding_dim, seq_len)
        embedded_transposed = embedded.transpose(1, 2)
        
        # Apply multiple convolutional filters
        conv_outputs = []
        for conv in self.convs:
            conv_out = F.relu(conv(embedded_transposed))  # (batch, num_filters, seq_len - kernel + 1)
            # Transpose back to (batch, seq_len, num_filters)
            conv_out = conv_out.transpose(1, 2)
            conv_outputs.append(conv_out)
        
        # Concatenate all CNN outputs
        # Need to pad to same length
        max_len = max(co.size(1) for co in conv_outputs)
        padded_convs = []
        for co in conv_outputs:
            if co.size(1) < max_len:
                padding = torch.zeros(co.size(0), max_len - co.size(1), co.size(2)).to(co.device)
                co = torch.cat([co, padding], dim=1)
            padded_convs.append(co)
        
        cnn_output = torch.cat(padded_convs, dim=2)  # (batch, max_len, total_filters)
        cnn_output = self.dropout(cnn_output)
        
        # BiLSTM
        # Adjust lengths for CNN output
        adjusted_lengths = torch.clamp(lengths - max(self.convs[0].kernel_size[0] - 1 for _ in range(1)), min=1)
        
        packed = pack_padded_sequence(
            cnn_output[:, :cnn_output.size(1), :],
            adjusted_lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )
        
        packed_output, (hidden, cell) = self.lstm(packed)
        
        # Use last hidden state
        forward_hidden = hidden[-2, :, :]
        backward_hidden = hidden[-1, :, :]
        encoded = torch.cat([forward_hidden, backward_hidden], dim=1)
        
        return encoded
    
    def forward(self, clause1, clause2, lengths1, lengths2):
        """Forward pass for clause pair."""
        # Encode both clauses
        encoded1 = self.encode_clause(clause1, lengths1)
        encoded2 = self.encode_clause(clause2, lengths2)
        
        # Compute similarity features
        diff = torch.abs(encoded1 - encoded2)
        prod = encoded1 * encoded2
        combined = torch.cat([encoded1, encoded2, diff, prod], dim=1)
        
        # Similarity scoring
        x = self.fc1(combined)
        x = self.batch_norm1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.batch_norm2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc3(x)
        output = torch.sigmoid(x)
        
        return output


# ============================================================================
# TRAINING AND EVALUATION
# ============================================================================

class ModelTrainer:
    """Handles model training and evaluation."""
    
    def __init__(self, model, device, model_name="Model"):
        self.model = model.to(device)
        self.device = device
        self.model_name = model_name
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
    def train_epoch(self, train_loader, optimizer, criterion):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        from tqdm import tqdm
        for clause1, clause2, lengths1, lengths2, labels in tqdm(train_loader, desc="Training", leave=False):
            clause1 = clause1.to(self.device)
            clause2 = clause2.to(self.device)
            lengths1 = lengths1.to(self.device)
            lengths2 = lengths2.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = self.model(clause1, clause2, lengths1, lengths2)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            predicted = (outputs >= 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def evaluate(self, val_loader, criterion):
        """Evaluate the model."""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for clause1, clause2, lengths1, lengths2, labels in val_loader:
                clause1 = clause1.to(self.device)
                clause2 = clause2.to(self.device)
                lengths1 = lengths1.to(self.device)
                lengths2 = lengths2.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(clause1, clause2, lengths1, lengths2)
                loss = criterion(outputs, labels)
                
                # Statistics
                total_loss += loss.item()
                predicted = (outputs >= 0.5).float()
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
                
                # Store for metrics calculation
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(outputs.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy, all_predictions, all_labels, all_probs
    
    def train(self, train_loader, val_loader, epochs=20, learning_rate=0.001):
        """Train the model for multiple epochs."""
        print(f"\n{'='*60}")
        print(f"TRAINING {self.model_name}")
        print(f"{'='*60}")
        
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3
        )
        
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 7
        
        for epoch in range(epochs):
            # Train
            train_loss, train_acc = self.train_epoch(train_loader, optimizer, criterion)
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            
            # Validate
            val_loss, val_acc, _, _, _ = self.evaluate(val_loader, criterion)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            
            # Learning rate scheduling
            old_lr = optimizer.param_groups[0]['lr']
            scheduler.step(val_loss)
            new_lr = optimizer.param_groups[0]['lr']
            if old_lr != new_lr:
                print(f"   Learning rate reduced: {old_lr:.6f} → {new_lr:.6f}")
            
            print(f"Epoch [{epoch+1}/{epochs}] | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), f'{self.model_name}_best.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered after epoch {epoch+1}")
                    break
        
        # Load best model
        self.model.load_state_dict(torch.load(f'{self.model_name}_best.pth'))
        print(f"\n✓ Training completed! Best validation loss: {best_val_loss:.4f}")
        
        return self.model


class ModelEvaluator:
    """Comprehensive model evaluation with all required metrics."""
    
    def __init__(self, model, model_name, device):
        self.model = model
        self.model_name = model_name
        self.device = device
        self.results = {}
        
    def evaluate_comprehensive(self, test_loader):
        """Compute all evaluation metrics."""
        print(f"\n{'='*60}")
        print(f"EVALUATING {self.model_name}")
        print(f"{'='*60}")
        
        self.model.eval()
        all_predictions = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for clause1, clause2, lengths1, lengths2, labels in test_loader:
                clause1 = clause1.to(self.device)
                clause2 = clause2.to(self.device)
                lengths1 = lengths1.to(self.device)
                lengths2 = lengths2.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(clause1, clause2, lengths1, lengths2)
                predicted = (outputs >= 0.5).float()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(outputs.cpu().numpy())
        
        # Convert to numpy arrays
        all_predictions = np.array(all_predictions).flatten()
        all_labels = np.array(all_labels).flatten()
        all_probs = np.array(all_probs).flatten()
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='binary', zero_division=0
        )
        
        # ROC-AUC
        try:
            roc_auc = roc_auc_score(all_labels, all_probs)
            fpr, tpr, _ = roc_curve(all_labels, all_probs)
        except:
            roc_auc = 0.0
            fpr, tpr = None, None
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)
        
        # Store results
        self.results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'confusion_matrix': cm,
            'predictions': all_predictions,
            'labels': all_labels,
            'probabilities': all_probs,
            'fpr': fpr,
            'tpr': tpr
        }
        
        # Print results
        print(f"\nAccuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        print(f"ROC-AUC:   {roc_auc:.4f}")
        print(f"\nConfusion Matrix:")
        print(cm)
        print(f"\nClassification Report:")
        print(classification_report(all_labels, all_predictions, 
                                   target_names=['Dissimilar', 'Similar']))
        
        return self.results


# ============================================================================
# VISUALIZATION AND ANALYSIS
# ============================================================================

class ResultsVisualizer:
    """Visualize and compare model results."""
    
    @staticmethod
    def plot_training_curves(trainers, save_path='training_curves.png'):
        """Plot training and validation curves for all models."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss curves
        for trainer in trainers:
            epochs = range(1, len(trainer.train_losses) + 1)
            axes[0].plot(epochs, trainer.train_losses, label=f'{trainer.model_name} - Train', linestyle='--')
            axes[0].plot(epochs, trainer.val_losses, label=f'{trainer.model_name} - Val')
        
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Accuracy curves
        for trainer in trainers:
            epochs = range(1, len(trainer.train_accuracies) + 1)
            axes[1].plot(epochs, trainer.train_accuracies, label=f'{trainer.model_name} - Train', linestyle='--')
            axes[1].plot(epochs, trainer.val_accuracies, label=f'{trainer.model_name} - Val')
        
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Training and Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Training curves saved to {save_path}")
        plt.close()
    
    @staticmethod
    def plot_roc_curves(evaluators, save_path='roc_curves.png'):
        """Plot ROC curves for all models."""
        plt.figure(figsize=(10, 8))
        
        for evaluator in evaluators:
            results = evaluator.results
            if results['fpr'] is not None and results['tpr'] is not None:
                plt.plot(results['fpr'], results['tpr'], 
                        label=f"{evaluator.model_name} (AUC = {results['roc_auc']:.4f})",
                        linewidth=2)
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves - Model Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ ROC curves saved to {save_path}")
        plt.close()
    
    @staticmethod
    def plot_confusion_matrices(evaluators, save_path='confusion_matrices.png'):
        """Plot confusion matrices for all models."""
        n_models = len(evaluators)
        fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))
        
        if n_models == 1:
            axes = [axes]
        
        for idx, evaluator in enumerate(evaluators):
            cm = evaluator.results['confusion_matrix']
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                       xticklabels=['Dissimilar', 'Similar'],
                       yticklabels=['Dissimilar', 'Similar'])
            axes[idx].set_title(f'{evaluator.model_name}\nConfusion Matrix')
            axes[idx].set_ylabel('True Label')
            axes[idx].set_xlabel('Predicted Label')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Confusion matrices saved to {save_path}")
        plt.close()
    
    @staticmethod
    def plot_metrics_comparison(evaluators, save_path='metrics_comparison.png'):
        """Create bar plot comparing all metrics across models."""
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        model_names = [e.model_name for e in evaluators]
        
        data = {metric: [e.results[metric] for e in evaluators] for metric in metrics}
        
        x = np.arange(len(metrics))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for i, model_name in enumerate(model_names):
            values = [data[metric][i] for metric in metrics]
            ax.bar(x + i*width, values, width, label=model_name)
        
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Score')
        ax.set_title('Model Performance Comparison')
        ax.set_xticks(x + width)
        ax.set_xticklabels(['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 1.0])
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Metrics comparison saved to {save_path}")
        plt.close()
    
    @staticmethod
    def create_results_table(evaluators, save_path='results_table.csv'):
        """Create a comprehensive results table."""
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        
        results_dict = {'Model': [e.model_name for e in evaluators]}
        for metric in metrics:
            results_dict[metric.replace('_', ' ').title()] = [
                f"{e.results[metric]:.4f}" for e in evaluators
            ]
        
        df = pd.DataFrame(results_dict)
        df.to_csv(save_path, index=False)
        print(f"\n✓ Results table saved to {save_path}")
        print("\n" + "="*60)
        print("QUANTITATIVE RESULTS SUMMARY")
        print("="*60)
        print(df.to_string(index=False))
        
        return df


class QualitativeAnalyzer:
    """Analyze qualitative results - correctly and incorrectly matched clauses."""
    
    def __init__(self, model, vocabulary, device, model_name):
        self.model = model
        self.vocabulary = vocabulary
        self.device = device
        self.model_name = model_name
        
    def analyze_predictions(self, test_pairs, test_labels, num_examples=5):
        """Show examples of correct and incorrect predictions."""
        print(f"\n{'='*60}")
        print(f"QUALITATIVE ANALYSIS - {self.model_name}")
        print(f"{'='*60}")
        
        self.model.eval()
        
        correct_similar = []
        correct_dissimilar = []
        incorrect_similar = []
        incorrect_dissimilar = []
        
        with torch.no_grad():
            for idx, ((clause1, clause2), label) in enumerate(zip(test_pairs, test_labels)):
                # Encode clauses
                encoded1 = self.vocabulary.encode(clause1)[:100]
                encoded2 = self.vocabulary.encode(clause2)[:100]
                
                tensor1 = torch.LongTensor(encoded1).unsqueeze(0).to(self.device)
                tensor2 = torch.LongTensor(encoded2).unsqueeze(0).to(self.device)
                length1 = torch.LongTensor([len(encoded1)]).to(self.device)
                length2 = torch.LongTensor([len(encoded2)]).to(self.device)
                
                # Predict
                output = self.model(tensor1, tensor2, length1, length2)
                prediction = (output.item() >= 0.5)
                
                # Categorize
                if prediction == label:
                    if label == 1:
                        correct_similar.append((clause1, clause2, output.item()))
                    else:
                        correct_dissimilar.append((clause1, clause2, output.item()))
                else:
                    if label == 1:
                        incorrect_similar.append((clause1, clause2, output.item()))
                    else:
                        incorrect_dissimilar.append((clause1, clause2, output.item()))
                
                # Stop when we have enough examples
                if (len(correct_similar) >= num_examples and 
                    len(correct_dissimilar) >= num_examples and
                    len(incorrect_similar) >= num_examples and
                    len(incorrect_dissimilar) >= num_examples):
                    break
        
        # Display examples
        print("\n" + "="*60)
        print("CORRECTLY MATCHED SIMILAR CLAUSES")
        print("="*60)
        for i, (c1, c2, score) in enumerate(correct_similar[:num_examples], 1):
            print(f"\nExample {i} (Similarity Score: {score:.4f}):")
            print(f"Clause 1: {c1[:200]}...")
            print(f"Clause 2: {c2[:200]}...")
        
        print("\n" + "="*60)
        print("CORRECTLY MATCHED DISSIMILAR CLAUSES")
        print("="*60)
        for i, (c1, c2, score) in enumerate(correct_dissimilar[:num_examples], 1):
            print(f"\nExample {i} (Similarity Score: {score:.4f}):")
            print(f"Clause 1: {c1[:200]}...")
            print(f"Clause 2: {c2[:200]}...")
        
        print("\n" + "="*60)
        print("INCORRECTLY MATCHED SIMILAR CLAUSES (Should be Similar)")
        print("="*60)
        for i, (c1, c2, score) in enumerate(incorrect_similar[:num_examples], 1):
            print(f"\nExample {i} (Similarity Score: {score:.4f}):")
            print(f"Clause 1: {c1[:200]}...")
            print(f"Clause 2: {c2[:200]}...")
        
        print("\n" + "="*60)
        print("INCORRECTLY MATCHED DISSIMILAR CLAUSES (Should be Dissimilar)")
        print("="*60)
        for i, (c1, c2, score) in enumerate(incorrect_dissimilar[:num_examples], 1):
            print(f"\nExample {i} (Similarity Score: {score:.4f}):")
            print(f"Clause 1: {c1[:200]}...")
            print(f"Clause 2: {c2[:200]}...")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    print("\n" + "="*60)
    print("LEGAL CLAUSE SIMILARITY DETECTION SYSTEM")
    print("Deep Learning Assignment 2 - Google Colab Version")
    print("="*60)
    
    # Configuration
    BATCH_SIZE = 32
    EPOCHS = 20
    LEARNING_RATE = 0.001
    NUM_PAIRS = 20000
    MAX_VOCAB_SIZE = 30000
    EMBEDDING_DIM = 128
    HIDDEN_DIM = 256
    
    # Step 1: Load and preprocess data
    data_loader = LegalClauseDataLoader(data_dir=DATA_DIR)
    clauses, clause_types = data_loader.load_data()
    
    # Step 2: Build vocabulary
    vocabulary = Vocabulary(max_vocab_size=MAX_VOCAB_SIZE, min_freq=2)
    vocabulary.build_vocab(clauses)
    
    # Step 3: Generate clause pairs
    pair_generator = ClausePairGenerator(clauses, clause_types)
    pairs, labels = pair_generator.generate_pairs(num_pairs=NUM_PAIRS)
    
    # Step 4: Split data
    print(f"\n{'='*60}")
    print("SPLITTING DATA")
    print(f"{'='*60}")
    
    # Split into train, validation, and test sets
    train_pairs, temp_pairs, train_labels, temp_labels = train_test_split(
        pairs, labels, test_size=0.3, random_state=SEED, stratify=labels
    )
    val_pairs, test_pairs, val_labels, test_labels = train_test_split(
        temp_pairs, temp_labels, test_size=0.5, random_state=SEED, stratify=temp_labels
    )
    
    print(f"✓ Training set: {len(train_pairs)} pairs")
    print(f"✓ Validation set: {len(val_pairs)} pairs")
    print(f"✓ Test set: {len(test_pairs)} pairs")
    
    # Create datasets
    train_dataset = ClausePairDataset(train_pairs, train_labels, vocabulary)
    val_dataset = ClausePairDataset(val_pairs, val_labels, vocabulary)
    test_dataset = ClausePairDataset(test_pairs, test_labels, vocabulary)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                             collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                           collate_fn=collate_fn, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                            collate_fn=collate_fn, num_workers=0)
    
    # Step 5: Initialize models
    print(f"\n{'='*60}")
    print("INITIALIZING MODELS")
    print(f"{'='*60}")
    
    vocab_size = len(vocabulary)
    
    model1 = BiLSTMSiameseNetwork(
        vocab_size=vocab_size,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=2,
        dropout=0.3
    )
    print(f"\n1. BiLSTM Siamese Network initialized")
    print(f"   Parameters: {sum(p.numel() for p in model1.parameters()):,}")
    
    model2 = AttentionEncoder(
        vocab_size=vocab_size,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_heads=4,
        dropout=0.3
    )
    print(f"\n2. Attention-based Encoder initialized")
    print(f"   Parameters: {sum(p.numel() for p in model2.parameters()):,}")
    
    # Step 6: Train models
    trainer1 = ModelTrainer(model1, device, model_name="BiLSTM-Siamese")
    trainer1.train(train_loader, val_loader, epochs=EPOCHS, learning_rate=LEARNING_RATE)
    
    trainer2 = ModelTrainer(model2, device, model_name="Attention-Encoder")
    trainer2.train(train_loader, val_loader, epochs=EPOCHS, learning_rate=LEARNING_RATE)
    
    trainers = [trainer1, trainer2]
    
    # Step 7: Evaluate models
    evaluator1 = ModelEvaluator(model1, "BiLSTM-Siamese", device)
    results1 = evaluator1.evaluate_comprehensive(test_loader)
    
    evaluator2 = ModelEvaluator(model2, "Attention-Encoder", device)
    results2 = evaluator2.evaluate_comprehensive(test_loader)
    
    evaluators = [evaluator1, evaluator2]
    
    # Step 8: Visualize results
    print(f"\n{'='*60}")
    print("GENERATING VISUALIZATIONS")
    print(f"{'='*60}")
    
    visualizer = ResultsVisualizer()
    visualizer.plot_training_curves(trainers)
    visualizer.plot_roc_curves(evaluators)
    visualizer.plot_confusion_matrices(evaluators)
    visualizer.plot_metrics_comparison(evaluators)
    visualizer.create_results_table(evaluators)
    
    # Step 9: Qualitative analysis
    for evaluator in evaluators:
        analyzer = QualitativeAnalyzer(
            evaluator.model, vocabulary, device, evaluator.model_name
        )
        analyzer.analyze_predictions(test_pairs, test_labels, num_examples=3)
    
    # Step 10: Comparative analysis
    print(f"\n{'='*60}")
    print("COMPARATIVE ANALYSIS & DISCUSSION")
    print(f"{'='*60}")
    
    print("\n1. BiLSTM-Siamese Network:")
    print("   Strengths:")
    print("   - Effectively captures sequential dependencies in legal text")
    print("   - Bidirectional processing provides context from both directions")
    print("   - Shared weights ensure consistent encoding of clause pairs")
    print("   Weaknesses:")
    print("   - May struggle with very long clauses due to vanishing gradients")
    print("   - Sequential processing limits parallelization")
    
    print("\n2. Attention-based Encoder:")
    print("   Strengths:")
    print("   - Self-attention focuses on important legal terms and phrases")
    print("   - Parallel processing of all tokens improves efficiency")
    print("   - Better at capturing long-range dependencies")
    print("   Weaknesses:")
    print("   - Requires more data to learn effective attention patterns")
    print("   - Quadratic complexity with sequence length")
    
    print("\n" + "="*60)
    print("EXECUTION COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\n✓ Generated files:")
    print("  - training_curves.png")
    print("  - roc_curves.png")
    print("  - confusion_matrices.png")
    print("  - metrics_comparison.png")
    print("  - results_table.csv")
    print("  - *_best.pth (model checkpoints)")
    
    if IN_COLAB:
        print(f"\n✓ All files saved to: {os.getcwd()}")
        print("  You can download them from the Colab file browser")
        print("  Or they will be automatically saved if you mounted Drive with proper permissions")


if __name__ == "__main__":
    main()

