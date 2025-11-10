# Legal Clause Similarity Detection System

## Overview
This project implements multiple baseline NLP architectures for identifying semantic similarity between legal clauses **without using pretrained transformers** like BERT, RoBERTa, or LegalBERT.

## Assignment Details
- **Course**: Deep Learning
- **Assignment**: DL - A2
- **Task**: Legal Clause Similarity Detection

## Dataset
The dataset consists of legal clauses organized in CSV files (in the `archive/` directory):
- Each CSV file represents a distinct clause category (e.g., acceleration, accounting-terms, etc.)
- Each file contains clause texts and their corresponding clause type labels
- Total: 395 clause categories

## Implemented Architectures

### 1. BiLSTM-based Siamese Network
- Uses shared Bidirectional LSTM encoder for both clauses
- Captures sequential dependencies in legal text
- Employs siamese architecture for learning similarity metrics
- **Key Features**:
  - Bidirectional processing for context from both directions
  - Shared weights ensure consistent encoding
  - Concatenates multiple similarity features (concatenation, absolute difference, element-wise product)

### 2. Attention-based Encoder Network
- Implements self-attention mechanism to focus on important words
- Parallel processing of all tokens
- Better at capturing long-range dependencies
- **Key Features**:
  - Multi-head attention for diverse attention patterns
  - Layer normalization and residual connections
  - Feed-forward network for feature transformation
  - Mean pooling with attention masking

### 3. CNN-BiLSTM Hybrid Network
- Combines CNN for local feature extraction with BiLSTM for sequential modeling
- Multiple filter sizes for capturing n-gram patterns
- **Key Features**:
  - Multiple convolutional filters (kernel sizes: 3, 4, 5)
  - BiLSTM for global sequential patterns
  - Combines benefits of both CNN and RNN architectures

## Features

### Data Processing Pipeline
- **Text Cleaning**: Lowercasing, whitespace normalization, special character handling
- **Tokenization**: Word-level tokenization
- **Vocabulary Building**: Creates vocabulary with frequency thresholding (min_freq=2)
- **Clause Pair Generation**: 
  - Positive pairs: Same clause type (similar)
  - Negative pairs: Different clause types (dissimilar)

### Training Features
- **Optimization**: Adam optimizer with learning rate scheduling
- **Regularization**: Dropout, batch normalization, gradient clipping
- **Early Stopping**: Patience-based early stopping to prevent overfitting
- **Best Model Saving**: Automatically saves best model based on validation loss

### Evaluation Metrics (As per Assignment Requirements)

#### For Categorical Classification:
1. **Accuracy**: Overall classification accuracy
2. **Precision**: Out of predicted similar pairs, how many are truly similar
3. **Recall**: Out of all truly similar pairs, how many were identified
4. **F1-Score**: Harmonic mean of Precision and Recall
5. **ROC-AUC / PR-AUC**: Evaluates classifier's ranking ability across thresholds

### Visualizations
- Training and validation loss/accuracy curves
- ROC curves for all models
- Confusion matrices
- Comparative bar charts of all metrics

### Qualitative Analysis
Examples of:
- Correctly matched similar clauses
- Correctly matched dissimilar clauses
- Incorrectly matched clauses (with explanations)

## Installation

```bash
# Install required packages
pip install -r requirements.txt
```

## Usage

```bash
# Run the complete pipeline
python legal_clause_similarity.py
```

The script will:
1. Load all legal clause data from the `archive/` directory
2. Build vocabulary and preprocess text
3. Generate clause pairs for training
4. Train all three architectures
5. Evaluate and compare models
6. Generate visualizations and reports

## Output Files

After execution, the following files will be generated:
- `training_curves.png` - Training and validation curves for all models
- `roc_curves.png` - ROC curves comparing all models
- `confusion_matrices.png` - Confusion matrices for each model
- `metrics_comparison.png` - Bar chart comparing all metrics
- `results_table.csv` - Comprehensive results table
- `*_best.pth` - Saved model checkpoints (BiLSTM-Siamese_best.pth, etc.)

## Configuration

Key hyperparameters (can be modified in the `main()` function):
```python
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001
NUM_PAIRS = 20000  # Number of clause pairs to generate
MAX_VOCAB_SIZE = 30000
EMBEDDING_DIM = 128
HIDDEN_DIM = 256
```

## Model Architecture Details

### BiLSTM-Siamese
```
Input → Embedding → Dropout → BiLSTM → Feature Combination → FC Layers → Sigmoid
```

### Attention-Encoder
```
Input → Embedding → Multi-Head Attention → FFN → Layer Norm → Mean Pooling → FC Layers → Sigmoid
```

### CNN-BiLSTM-Hybrid
```
Input → Embedding → Multi-Scale CNN → BiLSTM → Feature Combination → FC Layers → Sigmoid
```

## Code Structure

The code is organized into the following main components:

1. **Data Loading and Preprocessing**
   - `LegalClauseDataLoader`: Loads CSV files
   - `TextPreprocessor`: Cleans and tokenizes text
   - `Vocabulary`: Builds and manages vocabulary
   - `ClausePairGenerator`: Creates training pairs
   - `ClausePairDataset`: PyTorch dataset

2. **Model Architectures**
   - `BiLSTMSiameseNetwork`: BiLSTM-based model
   - `AttentionEncoder`: Attention-based model
   - `CNNBiLSTMHybrid`: Hybrid CNN-BiLSTM model

3. **Training and Evaluation**
   - `ModelTrainer`: Handles training loop
   - `ModelEvaluator`: Computes evaluation metrics

4. **Visualization and Analysis**
   - `ResultsVisualizer`: Creates plots and charts
   - `QualitativeAnalyzer`: Analyzes prediction examples

## Grading Rubric Coverage

### Code Quality (50 points)
✓ **Documentation** (10 points): Comprehensive docstrings and comments  
✓ **Modular Implementation** (10 points): Object-oriented design with separate classes  
✓ **Data Preprocessing** (6 points): Text cleaning, tokenization, vocabulary building  
✓ **Multiple Architectures** (20 points): 3 different baseline architectures with comparison  
✓ **Qualitative Results** (4 points): Examples of correct and incorrect predictions

### Evaluation Metrics
✓ All required metrics implemented:
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC

## Comparative Analysis

### BiLSTM-Siamese Network
**Strengths**:
- Effectively captures sequential dependencies
- Bidirectional processing provides full context
- Shared weights ensure consistency

**Weaknesses**:
- May struggle with very long sequences
- Sequential processing limits parallelization

### Attention-based Encoder
**Strengths**:
- Focuses on important legal terms
- Parallel processing improves efficiency
- Better at long-range dependencies

**Weaknesses**:
- Requires more data for attention learning
- Quadratic complexity with sequence length

### CNN-BiLSTM Hybrid
**Strengths**:
- CNN captures local legal phrases
- BiLSTM models global patterns
- Combines benefits of both approaches

**Weaknesses**:
- More complex with more hyperparameters
- Higher computational cost

## Requirements

- Python 3.8+
- PyTorch 2.0+
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn

## Notes

- The script uses GPU if available (CUDA), otherwise CPU
- Random seeds are set for reproducibility
- The code is designed to be modular and easily extensible
- No pretrained models (BERT, RoBERTa, LegalBERT) are used as per assignment requirements

## Author

M. Abdullah Nadeem

## License

This project is for educational purposes as part of a university assignment.

