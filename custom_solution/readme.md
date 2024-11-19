# Wikidata Entity Classification Solution with Multi-Task BERT Architecture

## Detailed Solution Architecture

### 1. Base Model
- Use of BERT (bert-base-uncased) as the main encoder
- Transformer architecture with 12 layers, 768 hidden dimensions
- WordPiece-based tokenization with 30522 token vocabulary

### 2. Multi-Task Architecture
The architecture consists of three distinct classification branches:

#### Branch 1: B/I/O Classification
- Output: 3 classes (Begin/Inside/Outside)
- Detects entity boundaries
- Architecture:
  ```
  Input -> BERT -> Common Linear -> ReLU -> Linear(768, 3)
  ```

#### Branch 2: Prefix Classification
- Output: 5 classes (wd/wdt/ws/wq/none)
- Identifies Wikidata entity type
- Architecture:
  ```
  Input -> BERT -> Common Linear -> ReLU -> Linear(768, 5)
  ```

#### Branch 3: Label Classification
- Output: N classes (all possible Wikidata labels)
- Identifies specific entity
- Architecture:
  ```
  Input -> BERT -> Common Linear -> ReLU -> Linear(768, N)
  ```

### 3. Common Linear Layer
```python
self.common_layer = nn.Linear(config.hidden_size, config.hidden_size)
self.relu = nn.ReLU()
```
This layer enables:
- Feature sharing between tasks
- Reduction in parameter count
- Better generalization

### 4. Loss Function
```python
loss = loss_fct_BIO(logits.view(-1, self.num_BIO), labels[:, :, 0].view(-1)) + \
       loss_fct_prefix(logits_dim2.view(-1, self.num_prefix), labels[:, :, 1].view(-1)) + \
       loss_fct_wikidata(logits_dim3.view(-1, self.num_wikidata_label), labels[:, :, 2].view(-1))
```
- Combination of three cross-entropy losses
- Equal weighting between tasks
- Ignores padding tokens

## Data Enrichment

### 1. Wikidata Augmentation
```python
# For each Wikidata entity
for id, info in wikidata_data.items():
    # Generate sentences with label
    sentence_to_add = ['wikidata', 'label', ':'] + splitted_label
    # Generate sentences with description
    sent = ['wikidata', 'description', ':'] + description.split()
```

### 2. Augmentation Strategies
- Addition of sentences with labels
- Addition of sentences with descriptions
- Random combination of components
- Increased vocabulary diversity

## Data Preprocessing

### 1. Tokenization
```python
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
```
- Use of BERT tokenizer
- Special token handling ([CLS], [SEP])
- Padding to maximum length of 150

### 2. Label Encoding
```python
def transform_labels_to_triplet(labels):
    # Transform into triplets (B/I/O, prefix, label)
    triplet = [None, None, None]
    triplet[0] = l[0]  # B/I/O
    triplet[1] = l[2:].split(':')[0]  # Prefix
    triplet[2] = l[2:].split(':')[1]  # Label
```

## Training

### 1. Hyperparameters
- Learning rate: 5e-5
- Batch size: 32 then 16
- Epochs: 10 + 4
- Optimizer: AdamW
- Weight decay: 0.01

### 2. Training Strategy
1. Initial training with batch 32
2. Fine-tuning with batch 16
3. Continuous validation on test set

## Architecture Advantages

1. **Efficient Decomposition**
   - Separation of problem aspects
   - Progressive concept learning

2. **Knowledge Sharing**
   - Common linear layer
   - Information transfer between tasks

3. **Flexibility**
   - Easily extensible
   - Adaptable to new entity types

4. **Robustness**
   - Better handling of ambiguous cases
   - Reduction in overfitting

## Limitations and Challenges

1. **Unknown Entities**
   - Difficulty with unseen entities
   - Need for unknown case handling strategy

2. **Computational Complexity**
   - Three classification heads
   - Increased training time

3. **Task Balancing**
   - Relative importance of losses
   - Convergence of different branches

## Metrics and Performance

```
Validation results:
- Loss: 0.63
- Accuracy: 0.90
- F1 Score: 0.76
```
