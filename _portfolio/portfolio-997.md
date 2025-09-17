---
title: "NLP Challenge: Authentic Sentence Detection and Synthetic Corruption Generation"
excerpt: "My experience with a challenging NLP task that involved distinguishing English sentences from their corrupted versions and generating new corruptions.
collection: portfolio
---

Description:
------
In the dynamic field of Natural Language Processing (NLP), challenges often push the boundaries of what we can achieve with language models. Recently, I undertook a research challenge that involved two intriguing tasks: building a classifier to distinguish between an English sentence and its corrupted version, and generating new corruptions that are challenging for the classifier to identify. This project not only honed my skills in NLP but also provided valuable insights into model robustness and data manipulation.

The challenge was divided into two parts. The first task required developing a classifier capable of identifying which of two sentences was the authentic English version. The second task involved creating new, synthetic corruptions of the original sentences that would be difficult for the classifier to distinguish from the authentic ones. This exercise was designed to test the limits of NLP models and to explore the creation of challenging datasets.

Task 1: Building the Classifier
------
For the first task, I opted for an LSTM (Long Short-Term Memory) model due to its effectiveness in handling sequential data. The process began with data preprocessing, where I cleaned and prepared the dataset for training. This involved converting text to lowercase, removing punctuation, and creating a vocabulary to map words to integers for input into the neural network.

The LSTM model was designed with an embedding layer, followed by LSTM layers, and a fully connected layer with a sigmoid activation function for binary classification. The model was trained using a binary cross-entropy loss function, and I employed Adam optimization for training.

During training, I encountered challenges such as overfitting and ensuring adequate generalization. To address these, I implemented techniques like dropout regularization and early stopping. The model achieved a validation accuracy close to the expected benchmark, demonstrating its effectiveness in distinguishing between authentic and corrupted sentences.

Task 2: Generating New Corruptions
------
The second task was more creative, involving the generation of new corruptions that would challenge the classifier. I devised several strategies, including typographical errors, punctuation mistakes, and word duplications, among others. To ensure uniqueness, I implemented checks to prevent direct copying from the training data.

One of the key aspects of this task was ensuring that the new corruptions were plausible enough to fool the classifier. This required a delicate balance between making the corruption evident enough to be considered incorrect but not so obvious that it was easily identifiable.

Reflections and Learnings
------
This challenge provided a deep dive into the intricacies of NLP model robustness and data manipulation. I learned the importance of careful data preprocessing and the impact of model architecture choices on performance. Additionally, the exercise highlighted the creativity required in generating synthetic data that can effectively test model capabilities.

Source code
------
Task 1:
```python
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import string
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import accuracy_score

def calculate_accuracy(model, loader, device):
    """Calculate accuracy on given loader."""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for texts_a, texts_b in loader:
            texts_a, texts_b = texts_a.to(device), texts_b.to(device)
            
            # Get predictions
            output_a = model(texts_a)
            output_b = model(texts_b)
            
            # Create predictions (1 if output_a > output_b, 0 otherwise)
            batch_preds = (output_a > output_b).float().squeeze()
            batch_labels = torch.ones_like(batch_preds)
            
            batch_preds = batch_preds.cpu().numpy()
            batch_labels = batch_labels.cpu().numpy()
            
            all_preds.extend(batch_preds)
            all_labels.extend(batch_labels)
    
    return accuracy_score(all_labels, all_preds)

def plot_training_progress(history):
    """Plot training and validation metrics."""
    train_loss = [loss.cpu().numpy() if torch.is_tensor(loss) else loss for loss in history['train_loss']]
    val_loss = [loss.cpu().numpy() if torch.is_tensor(loss) else loss for loss in history['val_loss']]
    train_acc = [acc.cpu().numpy() if torch.is_tensor(acc) else acc for acc in history['train_acc']]
    val_acc = [acc.cpu().numpy() if torch.is_tensor(acc) else acc for acc in history['val_acc']]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot losses
    ax1.plot(train_loss, label='Training Loss')
    ax1.plot(val_loss, label='Validation Loss')
    ax1.set_title('Loss Over Time')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracies
    ax2.plot(train_acc, label='Training Accuracy')
    ax2.plot(val_acc, label='Validation Accuracy')
    ax2.set_title('Accuracy Over Time')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_progress.png')
    plt.close()

class TextDataset(Dataset):
    def __init__(self, texts_a, texts_b, vocab):
        self.texts_a = texts_a
        self.texts_b = texts_b
        self.vocab = vocab
        
    def __len__(self):
        return len(self.texts_a)
    
    def __getitem__(self, idx):
        text_a = torch.tensor([self.vocab.get(w, 0) for w in self.texts_a[idx].split()])
        text_b = torch.tensor([self.vocab.get(w, 0) for w in self.texts_b[idx].split()])
        return text_a, text_b

def collate_fn(batch):
    texts_a, texts_b = zip(*batch)
    texts_a_padded = pad_sequence(texts_a, batch_first=True)
    texts_b_padded = pad_sequence(texts_b, batch_first=True)
    return texts_a_padded, texts_b_padded

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim=100, hidden_dim=128, num_layers=2, dropout=0.5):
        super(LSTMClassifier, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = x.long()  # Ensure input is integer type
        x = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        
        # LSTM output
        _, (h_n, _) = self.lstm(x)  # h_n shape: (num_layers, batch_size, hidden_dim)
        
        # Take the last layer's hidden state
        out = h_n[-1]  # (batch_size, hidden_dim)
        
        # Fully connected layer
        out = self.fc(out)
        out = self.sigmoid(out)
        
        return out

def load_and_preprocess_data(filename):
    """Load and preprocess the data from file."""
    texts_a = []
    texts_b = []
    
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                # Basic preprocessing
                text_a = parts[0].lower().translate(str.maketrans('', '', string.punctuation))
                text_b = parts[1].lower().translate(str.maketrans('', '', string.punctuation))
                texts_a.append(text_a)
                texts_b.append(text_b)
    
    return texts_a, texts_b

def build_vocab(texts, max_vocab_size=10000):
    """Build vocabulary from all texts."""
    words = []
    for text in texts:
        words.extend(text.split())
    
    counter = Counter(words)
    most_common = counter.most_common(max_vocab_size - 1)  # -1 for <UNK>
    vocab = {'<UNK>': 0}
    for word, _ in most_common:
        vocab[word] = len(vocab)
    
    return vocab

def train_model(model, train_loader, val_loader, device, epochs=100, patience=3):
    model = model.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters())
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    # For tracking metrics
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    
    print("Starting training...")
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        start_time = time.time()
        
        # Progress bar for training
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Training]')
        for texts_a, texts_b in train_pbar:
            texts_a, texts_b = texts_a.to(device), texts_b.to(device)
            labels_a = torch.ones(texts_a.size(0), 1).to(device)
            labels_b = torch.zeros(texts_b.size(0), 1).to(device)
            
            optimizer.zero_grad()
            
            output_a = model(texts_a)
            output_b = model(texts_b)
            
            loss_a = criterion(output_a, labels_a)
            loss_b = criterion(output_b, labels_b)
            loss = (loss_a + loss_b) / 2
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        train_loss /= len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0
        
        # Progress bar for validation
        val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} [Validation]')
        with torch.no_grad():
            for texts_a, texts_b in val_pbar:
                texts_a, texts_b = texts_a.to(device), texts_b.to(device)
                labels_a = torch.ones(texts_a.size(0), 1).to(device)
                labels_b = torch.zeros(texts_b.size(0), 1).to(device)
                
                output_a = model(texts_a)
                output_b = model(texts_b)
                
                loss_a = criterion(output_a, labels_a)
                loss_b = criterion(output_b, labels_b)
                val_loss += (loss_a + loss_b) / 2
                
                val_pbar.set_postfix({'loss': f'{(loss_a + loss_b).item()/2:.4f}'})
        
        val_loss /= len(val_loader)
        
        # Calculate accuracies
        train_acc = calculate_accuracy(model, train_loader, device)
        val_acc = calculate_accuracy(model, val_loader, device)
        
        # Update history with NumPy arrays or scalars
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss.item())
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        # Plot current progress
        plot_training_progress(history)
        
        epoch_time = time.time() - start_time
        print(f'\nEpoch {epoch+1}/{epochs} Summary:')
        print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        print(f'Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}')
        print(f'Time: {epoch_time:.2f}s')
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            print("New best model saved!")
            # Save the best model
            torch.save(best_model_state, 'best_model.pth')
        else:
            patience_counter += 1
            print(f"Patience counter: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print(f'Early stopping triggered after epoch {epoch+1}')
                model.load_state_dict(best_model_state)
                break
    
    # Save final metrics to CSV
    pd.DataFrame(history).to_csv('training_metrics.csv', index=False)
    return model

def predict_and_save(model, test_loader, device, output_file='part1.txt'):
    model.eval()
    predictions = []
    
    print("\nGenerating predictions...")
    with torch.no_grad():
        for texts_a, texts_b in tqdm(test_loader, desc="Predicting"):
            texts_a, texts_b = texts_a.to(device), texts_b.to(device)
            
            output_a = model(texts_a)
            output_b = model(texts_b)
            
            preds = (output_a > output_b).squeeze()
            predictions.extend(['A' if p.item() else 'B' for p in preds])
    
    print(f"Saving predictions to {output_file}")
    with open(output_file, 'w') as f:
        for pred in predictions:
            f.write(f'{pred}\n')

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load and preprocess data
    train_texts_a, train_texts_b = load_and_preprocess_data('train.txt')
    test_texts_a, test_texts_b = load_and_preprocess_data('test.rand.txt')
    
    # Build vocabulary from all training texts
    all_train_texts = train_texts_a + train_texts_b
    vocab = build_vocab(all_train_texts)
    
    # Create datasets
    train_size = int(0.8 * len(train_texts_a))
    train_dataset = TextDataset(train_texts_a[:train_size], train_texts_b[:train_size], vocab)
    val_dataset = TextDataset(train_texts_a[train_size:], train_texts_b[train_size:], vocab)
    test_dataset = TextDataset(test_texts_a, test_texts_b, vocab)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    
    # Initialize model
    model = LSTMClassifier(len(vocab), embedding_dim=100, hidden_dim=128)
    
    # Train model
    model = train_model(model, train_loader, val_loader, device)
    
    # Generate predictions
    predict_and_save(model, test_loader, device)

if __name__ == '__main__':
    main()
```

Task 2:
```python
import random
import string

# Read train.txt with specified encoding
originals = []
corruptions = []
existing_lines_set = set()

with open('train.txt', 'r', encoding='utf-8') as f:
    for line in f:
        line = line.rstrip('\n')  # Remove newline character
        existing_lines_set.add(line)
        parts = line.split('\t')
        if len(parts) >= 2:
            originals.append(parts[0])
            corruptions.append(parts[1])

# Define keyboard layout for typo injection
keyboard = {
    'a': ['q', 'w', 's', 'z'],
    'b': ['v', 'g', 'h', 'n'],
    'c': ['x', 'd', 'f', 'v'],
    'd': ['s', 'e', 'r', 'f', 'c', 'x'],
    'e': ['w', 's', 'd', 'r'],
    'f': ['d', 'r', 't', 'g', 'v', 'c'],
    'g': ['f', 't', 'y', 'h', 'b', 'v'],
    'h': ['g', 'y', 'u', 'j', 'n', 'b'],
    'i': ['u', 'o', 'k', 'j'],
    'j': ['h', 'u', 'i', 'k', 'n', 'm'],
    'k': ['j', 'i', 'o', 'l', 'm'],
    'l': ['k', 'o', 'p'],
    'm': ['n', 'j', 'k', 'l'],
    'n': ['m', 'j', 'h', 'b'],
    'o': ['i', 'k', 'l', 'p', 'u'],
    'p': ['o', 'l'],
    'q': ['w', 'a', 's'],
    'r': ['e', 'd', 'f', 't'],
    's': ['w', 'e', 'd', 'a', 'z', 'x'],
    't': ['r', 'f', 'g', 'y'],
    'u': ['y', 't', 'g', 'h', 'j', 'i', 'o'],
    'v': ['f', 'g', 'b', 'c'],
    'w': ['q', 'a', 's', 'e'],
    'x': ['z', 's', 'd', 'c'],
    'y': ['t', 'g', 'h', 'u'],
    'z': ['a', 's', 'x'],
}

# Define homophones
homophones = {
    'to': 'too',
    'too': 'to',
    'two': 'to',
    'there': 'their',
    'their': 'there',
    'they’re': 'there',
    'its': 'it’s',
    'it’s': 'its',
    'were': 'where',
    'where': 'were',
}

# Define common words for insertion
common_words = ['the', 'and', 'is', 'in', 'that', 'it', 'of', 'a', 'with', 'as', 'for', 'not', 'on', 'be', 'by', 'this', 'are', 'from', 'or', 'an']

# Define corruption functions with required arguments
def typo_injection(sentence, keyboard_layout):
    words = sentence.split()
    word_to_alter = random.choice(words)
    char_to_alter = random.choice(word_to_alter)
    possible_typos = keyboard_layout.get(char_to_alter.lower(), [])
    if possible_typos:
        typo = random.choice(possible_typos)
        altered_word = word_to_alter.replace(char_to_alter, typo, 1)
        new_sentence = sentence.replace(word_to_alter, altered_word, 1)
        return new_sentence
    else:
        return sentence  # No typo possible

def punctuation_error(sentence):
    punctuation = string.punctuation
    if random.choice([True, False]):
        # Remove punctuation
        return sentence.translate(str.maketrans('', '', punctuation))
    else:
        # Add punctuation
        word = random.choice(sentence.split())
        return sentence.replace(word, word + random.choice(['.', ',', '!', '?']))

def word_duplication(sentence):
    words = sentence.split()
    if len(words) < 2:
        return sentence
    duplicated_word = random.choice(words)
    return sentence.replace(duplicated_word, duplicated_word + ' ' + duplicated_word, 1)

def word_omission(sentence):
    words = sentence.split()
    if len(words) < 2:
        return sentence
    omit_word = random.choice(['the', 'and', 'is', 'in', 'of', 'a', 'on', 'at', 'to', 'for'])
    if omit_word in words:
        return ' '.join([word for word in words if word != omit_word])
    else:
        return sentence

def homophone_replacement(sentence, homophones_dict):
    words = sentence.split()
    for i in range(len(words)):
        if words[i] in homophones_dict:
            words[i] = homophones_dict[words[i]]
            break
    return ' '.join(words)

def word_swap(sentence):
    words = sentence.split()
    if len(words) < 2:
        return sentence
    i, j = random.sample(range(len(words)), 2)
    words[i], words[j] = words[j], words[i]
    return ' '.join(words)

def letter_swap_within_word(sentence):
    words = sentence.split()
    if len(words) < 1:
        return sentence
    word_to_alter = random.choice(words)
    if len(word_to_alter) < 2:
        return sentence
    i, j = random.sample(range(len(word_to_alter)), 2)
    letters = list(word_to_alter)
    letters[i], letters[j] = letters[j], letters[i]
    altered_word = ''.join(letters)
    return sentence.replace(word_to_alter, altered_word, 1)

def random_word_insertion(sentence, common_words_list):
    insert_word = random.choice(common_words_list)
    insert_pos = random.randint(0, len(sentence))
    return sentence[:insert_pos] + ' ' + insert_word + ' ' + sentence[insert_pos:]

# Collect all corruption functions with required arguments
corruption_functions = [
    lambda sentence: typo_injection(sentence, keyboard),
    punctuation_error,
    word_duplication,
    word_omission,
    lambda sentence: homophone_replacement(sentence, homophones),
    word_swap,
    letter_swap_within_word,
    lambda sentence: random_word_insertion(sentence, common_words)
]

# Generate new corruptions
new_corruptions = []
for i in range(len(originals)):
    original = originals[i]
    existing_corruption = corruptions[i]
    # Select a random corruption function
    func = random.choice(corruption_functions)
    # Apply the function to generate a new corruption
    new_corruption = func(original)
    # Check uniqueness
    attempts = 0
    while (original + '\t' + new_corruption + '\n' in existing_lines_set or
           new_corruption == existing_corruption or
           new_corruption == original):
        func = random.choice(corruption_functions)
        new_corruption = func(original)
        attempts += 1
        if attempts > 20:  # Increase the number of attempts if necessary
            break
    new_corruptions.append(new_corruption)

# Write to part2.txt
with open('part2.txt', 'w', encoding='utf-8') as f:
    for i in range(len(originals)):
        f.write(originals[i] + '\t' + new_corruptions[i] + '\n')
```