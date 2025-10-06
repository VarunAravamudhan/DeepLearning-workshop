# DeepLearning-workshop


## Aim
To implement and train a Neural Network model using PyTorch for the classification of income levels (`>50K` or `<=50K`) from a census dataset.  
This project demonstrates an end-to-end workflow covering data preprocessing, feature encoding, model design, training, evaluation, and single-instance inference.

---

## Problem Statement
Develop a deep learning model that predicts whether an individual's annual income exceeds $50,000 based on demographic and work-related attributes.  
The dataset contains both categorical and continuous features which must be processed appropriately before model training.

---

## Software Requirements
- Python 3.8 or higher  
- PyTorch (CPU or GPU version)  
- NumPy, Pandas, scikit-learn  
- Matplotlib or Seaborn (optional for visualization)  
- Jupyter Notebook or any IDE (VS Code, PyCharm)

---

## Hardware Requirements
- Minimum 8 GB RAM  
- (Optional) NVIDIA GPU with CUDA support

---

## Algorithm / Workflow
1. Import Libraries  
2. Load and preprocess dataset  
3. Encode categorical and continuous features  
4. Convert data into PyTorch tensors  
5. Build neural network model with embeddings and dense layers  
6. Train model using CrossEntropyLoss and Adam optimizer  
7. Evaluate and save the model  
8. Perform inference on new data samples

---

## Program Code

```python
import os
import random
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RANDOM_SEED = 42
TRAIN_SIZE = 25000
TEST_SIZE = 5000
BATCH_SIZE = 512
EPOCHS = 300
LR = 0.001
DROPOUT_P = 0.4
HIDDEN_UNITS = 50

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_income_csv(path: str = "income.csv") -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Please place income.csv in the working directory.")
    df = pd.read_csv(path)
    return df

def preprocess_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str], str, Dict[str, List[str]]]:
    label_col = None
    for cand in ["income", "Income", "target", "label", "SalStat"]:
        if cand in df.columns:
            label_col = cand
            break
    if label_col is None:
        raise ValueError("Label column not found. Make sure a suitable label column ('income', 'Income', 'target', 'label', or 'SalStat') exists in the CSV.")

    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if label_col in categorical_cols:
        categorical_cols.remove(label_col)

    continuous_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    if label_col in continuous_cols:
        continuous_cols.remove(label_col)

    for c in categorical_cols + [label_col]:
        if c in df.columns and df[c].dtype == object:
            df[c] = df[c].str.strip()

    df = df.copy()
    df[label_col] = df[label_col].map(lambda x: 1 if str(x).strip().startswith('>') else 0)

    category_mapping = {}
    for col in categorical_cols:
        df[col] = pd.Categorical(df[col])
        category_mapping[col] = list(df[col].cat.categories)
        df[col] = df[col].cat.codes.astype('int64')

    before = len(df)
    df = df.dropna().reset_index(drop=True)
    after = len(df)
    if after < before:
        print(f"Dropped {before - after} rows containing NaNs during preprocessing.")

    return df, categorical_cols, continuous_cols, label_col, category_mapping

class CensusTabularDataset(Dataset):
    def __init__(self, cat_tensor: torch.LongTensor, cont_tensor: torch.FloatTensor, labels: torch.LongTensor):
        self.cat = cat_tensor
        self.cont = cont_tensor
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.cat[idx], self.cont[idx], self.labels[idx]

class TabularModel(nn.Module):
    def __init__(self, categorical_cardinalities: List[int], continuous_size: int,
                 emb_drop_p: float = 0.0, hidden_units: int = 50, dropout_p: float = 0.4):
        super().__init__()
        self.embeddings = nn.ModuleList()
        self.embedding_output_dim = 0
        for card in categorical_cardinalities:
            emb_size = min(50, (card + 1) // 2)
            self.embeddings.append(nn.Embedding(card, emb_size))
            self.embedding_output_dim += emb_size

        self.bn_cont = nn.BatchNorm1d(continuous_size) if continuous_size > 0 else None
        input_dim = self.embedding_output_dim + (continuous_size if continuous_size > 0 else 0)

        self.fc1 = nn.Linear(input_dim, hidden_units)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_p)
        self.output = nn.Linear(hidden_units, 2)

    def forward(self, x_cat, x_cont):
        if len(self.embeddings) > 0:
            emb_outs = []
            for i, emb in enumerate(self.embeddings):
                emb_outs.append(emb(x_cat[:, i]))
            x = torch.cat(emb_outs, 1)
        else:
            x = torch.tensor([], device=x_cont.device)

        if x_cont is not None and x_cont.shape[1] > 0:
            if self.bn_cont is not None:
                cont = self.bn_cont(x_cont)
            else:
                cont = x_cont
            if x.numel() > 0:
                x = torch.cat([x, cont], 1)
            else:
                x = cont

        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.output(x)
        return x

def prepare_tensors(df: pd.DataFrame, categorical_cols: List[str], continuous_cols: List[str], label_col: str,
                    scaler: StandardScaler = None) -> Tuple[torch.LongTensor, torch.FloatTensor, torch.LongTensor, StandardScaler]:

    cat_arr = df[categorical_cols].values.astype('int64') if len(categorical_cols) > 0 else np.zeros((len(df), 0), dtype='int64')
    cat_tensor = torch.from_numpy(cat_arr).long()

    cont_arr = df[continuous_cols].values.astype('float32') if len(continuous_cols) > 0 else np.zeros((len(df), 0), dtype='float32')
    if scaler is None and len(continuous_cols) > 0:
        scaler = StandardScaler()
        cont_arr = scaler.fit_transform(cont_arr)
    elif len(continuous_cols) > 0:
        cont_arr = scaler.transform(cont_arr)
    cont_tensor = torch.from_numpy(cont_arr.astype('float32'))

    labels = torch.from_numpy(df[label_col].values.astype('int64'))

    return cat_tensor, cont_tensor, labels, scaler

def train_epoch(model: nn.Module, dataloader: DataLoader, criterion, optimizer) -> float:
    model.train()
    running_loss = 0.0
    for x_cat, x_cont, y in dataloader:
        x_cat = x_cat.to(DEVICE)
        x_cont = x_cont.to(DEVICE)
        y = y.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(x_cat, x_cont)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * x_cat.size(0)
    return running_loss / len(dataloader.dataset)

def eval_model(model: nn.Module, dataloader: DataLoader, criterion) -> Tuple[float, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for x_cat, x_cont, y in dataloader:
            x_cat = x_cat.to(DEVICE)
            x_cont = x_cont.to(DEVICE)
            y = y.to(DEVICE)
            outputs = model(x_cat, x_cont)
            loss = criterion(outputs, y)
            running_loss += loss.item() * x_cat.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return running_loss / len(dataloader.dataset), correct / total

def predict_instance(model: nn.Module, instance: Dict[str, Any], categorical_cols: List[str], continuous_cols: List[str],
                     category_mapping: Dict[str, List[str]], scaler: StandardScaler) -> str:
    row = {}
    for c in categorical_cols + continuous_cols:
        if c in instance:
            row[c] = instance[c]
        else:
            raise ValueError(f"Missing column {c} in the instance. Provide all categorical and continuous columns.")
    df_row = pd.DataFrame([row])

    for col in categorical_cols:
        cats = category_mapping[col]
        try:
            code = cats.index(df_row.loc[0, col])
        except ValueError:
            code = 0
        df_row[col] = code

    if len(continuous_cols) > 0:
        cont = scaler.transform(df_row[continuous_cols].values.astype('float32'))
        cont_tensor = torch.from_numpy(cont.astype('float32')).to(DEVICE)
    else:
        cont_tensor = torch.zeros((1, 0)).to(DEVICE)

    if len(categorical_cols) > 0:
        cat_arr = df_row[categorical_cols].values.astype('int64')
        cat_tensor = torch.from_numpy(cat_arr).long().to(DEVICE)
    else:
        cat_tensor = torch.zeros((1, 0)).long().to(DEVICE)

    model.eval()
    with torch.no_grad():
        out = model(cat_tensor, cont_tensor)
        pred = torch.argmax(out, dim=1).item()
    return ">50K" if pred == 1 else "<=50K"

def main():
    df = load_income_csv("income.csv")
    print("Loaded dataframe shape:", df.shape)
    print("Columns in the DataFrame:", df.columns.tolist())
    df_proc, categorical_cols, continuous_cols, label_col, category_mapping = preprocess_dataframe(df)
    print("Categorical cols:", categorical_cols)
    print("Continuous cols:", continuous_cols)
    print("Label col:", label_col)
    if len(df_proc) < TRAIN_SIZE + TEST_SIZE:
        raise ValueError(f"Dataset has only {len(df_proc)} rows after cleaning but requires {TRAIN_SIZE+TEST_SIZE} rows.")

    df_proc = df_proc.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    df_train = df_proc.iloc[:TRAIN_SIZE].reset_index(drop=True)
    df_test = df_proc.iloc[TRAIN_SIZE:TRAIN_SIZE+TEST_SIZE].reset_index(drop=True)

    cat_train, cont_train, y_train, scaler = prepare_tensors(df_train, categorical_cols, continuous_cols, label_col, scaler=None)
    cat_test, cont_test, y_test, _ = prepare_tensors(df_test, categorical_cols, continuous_cols, label_col, scaler=scaler)

    train_dataset = CensusTabularDataset(cat_train, cont_train, y_train)
    test_dataset = CensusTabularDataset(cat_test, cont_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    categorical_cardinalities = [len(category_mapping[c]) for c in categorical_cols]
    model = TabularModel(categorical_cardinalities, len(continuous_cols), emb_drop_p=0.0, hidden_units=HIDDEN_UNITS, dropout_p=DROPOUT_P)
    model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_test_acc = 0.0
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer)
        test_loss, test_acc = eval_model(model, test_loader, criterion)
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | Test Acc: {test_acc*100:.2f}%")
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save({'model_state_dict': model.state_dict(), 'scaler': scaler, 'categorical_cols': categorical_cols, 'continuous_cols': continuous_cols, 'category_mapping': category_mapping}, 'best_tabular_model.pth')

    test_loss, test_acc = eval_model(model, test_loader, criterion)
    print("\\nTraining complete.")
    print(f"Final Test Loss: {test_loss:.4f}")
    print(f"Final Test Accuracy: {test_acc*100:.2f}%")

    sample_row = df_test.iloc[0]
    sample_input = {}
    for c in categorical_cols:
        cats = category_mapping[c]
        code = int(sample_row[c])
        val = cats[code] if 0 <= code < len(cats) else cats[0]
        sample_input[c] = val
    for c in continuous_cols:
        sample_input[c] = float(sample_row[c])

    print('\\nSample input (human readable):')
    print(sample_input)
    pred_label = predict_instance(model, sample_input, categorical_cols, continuous_cols, category_mapping, scaler)
    print('Model prediction for sample:', pred_label)

if __name__ == '__main__':
    main()
```
## OUTPUT
### DATA LOADING
<img width="456" height="81" alt="image" src="https://github.com/user-attachments/assets/42d4b25e-5813-4411-9d0e-e3b0379d3f54" />

<img width="556" height="112" alt="image" src="https://github.com/user-attachments/assets/bddfde5b-9247-4024-9901-3e45fcc34079" />

### DATA PREPROCESSING
<img width="1288" height="53" alt="image" src="https://github.com/user-attachments/assets/438249d0-5b53-46e6-ba4e-d9af36e52a55" />

### EVALUATION

<img width="185" height="67" alt="image" src="https://github.com/user-attachments/assets/41425bdd-bc01-4aa2-a696-10318f409965" />

### SAMPLE PREDICTION
<img width="1553" height="53" alt="image" src="https://github.com/user-attachments/assets/1725446c-1dff-40d5-a964-6bd9622a3653" />

## RESULT
The model successfully trained and achieved good classification accuracy on the test dataset.




