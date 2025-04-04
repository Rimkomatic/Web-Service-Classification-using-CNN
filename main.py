from collections import Counter
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
from transformers import BertTokenizer, BertModel
import gc
import math
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


####################################################
#
#                   LOAD DATA
#
#####################################################


data_dir = "./input/programmable-web/"


def load_csv(file_name):
    file_path = os.path.join(data_dir, file_name)
    return pd.read_csv(file_path)


category = load_csv("category.csv")
api_sketch = load_csv("apisketch.csv")
api_basic = load_csv("apibasic.csv")
api_addition = load_csv("apiaddition.csv")
api_cate = load_csv("apicate.csv")
mashup_sketch = load_csv("mashupsketch.csv")
mashup = load_csv("mashup.csv")
mashup_cate = load_csv("mashupcate.csv")
mashup_api = load_csv("mashupapi.csv")

api_data = pd.merge(api_basic, api_cate, left_on='ID',
                    right_on='ApiID', how='left')
api_data = pd.merge(api_data, category, left_on='CateID',
                    right_on='ID', how='left')
api_data = api_data.rename(
    columns={'Name_x': 'API_Name', 'Name_y': 'Category_Name'})

categories_df = api_data[['Category_Name', 'Amount']].drop_duplicates()
categories_sorted = categories_df.sort_values(by='Amount', ascending=False)


###################################################################
#               CREATE TOP 200 DATA
###################################################################
top200_categories = categories_sorted.head(200)['Category_Name'].tolist()

print("TOP 200 Category: ", top200_categories[:10])

api_data_filtered = api_data[api_data['Category_Name'].isin(
    top200_categories)].copy()

top120_categories = categories_sorted.head(120)['Category_Name'].tolist()
top150_categories = categories_sorted.head(150)['Category_Name'].tolist()
top180_categories = categories_sorted.head(180)['Category_Name'].tolist()

PW120 = api_data_filtered[api_data_filtered['Category_Name'].isin(
    top120_categories)].copy()
PW150 = api_data_filtered[api_data_filtered['Category_Name'].isin(
    top150_categories)].copy()
PW180 = api_data_filtered[api_data_filtered['Category_Name'].isin(
    top180_categories)].copy()

print("Total rows in PW120:", len(PW120))
print("Total rows in PW150:", len(PW150))
print("Total rows in PW180:", len(PW180))


print(PW120.info())

PW120 = PW120[['API_Name', 'Description', 'Category_Name']]
PW120.head()

PW120 = PW120.dropna()
len(PW120['API_Name'].unique()), len(PW120['Category_Name'].unique())

PW120 = PW120.groupby(['API_Name', 'Description'])[
    'Category_Name'].apply(lambda x: ', '.join(x)).reset_index()
PW120.head(15)


#########################################################################
#
#                       DATA PREPARATION
#
#########################################################################

def extract_labels(label_str):
    return [l.strip() for l in label_str.split(',')]


PW120['Label_List'] = PW120['Category_Name'].apply(extract_labels)
PW120['Primary_Label'] = PW120['Label_List'].apply(
    lambda x: x[0] if len(x) > 0 else None)
print("\nSample with Label_List and Primary_Label:")
print(PW120[['API_Name', 'Label_List', 'Primary_Label']].head())

# --- 0C: Build a global mapping of all unique labels ---


def get_all_categories(df, col="Label_List"):
    all_labels = set()
    for labels in df[col]:
        for label in labels:
            all_labels.add(label)
    return sorted(list(all_labels))


all_categories = get_all_categories(PW120, col="Label_List")
cat2idx = {cat: idx for idx, cat in enumerate(all_categories)}
num_categories = len(cat2idx)
print("\nTotal number of unique categories:", num_categories)


# We need every primary label to appear at least 3 times for stratification.
primary_counts = PW120['Primary_Label'].value_counts()
print("\nPrimary label counts before augmentation:")
print(primary_counts)


# Count labels
label_counts = PW120['Primary_Label'].value_counts()

# Identify underrepresented labels (less than 3)
underrepresented = label_counts[label_counts < 5]

# List to hold all duplicated rows
duplicates = []

for label, count in underrepresented.items():
    rows = PW120[PW120['Primary_Label'] == label]
    n_to_add = 5 - count
    dup = rows.sample(n=n_to_add, replace=True, random_state=42)
    duplicates.append(dup)

# Concatenate original and duplicated data
PW120_augmented = pd.concat([PW120, *duplicates], ignore_index=True)

# Ensure all classes have at least 5 by final padding
final_counts = PW120_augmented['Primary_Label'].value_counts()
final_underrep = final_counts[final_counts < 5]

extra_dupes = []
for label, count in final_underrep.items():
    rows = PW120_augmented[PW120_augmented['Primary_Label'] == label]
    n_to_add = 5 - count
    dup = rows.sample(n=n_to_add, replace=True, random_state=42)
    extra_dupes.append(dup)

PW120_augmented = pd.concat([PW120_augmented, *extra_dupes], ignore_index=True)

print("\nPrimary label counts after augmentation:")
print(PW120_augmented['Primary_Label'].value_counts())

PW120.head()


# ----------------- Clean Data --------------------------
def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = text.replace("\n", " ")  # Remove newlines
    text = text.replace("\r", " ")  # Remove carriage returns
    return text


PW120_augmented['Description'] = PW120_augmented['Description'].apply(
    clean_text)


PW120_augmented.head()

PW120_augmented.to_csv("cleaned_api_data.csv", index=False)
print("Data preprocessing complete. Cleaned data saved as 'cleaned_api_data.csv'.")

PW120_train, temp_df = train_test_split(
    PW120_augmented, test_size=0.4, random_state=42, stratify=PW120_augmented["Primary_Label"])
PW120_val, PW120_test = train_test_split(
    temp_df, test_size=0.5, random_state=42, stratify=temp_df["Primary_Label"])
print("\nSplit sizes:")
print("Train:", len(PW120_train), "Val:", len(
    PW120_val), "Test:", len(PW120_test))


#########################################################################
#
#           Category Ateentive Deep Service Feature Extraction
#
##########################################################################


L_desc = 65          # Fixed length for service descriptions.
L_name = 10          # Fixed length for service names.
cate_length = 15      # Fixed length for tokenizing category names.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load BERT and tokenizer for feature extraction.
tokenizer_extr = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model_extr = BertModel.from_pretrained("bert-base-uncased")
bert_model_extr.to(device)
bert_model_extr.eval()
d = bert_model_extr.config.hidden_size  # typically 768


def get_embeddings_batch(text_list, fixed_length):
    encoding = tokenizer_extr(text_list, padding="max_length", truncation=True,
                              max_length=fixed_length, return_tensors="pt")
    for key in encoding:
        encoding[key] = encoding[key].to(device)
    with torch.no_grad():
        outputs = bert_model_extr(**encoding)
    return outputs.last_hidden_state  # shape: [B, fixed_length, d]


def embed_category(text):
    emb = get_embeddings_batch([text], cate_length)  # [1, cate_length, d]
    return emb.squeeze(0).mean(dim=0)


unique_cats = PW120_augmented['Category_Name'].unique()
category_embeddings = {cat: embed_category(cat) for cat in unique_cats}
W_l = torch.stack(list(category_embeddings.values())).to(device)


def apply_attention_batch(S, L):
    scaling = d
    CM = torch.matmul(S, L.transpose(0, 1)) / \
        scaling  # [B, seq_length, n_unique]
    CV, _ = torch.max(CM, dim=2)                       # [B, seq_length]
    SA = F.softmax(CV, dim=1)                           # [B, seq_length]
    S_a = SA.unsqueeze(2) * S                          # [B, seq_length, d]
    return S_a


filter_sizes = [2, 3, 4]
num_filters = 100
total_filters = num_filters * len(filter_sizes)


##############################################################################
#                               TEXT CNN
##############################################################################

class TextCNN(nn.Module):
    def __init__(self, d, filter_sizes, num_filters):
        super(TextCNN, self).__init__()
        self.convs = nn.ModuleList([nn.Conv1d(in_channels=d, out_channels=num_filters, kernel_size=fs)
                                    for fs in filter_sizes])

    def forward(self, x):
        conv_results = [F.relu(conv(x)) for conv in self.convs]
        pooled = [F.max_pool1d(result, kernel_size=result.shape[2]).squeeze(2)
                  for result in conv_results]
        return torch.cat(pooled, dim=1)  # [B, total_filters]


text_cnn = TextCNN(d, filter_sizes, num_filters).to(device)
text_cnn.eval()

# Projection for service name features
linear_proj = nn.Linear(d, total_filters).to(device)
linear_proj.eval()
r_n = 0.5


for i in range(num_batches_extr):
    batch_df = PW120_augmented.iloc[i*batch_size_extr: (i+1)*batch_size_extr]
    batch_descriptions = batch_df["Description"].tolist()
    batch_names = batch_df["API_Name"].tolist()

    W_d_batch = get_embeddings_batch(
        batch_descriptions, L_desc)  # [B, L_desc, d]
    W_n_batch = get_embeddings_batch(
        batch_names, L_name)         # [B, L_name, d]

    S_d_a_batch = apply_attention_batch(W_d_batch, W_l)  # [B, L_desc, d]
    S_n_a_batch = apply_attention_batch(W_n_batch, W_l)  # [B, L_name, d]

    x_descr_batch = S_d_a_batch.transpose(1, 2)          # [B, d, L_desc]
    f_d_batch = text_cnn(x_descr_batch)                    # [B, total_filters]

    f_n_batch, _ = torch.max(S_n_a_batch, dim=1)           # [B, d]
    f_n_proj_batch = linear_proj(f_n_batch)                # [B, total_filters]

    V_batch = f_d_batch + r_n * f_n_proj_batch             # [B, total_filters]
    final_features_list.append(V_batch.cpu())
final_features_all = torch.cat(
    final_features_list, dim=0).detach().numpy().tolist()
PW120_augmented["Final_V"] = final_features_all
print("\nFeature extraction complete. Sample Final_V:")
print(PW120_augmented[["API_Name", "Final_V"]].head())

display(PW120_augmented.head())


###############################################################################
#                           CUSTOM DATASET
###############################################################################
class PW120Dataset(Dataset):
    def __init__(self, dataframe, cat2idx):
        self.df = dataframe.reset_index(drop=True)
        self.cat2idx = cat2idx

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        desc = row["Description"]
        name = row["API_Name"]
        if isinstance(row["Category_Name"], str):
            label_list = [l.strip() for l in row["Category_Name"].split(',')]
        else:
            label_list = list(row["Category_Name"])
        primary_label = label_list[0]
        primary_idx = self.cat2idx[primary_label]
        label_list_idx = [self.cat2idx[l]
                          for l in label_list if l in self.cat2idx]
        return desc, name, primary_idx, label_list_idx


def custom_collate(batch):
    # batch is a list of tuples: (desc, name, primary_idx, label_list_idx)
    desc, name, primary, label_list = zip(*batch)
    primary_tensor = torch.tensor(primary, dtype=torch.long)
    return list(desc), list(name), primary_tensor, list(label_list)


PW120_split = PW120_augmented.copy()
PW120_train, temp_split = train_test_split(
    PW120_split, test_size=0.4, random_state=42, stratify=PW120_split["Primary_Label"])
PW120_val, PW120_test = train_test_split(
    temp_split, test_size=0.5, random_state=42, stratify=temp_split["Primary_Label"])
print("\nDataset splits:")
print("Train:", len(PW120_train), "Val:", len(
    PW120_val), "Test:", len(PW120_test))

train_dataset = PW120Dataset(PW120_train, cat2idx)
val_dataset = PW120Dataset(PW120_val, cat2idx)
test_dataset = PW120Dataset(PW120_test, cat2idx)

batch_size_train = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size_train,
                          shuffle=True, num_workers=2, collate_fn=custom_collate)
val_loader = DataLoader(val_dataset, batch_size=batch_size_train,
                        shuffle=False, num_workers=2, collate_fn=custom_collate)
test_loader = DataLoader(test_dataset, batch_size=batch_size_train,
                         shuffle=False, num_workers=2, collate_fn=custom_collate)


#####################################################################################################
#
#                                       LOSS FUNCTION
#
#####################################################################################################

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class LargeMarginSoftmaxLoss(nn.Module):
    def __init__(self, margin=0.35, scale=30, reduction='mean'):
        super(LargeMarginSoftmaxLoss, self).__init__()
        self.margin = margin
        self.scale = scale
        self.reduction = reduction

    def forward(self, logits, targets):
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, targets.view(-1, 1), 1)
        logits_adjusted = logits - one_hot * self.margin
        logits_scaled = self.scale * logits_adjusted
        loss = F.cross_entropy(logits_scaled, targets,
                               reduction=self.reduction)
        return loss


######################################################################################
#
#                       MODEL DEFINATIONS
#
######################################################################################
class LACNN(nn.Module):
    def __init__(self, embed_dim, num_filters, kernel_sizes, dropout=0.5, sn_rate=0.5):
        super(LACNN, self).__init__()
        self.convs = nn.ModuleList([nn.Conv1d(in_channels=embed_dim, out_channels=num_filters, kernel_size=k)
                                    for k in kernel_sizes])
        self.dropout = nn.Dropout(dropout)
        self.sn_rate = sn_rate

    def forward(self, desc_emb, name_emb):
        x = desc_emb.transpose(1, 2)
        conv_results = [F.relu(conv(x)) for conv in self.convs]
        pool_results = [F.max_pool1d(cr, kernel_size=cr.size(
            2)).squeeze(2) for cr in conv_results]
        desc_feature = torch.cat(pool_results, dim=1)
        name_feature = torch.max(name_emb, dim=1)[0]
        return desc_feature, name_feature


class SFA(nn.Module):
    def __init__(self, feature_dim, num_categories):
        super(SFA, self).__init__()
        self.fc_relevance = nn.Linear(feature_dim, num_categories)
        self.fc_ht = nn.Linear(feature_dim, feature_dim)
        self.central_features = nn.Parameter(
            torch.randn(num_categories, feature_dim))

    def forward(self, V):
        beta = F.softmax(self.fc_relevance(V), dim=1)
        m1 = torch.matmul(beta, self.central_features)
        p = torch.tanh(self.fc_ht(V))
        m2 = p * m1
        F_long = V + m2
        return F_long


class DeepLTSC(nn.Module):
    def __init__(self, num_categories, embed_dim=768, num_filters=100, kernel_sizes=[3, 4, 5],
                 dropout=0.5, sn_rate=0.5, use_sfa=True):
        super(DeepLTSC, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.embed_dim = embed_dim
        self.use_sfa = use_sfa
        self.category_embeddings = nn.Embedding(num_categories, embed_dim)
        self.lacnn = LACNN(embed_dim, num_filters,
                           kernel_sizes, dropout, sn_rate)
        fusion_input_dim = num_filters * len(kernel_sizes) + embed_dim
        self.fc_fusion = nn.Linear(
            fusion_input_dim, num_filters * len(kernel_sizes))
        self.dropout = nn.Dropout(dropout)
        if use_sfa:
            self.sfa = SFA(num_filters * len(kernel_sizes), num_categories)
        self.classifier = nn.Linear(
            num_filters * len(kernel_sizes), num_categories)

    def apply_label_attention(self, embeddings, category_idx):
        cat_emb = self.category_embeddings(category_idx)
        cat_emb = cat_emb.unsqueeze(1)
        attn_scores = torch.bmm(embeddings, cat_emb.transpose(1, 2)).squeeze(2)
        attn_weights = F.softmax(attn_scores, dim=1).unsqueeze(2)
        attended = embeddings * attn_weights
        return attended

    def forward(self, desc_text, name_text, category_idx):
        desc_inputs = self.tokenizer(
            desc_text, return_tensors="pt", padding=True, truncation=True)
        name_inputs = self.tokenizer(
            name_text, return_tensors="pt", padding=True, truncation=True)
        device = next(self.parameters()).device
        desc_inputs = {k: v.to(device) for k, v in desc_inputs.items()}
        name_inputs = {k: v.to(device) for k, v in name_inputs.items()}
        desc_outputs = self.bert(**desc_inputs)
        name_outputs = self.bert(**name_inputs)
        desc_embeddings = desc_outputs.last_hidden_state
        name_embeddings = name_outputs.last_hidden_state
        desc_attended = self.apply_label_attention(
            desc_embeddings, category_idx)
        name_attended = self.apply_label_attention(
            name_embeddings, category_idx)
        desc_feature, name_feature = self.lacnn(desc_attended, name_attended)
        combined = torch.cat([desc_feature, name_feature], dim=1)
        combined = self.fc_fusion(combined)
        combined = self.dropout(combined)
        V = combined
        F_long = self.sfa(V) if self.use_sfa else V
        logits = self.classifier(F_long)
        return logits


############################################################################################################
#
#                                   MODEL TRAIN AND EVALUATION
#
############################################################################################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DeepLTSC(num_categories=num_categories, embed_dim=768, num_filters=100,
                 kernel_sizes=[3, 4, 5], dropout=0.5, sn_rate=0.5, use_sfa=True)
model.to(device)
model.train()

focal_loss_fn = FocalLoss(gamma=0.5, alpha=0.7)
lm_loss_fn = LargeMarginSoftmaxLoss(margin=0.35, scale=30)
lambda_loss = 1.0

optimizer = optim.Adam(model.parameters(), lr=2e-5)
num_epochs = 3


def evaluate(model, dataloader):
    model.eval()
    total = 0
    correct = 0
    preds_all = []
    primary_true = []
    running_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            desc_text, name_text, primary_labels, label_list = batch
            primary_labels = primary_labels.to(device)
            logits = model(desc_text, name_text, primary_labels)
            loss_focal = focal_loss_fn(logits, primary_labels)
            loss_lm = lm_loss_fn(logits, primary_labels)
            loss = loss_focal + lambda_loss * loss_lm
            running_loss += loss.item()
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            for i, p in enumerate(preds):
                total += 1
                if p in label_list[i]:
                    correct += 1
                preds_all.append(p)
                primary_true.append(primary_labels[i].item())
    avg_loss = running_loss / len(dataloader)
    acc = correct / total
    macro_f1 = f1_score(primary_true, preds_all, average='macro')
    model.train()
    return avg_loss, acc, macro_f1


print("\nStarting training...\n")
for epoch in range(num_epochs):
    running_loss = 0.0
    for batch in train_loader:
        desc_text, name_text, primary_labels, _ = batch
        primary_labels = primary_labels.to(device)
        optimizer.zero_grad()
        logits = model(desc_text, name_text, primary_labels)
        loss_focal = focal_loss_fn(logits, primary_labels)
        loss_lm = lm_loss_fn(logits, primary_labels)
        loss = loss_focal + lambda_loss * loss_lm
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    avg_train_loss = running_loss / len(train_loader)
    val_loss, val_acc, val_f1 = evaluate(model, val_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {avg_train_loss:.4f} | "
          f"Val Loss: {val_loss:.4f} | Val Accuracy (in set): {val_acc:.4f} | Val Macro F1: {val_f1:.4f}")

print("\nTraining finished!")
test_loss, test_acc, test_f1 = evaluate(model, test_loader)
print(f"\nTest Loss: {test_loss:.4f} | Test Accuracy (in set): {
      test_acc:.4f} | Test Macro F1 (primary label): {test_f1:.4f}")


primary_counts = PW120_augmented['Primary_Label'].value_counts()
print("Primary label counts:\n", primary_counts)

top5_labels = primary_counts.head(5).index.tolist()
print("Top 5 labels:", top15_labels)

# Filter PW120_augmented to only include samples with Primary_Label among top15.
PW5 = PW120_augmented[PW120_augmented['Primary_Label'].isin(
    top5_labels)].copy()
print("Number of samples in top 5 classes:", len(PW5))

# Extract the deep feature vectors from Final_V.
features = PW5["Final_V"].tolist()
X = np.array(features)  # X shape: [n_samples, feature_dim]

# Create a new mapping for the top-15 classes.
# Here, we can simply assign an index based on the sorted order (or by frequency).
top5_cat2idx = {cat: idx for idx, cat in enumerate(sorted(top5_labels))}
print("Top5 cat2idx mapping:", top5_cat2idx)

# Map the primary labels of PW15 to integer indices.
labels = PW5["Primary_Label"].tolist()
colors = [top5_cat2idx[label] for label in labels]

# --- Step 3: Run t-SNE ---
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)

plt.figure(figsize=(10, 8))

# Reduce scatter point size (s) and use a better colormap
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1],
                      c=colors, cmap="Set1", s=2, alpha=0.8)

# Improve colorbar readability
cbar = plt.colorbar(scatter, ticks=range(len(top15_cat2idx)))
cbar.set_label("Top 15 Category Index")
cbar.set_ticks(list(top15_cat2idx.values()))

plt.title("t-SNE Scatter Plot of Web Service Features (Top 15 Classes)")
plt.show()
