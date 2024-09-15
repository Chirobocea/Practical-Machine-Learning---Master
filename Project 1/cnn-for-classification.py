import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, Dataset
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from sklearn.utils.class_weight import compute_class_weight
from transformers import RobertaTokenizer
import wandb
import random

from kaggle_secrets import UserSecretsClient

# set your own Weights & Biases api key into kaggle
user_secrets = UserSecretsClient()
wandb_api = user_secrets.get_secret("wandb_api") 
wandb.login(key=wandb_api)

# Set the project and the hyperparameters
wandb.init(entity="mihail-chirobocea", project="cls", name="cnn9_cap20000")
wandb.config.epochs = 25
wandb.config.batch_size = 64
wandb.config.learning_rate = 0.01
wandb.config.embedding_dim = 128
wandb.config.num_classes = 4

# This function keeps the first 20 000 samples from a given class
def filter_dicts(json_list, target_label, keep_percentage): # keep_percentage is not used anymore
    dicts_with_target_label = [d for d in json_list if d.get("label") == target_label]
    dicts_with_other_labels = [d for d in json_list if d.get("label") != target_label]

    filtered_dicts_target_label = dicts_with_target_label[:20000]

    filtered_result = filtered_dicts_target_label + dicts_with_other_labels
    
    return filtered_result

# Load the data
with open('/kaggle/input/sentence-pair/train.json', 'r') as file:
    train_data = json.load(file)

with open('/kaggle/input/sentence-pair/validation.json', 'r') as file:
    valid_data = json.load(file)

# We keep only a part of this classes in order to get a slightly better balance
train_data = filter_dicts(json_list=train_data, target_label=2, keep_percentage=50)
train_data = filter_dicts(json_list=train_data, target_label=3, keep_percentage=50)
    
# We add a "[SEP]" between the two pairs of sentence in order to be easier to the model to differentiate them.
train_texts = [f"{item['sentence1']} [SEP] {item['sentence2']}" for item in train_data]
valid_texts = [f"{item['sentence1']} [SEP] {item['sentence2']}" for item in valid_data]

# Converting the arrays to numpy arrays
train_labels = np.array([item['label'] for item in train_data])
valid_labels = np.array([item['label'] for item in valid_data])

# Loading the trained tokenizer
tokenizer = RobertaTokenizer.from_pretrained('/kaggle/input/mytokenizer15k')

# Encoding data into tensors
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512, return_tensors='pt')
valid_encodings = tokenizer(valid_texts, truncation=True, padding=True, max_length=512, return_tensors='pt')

# Creating datasets with encodings, masks and labels
train_dataset = TensorDataset(
    train_encodings['input_ids'], 
    train_encodings['attention_mask'], 
    torch.tensor(train_labels)
)

valid_dataset = TensorDataset(
    valid_encodings['input_ids'], 
    valid_encodings['attention_mask'], 
    torch.tensor(valid_labels)
)

# Creating dataloaders based on wandb batch size
train_loader = DataLoader(train_dataset, batch_size=wandb.config.batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=wandb.config.batch_size, shuffle=False)

# Defining the residual blocks of the CNN
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)  #convolution
        self.bn1 = nn.BatchNorm1d(out_channels) # normalization
        self.silu = nn.SiLU()                   # activation
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.shortcut = nn.Sequential()
        # if the tensors match after the first convolutions, we apply a skip connection
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.silu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(residual) # here we apply the skip connection
        out = self.silu(out)
        return out
    

class ResNet18(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_classes, base_channels=64):
        super(ResNet18, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim) # we define an embedding to better represent each word
        self.conv1 = nn.Conv1d(embedding_dim, base_channels, kernel_size=7, stride=2, padding=3) # we use slightly bigger kernel for first conv in order to have larger context span
        self.bn1 = nn.BatchNorm1d(base_channels)
        self.silu = nn.SiLU()  
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)     # poolong

        self.layer1 = self.make_layer(base_channels, base_channels, 2)
        self.layer2 = self.make_layer(base_channels, base_channels * 2, 2, stride=2)        # pooling
        self.layer3 = self.make_layer(base_channels * 2, base_channels * 4, 2, stride=2)    # pooling
        self.layer4 = self.make_layer(base_channels * 4, base_channels * 8, 2, stride=2)    # pooling

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(base_channels * 8, num_classes)

    def make_layer(self, in_channels, out_channels, num_blocks, stride=1):
        # return num_blocks residual blocks
        layers = []
        # only the first layer has stride (pooling)
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def encode(self, x):
        x = self.embedding(x)
        x = x.permute(0, 2, 1)  # permute the axies to math for convolutions
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.silu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x
    
    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0, 2, 1) 
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.silu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# set some parameters
vocab_size = len(tokenizer.get_vocab())
embedding_dim = wandb.config.embedding_dim
num_classes = wandb.config.num_classes
model = ResNet18(vocab_size, embedding_dim, num_classes) # definign the model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # define the device 

class_labels = np.unique(train_labels)
class_weights = compute_class_weight(class_weight='balanced', classes=class_labels, y=train_labels) # calculate class weights in order to fix the inbalance of the number of samples from each calss
class_weights = torch.tensor(class_weights, dtype=torch.float32)
class_weights = class_weights.to(device) # move tensors to device

criterion = nn.CrossEntropyLoss(weight = class_weights) # define the cross entropy loss with given weights
optimizer = optim.AdamW(model.parameters(), lr=wandb.config.learning_rate) # defien adam weighted optimizer

model.to(device) # move model to device

best_macro_f1 = 0.0
best_model_path = "best_model.pth"

def calculate_f1(predictions, labels):
    f1 = f1_score(labels.cpu().numpy(), predictions.cpu().numpy(), average='macro') # calculate macro f1 score
    return f1

wandb_step = 0
total_steps = len(train_loader) * wandb.config.epochs

for epoch in range(wandb.config.epochs): # iterate trough each epoch
    model.train() # put model in training mode
    for batch in train_loader: # iterate trough batch
        input_ids, attention_mask, labels = batch 
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device) #  put tensors to device

        optimizer.zero_grad() # remove gradients 
        output = model(input_ids) # forward pass
        loss = criterion(output, labels) # calculate loss
        loss.backward() # calculate gradients
        optimizer.step() # propagate gradients

        if wandb_step % 10 == 0:
            wandb.log({"Train Loss": loss.item()}, step=wandb_step) # add loss metrics to wandb

        wandb_step += 1 # increase steps for wandb

        current_lr = wandb.config.learning_rate * (1.0 - wandb_step / total_steps) # decrease learning rate after each step
     
        if wandb_step % 10 == 0:
            wandb.log({"Learning Rate": current_lr}, step=wandb_step) # add learning rate to wandb
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr # set the new learning rate tot the optimizer

    model.eval() # put model in evaluation mode
    all_predictions = []
    all_labels = []

    with torch.no_grad(): # make sure that gradients are not computed
        for batch in valid_loader: # iterate trough valdiation
            input_ids, attention_mask, labels = batch
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

            output = model(input_ids)
            predictions = torch.argmax(output, dim=1) # take the prediction base on the highest probabilitie (argmax)

            # move predictions and grand truth to cpu
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    macro_f1 = calculate_f1(torch.tensor(all_predictions), torch.tensor(all_labels))
    wandb.log({"Epoch": epoch + 1, "Macro F1": macro_f1, "Learning Rate": current_lr}) # add metrics to wandb for validation after each epoch

    print(f"Epoch {epoch + 1}/{wandb.config.epochs}, Macro F1 Score: {macro_f1}, Learning Rate: {current_lr}")

    # keep track of better models, absed on valdiation and save them
    if macro_f1 > best_macro_f1:
        best_macro_f1 = macro_f1
        torch.save(model.state_dict(), best_model_path)
        print("Best model saved!")

# reload the best model
best_model = ResNet18(vocab_size, embedding_dim, num_classes)
best_model.load_state_dict(torch.load(best_model_path))
best_model.to(device)

best_model.eval()
all_predictions = []
all_labels = []

with torch.no_grad():
    for batch in valid_loader:
        input_ids, attention_mask, labels = batch
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

        output = best_model(input_ids)
        predictions = torch.argmax(output, dim=1)

        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())


# make a classification report (more metrics)
final_classification_report = classification_report(all_labels, all_predictions)
print(final_classification_report)
wandb.finish() 


import csv
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, Dataset
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from sklearn.utils.class_weight import compute_class_weight
from transformers import RobertaTokenizer
import wandb

with open('/kaggle/input/sentence-pair/test.json', 'r') as file:
    test_data = json.load(file)
    

test_texts = [f"{item['sentence1']} [SEP] {item['sentence2']}" for item in test_data]

# load the guids
test_guid = np.array([item['guid'] for item in test_data])

tokenizer = RobertaTokenizer.from_pretrained('/kaggle/input/mytokenizer15k')

test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=512, return_tensors='pt')

test_dataset = TensorDataset(
    test_encodings['input_ids'], 
    test_encodings['attention_mask']
)

test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

vocab_size = len(tokenizer.get_vocab())
embedding_dim = 128
num_classes = 4
model = ResNet18(vocab_size, embedding_dim, num_classes)
model.load_state_dict(torch.load('/kaggle/input/cnn-v8/best_model.pth'))
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
all_predictions = []
# inference for test data
with torch.no_grad():
    for batch in test_loader:
        input_ids, attention_mask = batch
        input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)

        output = model(input_ids)
        predictions = torch.argmax(output, dim=1)

        all_predictions.extend(predictions.cpu().numpy())

# saving test predictions as csv    
with open('submission.csv', 'w', newline='') as csvfile:
    fieldnames = ['guid', 'label']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for guid, label in zip(test_guid, all_predictions): # group predictions with guid into a csv
        writer.writerow({'guid': guid, 'label': label})