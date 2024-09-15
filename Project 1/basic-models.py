import json
from transformers import AutoTokenizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import numpy as np

#load data
with open('sentence-pair-classification-pml-2023/train.json', 'r') as file:
    train_data = json.load(file)

with open('sentence-pair-classification-pml-2023/validation.json', 'r') as file:
    valid_data = json.load(file)

#loading a trained (by us) tokenizer
tokenizer = AutoTokenizer.from_pretrained("./Send/MyTokenizer15k")

# tokenize each pair of sentences
# special separator is added by default between them
def tokenize_data(data, tokenizer):
    sentences = [(item['sentence1'], item['sentence2']) for item in data]
    labels = [item['label'] for item in data]

    encoded_data = tokenizer(sentences, truncation=True, padding=True, return_tensors='np', max_length=512)
    return encoded_data, labels

train_encoded, train_labels = tokenize_data(train_data, tokenizer)
valid_encoded, valid_labels = tokenize_data(valid_data, tokenizer)

# make numpy arrays for encodings
X_train = np.hstack([train_encoded['input_ids'], train_encoded['attention_mask']])
X_valid = np.hstack([valid_encoded['input_ids'], valid_encoded['attention_mask']])

# train a random forest
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, train_labels)

# predict with the model and make clasification report
predictions = clf.predict(X_valid)
print(classification_report(valid_labels, predictions))


import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
import numpy as np

with open('sentence-pair-classification-pml-2023/train.json', 'r') as file:
    train_data = json.load(file)

with open('sentence-pair-classification-pml-2023/validation.json', 'r') as file:
    valid_data = json.load(file)

train_texts = [f"{item['sentence1']} {item['sentence2']}" for item in train_data]
valid_texts = [f"{item['sentence1']} {item['sentence2']}" for item in valid_data]

vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(train_texts) # fit the count vectorizer only on training data
X_valid = vectorizer.transform(valid_texts)     # transform the values for both training and validation

train_labels = np.array([item['label'] for item in train_data])
valid_labels = np.array([item['label'] for item in valid_data])

# train random forest
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, train_labels)

predictions = clf.predict(X_valid)
print(classification_report(valid_labels, predictions))


import csv

with open('sentence-pair-classification-pml-2023/test.json', 'r') as file:
    test_data = json.load(file)

test_texts = [f"{item['sentence1']} {item['sentence2']}" for item in test_data]
X_test = vectorizer.transform(test_texts) # transform the test data with the same vectorizer

# make inference
predictions = clf.predict(X_test)
test_guid = np.array([item['guid'] for item in test_data]) # load guids

# save predictions for test data
with open('submission_rf_cv.csv', 'w', newline='') as csvfile:
    fieldnames = ['guid', 'label']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for guid, label in zip(test_guid, predictions):
        writer.writerow({'guid': guid, 'label': label})


import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
import numpy as np

with open('sentence-pair-classification-pml-2023/train.json', 'r') as file:
    train_data = json.load(file)

with open('sentence-pair-classification-pml-2023/validation.json', 'r') as file:
    valid_data = json.load(file)

train_texts = [f"{item['sentence1']} {item['sentence2']}" for item in train_data]
valid_texts = [f"{item['sentence1']} {item['sentence2']}" for item in valid_data]

vectorizer = TfidfVectorizer() # define tf-idf vectorizer
X_train = vectorizer.fit_transform(train_texts) 
X_valid = vectorizer.transform(valid_texts)

train_labels = np.array([item['label'] for item in train_data])
valid_labels = np.array([item['label'] for item in valid_data])

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, train_labels)

predictions = clf.predict(X_valid)
print(classification_report(valid_labels, predictions))


import json
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
import numpy as np

with open('sentence-pair-classification-pml-2023/train.json', 'r') as file:
    train_data = json.load(file)

with open('sentence-pair-classification-pml-2023/validation.json', 'r') as file:
    valid_data = json.load(file)

train_texts = [f"{item['sentence1']} {item['sentence2']}" for item in train_data]
valid_texts = [f"{item['sentence1']} {item['sentence2']}" for item in valid_data]

vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(train_texts)
X_valid = vectorizer.transform(valid_texts)

train_labels = np.array([item['label'] for item in train_data])
valid_labels = np.array([item['label'] for item in valid_data])

clf = SVC(kernel='rbf', C=0.01, random_state=42)  # define a SVM
clf.fit(X_train, train_labels)

predictions = clf.predict(X_valid)
print(classification_report(valid_labels, predictions))
