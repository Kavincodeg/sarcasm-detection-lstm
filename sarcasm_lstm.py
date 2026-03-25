# SARCASM DETECTION USING LSTM (FINAL PERFECT VERSION)

import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, Dropout

print("\nSarcasm Detection System Starting... 🚀\n")

# Load dataset (JSONL format)
data = []
with open("Sarcasm_Headlines_Dataset.json", 'r') as f:
    for line in f:
        data.append(json.loads(line))

# Extract data
sentences = [item['headline'] for item in data]
labels = [item['is_sarcastic'] for item in data]

# Tokenization
vocab_size = 5000
max_len = 40

tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)

sequences = tokenizer.texts_to_sequences(sentences)
padded = pad_sequences(sequences, maxlen=max_len, padding='post')

labels = np.array(labels)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    padded, labels, test_size=0.2, random_state=42
)

# Compute class weights
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights = dict(enumerate(class_weights))

# Model
model = Sequential([
    Embedding(vocab_size, 128, input_length=max_len),
    
    Bidirectional(LSTM(64)),
    
    Dense(64, activation='relu'),
    Dropout(0.5),
    
    Dense(1, activation='sigmoid')
])

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# Train
print("Training model...\n")
model.fit(
    X_train, y_train,
    epochs=7,
    validation_data=(X_test, y_test),
    class_weight=class_weights
)

# Evaluate
loss, accuracy = model.evaluate(X_test, y_test)
print("\nModel Accuracy:", accuracy)

# FINAL PREDICTION FUNCTION (FIXED)
def predict_sarcasm(text):
    seq = tokenizer.texts_to_sequences([text])
    pad = pad_sequences(seq, maxlen=max_len, padding='post')
    pred = model.predict(pad, verbose=0)[0][0]

    # 🔥 refined sarcasm keywords (NO "love")
    sarcasm_keywords = [
    "yeah", "oh great", "great", "wow",
    "fantastic", "amazing", "perfect",
    "just what i needed"
    ]

    text_lower = text.lower()
    keyword_flag = any(word in text_lower for word in sarcasm_keywords)

    # Hybrid decision
    if pred > 0.3 or keyword_flag:
        return "Sarcastic 😏"
    else:
        return "Normal 😊"

# User input loop
print("\nSystem Ready! Type 'exit' to stop.\n")

while True:
    text = input("Enter sentence: ")

    if text.lower() == "exit":
        print("Exiting... 👋")
        break

    print("Prediction:", predict_sarcasm(text))