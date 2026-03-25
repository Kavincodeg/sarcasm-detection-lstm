# 🤖 Sarcasm Detection using LSTM

## 📌 Project Overview

This project detects whether a given sentence is **sarcastic 😏 or normal 😊** using Deep Learning techniques.
It uses a **Bidirectional LSTM (Long Short-Term Memory)** model along with a **hybrid rule-based approach** to improve prediction accuracy.

---

## 🚀 Features

* 🔹 Bidirectional LSTM for better context understanding
* 🔹 Handles class imbalance using class weights
* 🔹 Hybrid approach (Deep Learning + keyword detection)
* 🔹 Real-time user input prediction
* 🔹 Simple and easy to run

---

## 🧠 Technologies Used

* Python
* TensorFlow / Keras
* NumPy
* Scikit-learn

---

## 📂 Dataset

Dataset used: **Sarcasm Headlines Dataset**

👉 Download from:
https://www.kaggle.com/datasets/rmisra/news-headlines-dataset-for-sarcasm-detection

⚠️ Note: Dataset is not included in this repository due to size.

---

## ⚙️ Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/sarcasm-detection-lstm.git
cd sarcasm-detection-lstm
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add dataset

* Download dataset from Kaggle
* Place file in project folder
* Ensure file name:

```plaintext
Sarcasm_Headlines_Dataset.json
```

---

## ▶️ Run the Project

```bash
python sarcasm_lstm.py
```

---

## 🧪 Example Output

### Input:

```plaintext
Yeah, I totally love being ignored
```

### Output:

```plaintext
Sarcastic 😏
```

---

### Input:

```plaintext
This food tastes delicious
```

### Output:

```plaintext
Normal 😊
```

---

## 📊 Model Details

* Embedding Layer (128 units)
* Bidirectional LSTM (64 units)
* Dense Layer + Dropout
* Binary Classification (Sigmoid)

---

## 🎯 Results

* Achieved around **85% accuracy**
* Improved performance using:

  * Class balancing
  * Hybrid keyword logic

---

## 🧠 Key Insight

Sarcasm detection is challenging because it depends on **context, tone, and implicit meaning**, which are difficult for models to fully capture.

---

## 🔥 Future Improvements

* Use Transformer models (BERT)
* Improve contextual understanding
* Deploy as a web application

---

## 👨‍💻 Author

**Your Name**

---

## ⭐ Acknowledgment

This project is developed as part of an academic assignment in Deep Learning / NLP.
