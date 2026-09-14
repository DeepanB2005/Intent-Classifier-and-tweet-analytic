# Intent Classifier and Support Agent

# 🛠️ Support Agent Pipeline

The trained intent classifier is integrated into a larger support-agent workflow.

After the customer's intent is identified, the system retrieves relevant support information and uses it to construct an appropriate response.

```text
Customer Message
       │
       ▼
Intent Classification
       │
       ▼
Predicted Intent
       │
       ▼
Relevant Knowledge Retrieval
       │
       ▼
Action Generation
       │
       ▼
Response Generation
       │
       ▼
Handling Decision
       │
       ▼
Structured JSON Output
```

The support agent produces structured information such as:

```json
{
    "intent": "account_access",
    "action": "Ask for more details about the login issue",
    "response": "We want to help you regain access to your account...",
    "decision": "AUTO_HANDLE",
    "reason": "The issue is a standard account-access problem."
}
```

---

# 🔎 Knowledge Retrieval

The support-agent stage uses **retrieval-based support knowledge** to provide relevant information for the predicted customer intent.

FAISS is used for efficient vector similarity search.

```text
Support Knowledge
       │
       ▼
Vector Representation
       │
       ▼
FAISS Index
       │
       ▼
Similarity Search
       │
       ▼
Relevant Support Information
```

This retrieved information is then provided to the response-generation component.

---

# ✨ Response Generation

Google Gemini is used in the support-agent stage to generate the customer-facing response and structured support output.

The LLM receives the relevant context and predicted intent and produces:

* Recommended action
* Customer-facing response
* Handling decision
* Reason for the decision

The output is structured using a defined schema rather than relying only on free-form text.

---


### Inference

The trained classifier can then be used to predict the intent of new customer messages.

Example:

```python
message = "I was charged twice for the same order"

prediction = classifier.predict(features)

print(prediction)
```

---

# 💻 Running the Notebook

The notebook is designed to be **environment-independent**.

## 1. Install Dependencies

Run the following cell:

```python
%pip install -q pandas numpy scikit-learn sentence-transformers faiss-cpu joblib pydantic google-genai
```

Using `%pip` makes the installation suitable for notebook environments.

---

## 2. Clone the Repository

If the repository is not already available:

```python
import subprocess
from pathlib import Path

REPO_URL = "https://github.com/DeepanB2005/Intent-Classifier-and-tweet-analytic.git"
REPO_DIR = Path.cwd() / "Intent-Classifier-and-tweet-analytic"

if not REPO_DIR.exists():
    subprocess.run(
        ["git", "clone", REPO_URL, str(REPO_DIR)],
        check=True
    )

print("Repository:", REPO_DIR)
```

The notebook does not assume a fixed directory such as:

```text
/kaggle/working/
```

---

## 3. Configure the Gemini API

For the support-agent component, provide your Gemini API key through the `GEMINI_API_KEY` environment variable.

For interactive notebook use:

```python
import os
from getpass import getpass

if not os.environ.get("GEMINI_API_KEY"):
    os.environ["GEMINI_API_KEY"] = getpass(
        "Enter your Gemini API key: "
    )

if not os.environ.get("GEMINI_API_KEY"):
    raise ValueError("GEMINI_API_KEY was not provided.")
```

## 4. Import the support agent

After setting the project path:
```
from support_agent.main import support_agent

Then inference remains simple:

message = "I cannot log into my Apple account"

result = support_agent(message)

print(result)
```
## 5. Environment-independent test section
   ```
       test_messages = [
           "I cannot log into my Apple account",
           "I was charged twice for the same purchase",
           "I want a refund for my purchase",
           "Where is my order?",
           "My iPhone screen is not working"
       ]

       for message in test_messages:
           print("=" * 70)
           print("CUSTOMER:", message)
       
           result = support_agent(message)
       
           print("INTENT:", result.get("intent"))
           print("ACTION:", result.get("action"))
           print("DECISION:", result.get("decision"))
           print("REASON:", result.get("reason"))
           print("RESPONSE:", result.get("response"))
       ```
The API key should **not be hard-coded or committed to the repository**.

---

# 📁 Project Structure

```text
Intent-Classifier-and-tweet-analytic/
│
├── support_agent/
│   ├── main.py
│   └── ...
│
├── models/
│   └── ...
│
├── data/
│   └── ...
│
├── notebooks/
│   └── ...
│
└── README.md
```

The exact files and directories may vary according to the current project implementation.

---

# 🧪 Example

### Input

```text
"I cannot log into my Apple account"
```

### Intent Classification

```text
account_access
```

### Support Agent

The predicted intent is used to retrieve relevant support information and generate a suitable action and response.

### Final Output

```text
Intent: account_access

Action:
Ask for clarification about what happens when the customer
attempts to log in.

Decision:
AUTO_HANDLE

Response:
Customer-facing response generated using the retrieved
support context.
```

---

# 🧰 Tech Stack

* **Python**
* **Pandas**
* **NumPy**
* **Scikit-learn**
* **Sentence Transformers**
* **PyTorch**
* **FAISS**
* **Joblib**
* **Pydantic**
* **Google Gemini**

---

# 🔄 Complete System

```text
                 RAW CUSTOMER MESSAGE
                         │
                         ▼
                 TEXT PREPROCESSING
                         │
             ┌───────────┴───────────┐
             │                       │
             ▼                       ▼
          TF-IDF             Sentence Transformer
             │                 all-MiniLM-L6-v2
             │                       │
             ▼                       ▼
       Numerical Vector       Dense Embedding
             │                       │
             └───────────┬───────────┘
                         ▼
                   Linear SVM
                         │
                         ▼
                  Predicted Intent
                         │
                         ▼
                  FAISS Retrieval
                         │
                         ▼
              Relevant Support Context
                         │
                         ▼
                 Gemini Response
                         │
                         ▼
                Structured JSON
                         │
             ┌───────────┴───────────┐
             ▼                       ▼
        AUTO_HANDLE              ESCALATE
```

---

An NLP-based customer-support intent classification system that converts customer messages into numerical feature representations and classifies them into predefined support intents.

The project explores two different approaches for text feature extraction:

1. **TF-IDF (Term Frequency–Inverse Document Frequency)**
2. **Sentence Transformer embeddings using `all-MiniLM-L6-v2`**

The extracted numerical features are then provided to machine-learning classifiers. **Linear SVM** showed the most promising performance among the evaluated classifiers and is used as the primary classifier in the final implementation.

The trained intent classifier is further integrated into a support-agent pipeline for retrieving relevant support knowledge and generating structured customer-support responses.

---

## 🚀 Project Overview

The system follows the pipeline:

```text
                 CUSTOMER MESSAGE
                        │
                        ▼
              Social Media Text
                 Preprocessing
                        │
                        ▼
             ┌──────────┴──────────┐
             │                     │
             ▼                     ▼
          TF-IDF            Sentence Transformer
       Feature Extraction       Embeddings
             │                     │
             │              all-MiniLM-L6-v2
             │                     │
             └──────────┬──────────┘
                        │
                        ▼
                  Linear SVM
                  Classifier
                        │
                        ▼
                Predicted Intent
                        │
                        ▼
             Relevant Support Knowledge
                        │
                        ▼
                 Support Agent
                        │
              ┌─────────┴─────────┐
              ▼                   ▼
        Action Generation    Response Generation
                                  │
                                  ▼
                           Structured JSON
```

---

# 🧠 Feature Extraction

Machine-learning models cannot directly process raw text. Therefore, the input text is first converted into **numerical vectors**.

These numerical representations allow the classifier to process textual information mathematically.

Two feature-extraction techniques were implemented and compared.

---

## 1. TF-IDF

**Term Frequency–Inverse Document Frequency (TF-IDF)** represents text using a numerical vector based on the importance of words within a document and across the dataset.

The basic idea is:

* **Term Frequency (TF):** how frequently a word occurs in a document.
* **Inverse Document Frequency (IDF):** reduces the importance of words that occur across many documents.
* Words that are more informative for a particular document receive higher weights.

The resulting representation is a sparse numerical vector.

```text
Raw Text
   │
   ▼
Text Preprocessing
   │
   ▼
TF-IDF Vectorizer
   │
   ▼
Numerical Feature Vector
   │
   ▼
Linear SVM
```

TF-IDF provides a strong traditional baseline for text classification and is computationally efficient.

---

## 2. Sentence Transformer Embeddings

The second approach uses a pretrained Sentence Transformer:

```text
sentence-transformers/all-MiniLM-L6-v2
```

Instead of representing text mainly through individual word importance, the Sentence Transformer generates a **dense semantic embedding** for the complete input sentence.

```text
Raw Text
   │
   ▼
Text Preprocessing
   │
   ▼
all-MiniLM-L6-v2
   │
   ▼
Dense Sentence Embedding
   │
   ▼
Linear SVM
```

This allows semantically similar sentences to have similar vector representations even when they do not contain exactly the same words.

---

# 🤖 Classifier Experiments

Several classifiers were experimented with during development:

* Logistic Regression
* Support Vector Classifier (SVC)
* Linear Support Vector Machine (Linear SVM)

The experiments showed that **Linear SVM provided the most promising classification performance** for the implemented feature representations.

Therefore, Linear SVM is used as the primary classifier in the final pipeline.

```text
Feature Extraction
       │
       ▼
Numerical Vectors
       │
       ▼
   Linear SVM
       │
       ▼
 Predicted Intent
```

---

# 🔬 Model Comparison

The project compares the following approaches:

| Feature Extraction   | Classifier |
| -------------------- | ---------- |
| TF-IDF               | Linear SVM |
| Sentence Transformer | Linear SVM |

Additional classifier experiments were also performed using:

| Classifier          | Purpose                  |
| ------------------- | ------------------------ |
| Logistic Regression | Baseline / comparison    |
| SVC                 | Classifier comparison    |
| Linear SVM          | Primary/final classifier |

The final model selection is based on evaluation metrics such as:

* Accuracy
* Precision
* Recall
* Macro F1-score

Macro F1 is particularly useful for evaluating performance across multiple intents because it gives equal importance to each class.

---

# 📊 Evaluation

The classifier is evaluated using:

```text
Accuracy
Precision
Recall
F1-score
Macro F1-score
```

A classification report is generated to evaluate performance for individual intents as well as the overall model.

Example:

```python
from sklearn.metrics import classification_report, accuracy_score, f1_score

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Macro F1:", f1_score(y_test, y_pred, average="macro"))

print(classification_report(y_test, y_pred))
```

---

# 🎯 Intent Classification

The classifier maps incoming customer-support messages to predefined intents.

Example intents include:

```text
account_access
billing_payment
complaint
refund
order_delivery
product_issue
```

The complete set of intents is determined by the training dataset and project implementation.

Example:

```text
Input:
"I cannot log into my Apple account"

Output:
account_access
```

---


# 📌 Key Contributions

* Implemented **two different text feature-extraction approaches**.
* Compared traditional **TF-IDF** representations with **Sentence Transformer semantic embeddings**.
* Experimented with **Logistic Regression, SVC, and Linear SVM**.
* Selected **Linear SVM** as the primary classifier based on promising observed performance.
* Evaluated the classifier using multiple classification metrics.
* Integrated the intent classifier into a **support-agent pipeline**.
* Implemented **FAISS-based support knowledge retrieval**.
* Integrated **Gemini** for support-response generation.
* Designed structured output containing intent, action, response, decision, and reason.
* Designed the notebook to work across common notebook environments rather than depending on a specific platform.

---

# ⚠️ Security

Never expose API keys in source code or commit them to GitHub.

Use:

```text
GEMINI_API_KEY
```

as an environment variable.

---

# 🔮 Future Improvements

Possible future improvements include:

* Increasing the size and diversity of the training dataset
* Improving performance on minority intents
* Hyperparameter tuning for Linear SVM
* Comparing additional embedding models
* Adding classifier confidence scores
* Improving retrieval quality
* Expanding the support knowledge base
* Adding conversational context
* Building an API or web interface
* Deploying the support agent as a production service
