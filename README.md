# Intent Classifier and Support Agent

An AI-powered customer support intent classification and response system built using **Sentence Transformers, FAISS, and Google Gemini**.

The project classifies incoming customer-support messages into predefined intents and generates an appropriate support action, response, and handling decision.

## 🚀 Implementation in the Notebook

The attached Kaggle notebook demonstrates the complete **inference pipeline** using the trained implementation available in the project repository.

### Pipeline

```text
Customer Message
       │
       ▼
Support Agent
       │
       ├── Intent Classification (sentence transformer embedings + Linear SVM classifier)
       │
       ├── Relevant Support Knowledge (Retreival Augumented generation)
       │
       ├── Action Generation
       │
       ├── Response Generation (Gemini flash 3.5)
       │
       └── Handling Decision (Json Output)
              │
              ▼
       Structured Result
```

The notebook performs the following steps:

### 1. Clone the Project Repository

The notebook first clones the project repository:

```bash
git clone https://github.com/DeepanB2005/Intent-Classifier-and-tweet-analytic.git
```

and switches into the project directory.

### 2. Install Required Dependencies

The implementation uses the following major libraries:

* `pandas`
* `numpy`
* `faiss-cpu`
* `sentence-transformers`
* `scikit-learn`
* `joblib`
* `pydantic`
* `google-genai`
* 
```bash
!pip install -q pandas numpy faiss-cpu sentence-transformers scikit-learn joblib pydantic google-genai
```

These dependencies support embedding generation, similarity search, machine-learning components, structured outputs, and LLM-based response generation.

### 3. Configure Gemini API

The notebook retrieves the Gemini API key from **Kaggle Secrets** and exposes it through the `GEMINI_API_KEY` environment variable.

```python
from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()
os.environ["GEMINI_API_KEY"] = user_secrets.get_secret("GEMINI_API_KEY")
```

This keeps the API key outside the notebook source code.

### 4. Load the Support Agent

The trained/project implementation is loaded through:

```python
from support_agent.main import support_agent
```

The implementation loads the Sentence Transformer model:

```text
sentence-transformers/all-MiniLM-L6-v2
```

The notebook therefore uses the existing project implementation rather than retraining the models inside the notebook.

### 5. Single-Message Inference

A customer message can be passed directly to the support agent:

```python
message = "I cannot log into my Apple account"

result = support_agent(message)
```

The system returns a structured result containing:

```text
intent
action
response
decision
reason
```

Example:

```text
Intent: account_access
Action: Ask for clarification on what happens when the customer tries to log in.

Decision: AUTO_HANDLE
```

### 6. Multiple Test Cases

The notebook also evaluates the system on multiple example customer-support messages:

```python
test_messages = [
    "I cannot log into my Apple account",
    "I was charged twice for the same purchase",
    "I want a refund for my purchase",
    "Where is my order?",
    "My iPhone screen is not working"
]
```

Each message is passed through the same support-agent pipeline and the resulting intent, action, response, decision, and reason are displayed.

## 📌 Example Results

| Customer Query                            | Predicted Intent  | Decision      |
| ----------------------------------------- | ----------------- | ------------- |
| I cannot log into my Apple account        | `account_access`  | `AUTO_HANDLE` |
| I was charged twice for the same purchase | `billing_payment` | `ESCALATE`    |
| I want a refund for my purchase           | `refund`          | `AUTO_HANDLE` |
| Where is my order?                        | `order_delivery`  | `AUTO_HANDLE` |
| My iPhone screen is not working           | `product_issue`   | `AUTO_HANDLE` |

### Example Structured Output

```python
{
    "intent": "account_access",
    "action": "Ask for more details about what happens when attempting to log in",
    "response": "We want you to be able to access your Apple ID account, and we'll do all we can to help...",
    "decision": "AUTO_HANDLE",
    "reason": "The customer is experiencing a standard login issue..."
}
```

## 🧠 Technologies Used

### Sentence Transformers

The implementation uses:

```text
sentence-transformers/all-MiniLM-L6-v2
```

to generate semantic representations of customer messages.

This allows the system to identify the intent of messages based on their semantic meaning rather than relying only on exact keyword matches.

### FAISS

**FAISS** is used for efficient similarity search over vector representations, allowing incoming messages to be compared against relevant support examples/knowledge.

### Google Gemini

The Gemini API is integrated into the support-agent pipeline for generating structured support decisions and responses based on the classified customer issue and retrieved information.

### Scikit-learn

Scikit-learn is included as part of the project's machine-learning stack and supports the classification components used by the overall implementation.

## 🎯 Supported Intent Examples

The notebook demonstrates classification into intents including:

* `account_access`
* `billing_payment`
* `refund`
* `order_delivery`
* `product_issue`

The complete set of supported intents is defined by the underlying project implementation.

## 🔄 End-to-End Flow

For each customer message, the system follows this general process:

```text
1. Receive customer message
          ↓
2. Convert/query message using semantic representation
          ↓
3. Retrieve relevant support information
          ↓
4. Determine customer intent
          ↓
5. Generate recommended support action
          ↓
6. Generate customer-facing response
          ↓
7. Decide AUTO_HANDLE or ESCALATE
          ↓
8. Return structured result
```

## 🧪 Notebook

The Kaggle notebook demonstrates:

* Project setup
* Dependency installation
* Secure Gemini API configuration
* Loading the trained support-agent implementation
* Single-message inference
* Batch testing with multiple customer queries
* Display of intent, action, response, decision, and reasoning

## ⚠️ Important Note

The attached notebook **does not perform model training from scratch**. It loads the existing implementation from the project repository and demonstrates its inference and support-agent functionality.

The notebook is therefore intended as a **deployment/inference and demonstration notebook** for the trained system.

## 📂 Project Structure

The notebook expects the following project structure after cloning the repository:

```text
Intent-Classifier-and-tweet-analytic/
│
├── support_agent/
│   ├── main.py
│   └── ...
│
├── ...
│
└── notebook
```

The exact supporting files and model artifacts are maintained in the project repository.

## ▶️ Running the Notebook

The notebook is designed to run in a **Kaggle Notebook** environment.

### Requirements

* Kaggle Notebook
* Internet access for installing dependencies and cloning the repository
* Gemini API key stored in Kaggle Secrets as:

```text
GEMINI_API_KEY
```

### Execution

Run the notebook cells sequentially.

After setup, the support agent can be tested using:

```python
result = support_agent("Your customer-support message")
print(result)
```


The objective of the project is to build an automated customer-support system capable of:

* Understanding customer intent
* Retrieving relevant support information
* Generating an appropriate support action
* Producing a customer-facing response
* Determining whether the issue can be automatically handled or should be escalated

# Intent-Classifier-and-tweet-analytic
An NLP-based intent classification system that identifies customer queries from social media posts and categorizes them into predefined support intents. The project compares TF-IDF + traditional ML models with Sentence Transformer embeddings + classifiers such as Linear SVM and Logistic Regression.

Social-media text preprocessing
TF-IDF-based intent classification
Sentence Transformer embeddings
Linear SVM / Logistic Regression classification
Evaluation using Accuracy, Precision, Recall, and Macro F1
Designed to handle large-scale customer support datasets
Kaggle-ready training and inference pipeline

Tech Stack: Python · NLP · Scikit-learn · Sentence Transformers · Pandas · PyTorch



I have used 2 feature extraction techniques for converting the input data into numerical features 
the numerical veectors represent the importance of the every token in the input text data
and also the vectors will be in numerical format so that the model can understand and process it 

1.Term-frequency and Inverse Document Frequence
2.used a sentence transformer- All miniL LM v6

after this the both classifiers has used Linear SVM model as i have trained logistic regression,SVC also but linear SVM shows some promising performance

