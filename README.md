# Intent Classifier and Support Agent

An NLP-based customer-support intent classification system that classifies customer messages into predefined support intents.

The project implements and compares two text feature extraction approaches:

* **TF-IDF**
* **Sentence Transformer (`all-MiniLM-L6-v2`)**

Multiple classifiers were evaluated, including **Logistic Regression, SVC, and Linear SVM**. Linear SVM showed the most promising performance and is used as the primary classifier.

The trained intent classifier is integrated with a support-agent pipeline using **FAISS-based retrieval** and **Google Gemini** for generating support actions and responses.

---

## Implementation

The notebook implements the following stages:

1. Load and preprocess the customer-support dataset.
2. Convert text into numerical features using:

   * TF-IDF
   * Sentence Transformer embeddings
3. Train and evaluate:

   * Logistic Regression
   * SVC
   * Linear SVM
4. Select the Linear SVM classifier for the final implementation.
5. Save/load the trained model and required preprocessing components.
6. Use the predicted intent for support-knowledge retrieval with FAISS.
7. Generate the final support action and response using Google Gemini.
8. Return the result in a structured format.

---

## Requirements

The notebook can be run in common Python notebook environments such as:

* Jupyter Notebook
* JupyterLab
* Google Colab
* VS Code Notebook
* Kaggle Notebook

Python 3.x is recommended.

---

## Setup

### 1. Clone the Repository

Run the following in a notebook cell:

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

print(f"Repository: {REPO_DIR}")
```

Add the repository to the Python path:

```python
import sys

if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))
```

### 2. Install Dependencies

```python
%pip install -q pandas numpy scikit-learn sentence-transformers faiss-cpu joblib pydantic google-genai
```

### 3. Configure Gemini API

The support-agent component requires a Gemini API key.

Set the key using the `GEMINI_API_KEY` environment variable.

For interactive notebook execution:

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

Do not hard-code or commit the API key to the repository.

---

## Running the Notebook

After completing the setup, run the notebook cells sequentially.

The notebook supports:

### Training

Train the intent classifier using the prepared dataset and compare the implemented feature-extraction and classification approaches.

### Evaluation

Evaluate the models using:

* Accuracy
* Precision
* Recall
* F1-score
* Macro F1

### Inference

Use the trained classifier to predict the intent of new customer messages.

Example:

```python
message = "I cannot log into my Apple account"

prediction = classifier.predict(features)

print(prediction)
```

### Support Agent

The trained implementation can be loaded using:

```python
from support_agent.main import support_agent
```

Then:

```python
result = support_agent(
    "I cannot log into my Apple account"
)

print(result)
```

The support agent returns structured information containing:

```text
intent
action
response
decision
reason
```

---

## Project Structure

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

The exact files and model artifacts may vary depending on the current project version.

---

## Main Technologies

* Python
* Pandas
* NumPy
* Scikit-learn
* Sentence Transformers
* PyTorch
* FAISS
* Joblib
* Pydantic
* Google Gemini

---

## Models

### Feature Extraction

```text
TF-IDF
```

and

```text
Sentence Transformer
all-MiniLM-L6-v2
```

### Classifiers

```text
Logistic Regression
SVC
Linear SVM
```

**Linear SVM** is used as the primary classifier based on the evaluation performed during development.

---

## Notes

* The notebook is designed to be independent of a specific notebook platform.
* Internet access is required for installing dependencies and cloning the repository.
* The Sentence Transformer model may be downloaded automatically on first use.
* A Gemini API key is required only for the support-agent response-generation stage.
* Keep API credentials outside the source code.
