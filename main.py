import re
import html
import unicodedata
import joblib
import pandas as pd

from scipy.sparse import hstack
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURATION
# ============================================================

TFIDF_MODEL_PATH = "/content/tfidf_intent_classifier.pkl"
ST_MODEL_PATH = "/content/sentence_transformer_intent_classifier (1).pkl"

DEFAULT_SENTENCE_MODEL = "all-MiniLM-L6-v2"


# ============================================================
# PREPROCESSING
# ============================================================

def clean_tweet(text):

    if pd.isna(text):
        return ""

    text = str(text)

    # Decode HTML entities
    text = html.unescape(text)

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Unicode normalization
    text = unicodedata.normalize("NFKC", text)

    # Replace URLs
    text = re.sub(
        r"https?://\S+|www\.\S+",
        " URL ",
        text
    )

    # Remove mentions
    text = re.sub(
        r"@\w+",
        " ",
        text
    )

    # Keep hashtag word
    # #iphone -> iphone
    text = re.sub(
        r"#(\w+)",
        r"\1",
        text
    )

    # Lowercase
    text = text.lower()

    # Normalize repeated characters
    # heyyyy -> heyy
    text = re.sub(
        r"(.)\1{2,}",
        r"\1\1",
        text
    )

    # Normalize whitespace
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# ============================================================
# LOAD TF-IDF MODEL
# ============================================================

print("Loading TF-IDF model...")

tfidf_artifact = joblib.load(
    TFIDF_MODEL_PATH
)

tfidf_classifier = tfidf_artifact["model"]

tfidf_word_vectorizer = (
    tfidf_artifact["word_vectorizer"]
)

tfidf_char_vectorizer = (
    tfidf_artifact["char_vectorizer"]
)

print("TF-IDF model loaded.")


# ============================================================
# LOAD SENTENCE TRANSFORMER MODEL
# ============================================================

print("Loading Sentence Transformer model...")

st_artifact = joblib.load(
    ST_MODEL_PATH
)

st_classifier = st_artifact["classifier"]

# ------------------------------------------------------------
# IMPORTANT:
# Load LabelEncoder if it was saved
# ------------------------------------------------------------

st_label_encoder = st_artifact.get(
    "label_encoder",
    None
)

# ------------------------------------------------------------
# Fallback to classes if LabelEncoder isn't available
# ------------------------------------------------------------

st_classes = st_artifact.get(
    "classes",
    None
)

sentence_model_name = st_artifact.get(
    "sentence_model_name",
    DEFAULT_SENTENCE_MODEL
)

sentence_encoder = SentenceTransformer(
    sentence_model_name
)

print("Sentence Transformer model loaded.")


# ============================================================
# SENTENCE TRANSFORMER LABEL DECODER
# ============================================================

def decode_sentence_prediction(prediction):

    """
    Convert numerical SVM output into the actual intent label.

    Example:

        0 -> account_access
        1 -> billing_payment
        ...
    """

    prediction = int(prediction)

    # --------------------------------------------------------
    # Preferred method: LabelEncoder
    # --------------------------------------------------------

    if st_label_encoder is not None:

        return st_label_encoder.inverse_transform(
            [prediction]
        )[0]

    # --------------------------------------------------------
    # Fallback: saved class list
    # --------------------------------------------------------

    if st_classes is not None:

        if prediction < 0 or prediction >= len(st_classes):

            raise ValueError(
                f"Invalid prediction index {prediction}. "
                f"Available classes: {len(st_classes)}"
            )

        return st_classes[prediction]

    # --------------------------------------------------------
    # No decoder available
    # --------------------------------------------------------

    raise ValueError(
        "No label decoder found in the "
        "Sentence Transformer model artifact."
    )


# ============================================================
# TF-IDF PREDICTION
# ============================================================

def predict_tfidf(text):

    cleaned_text = clean_tweet(text)

    # Word TF-IDF
    word_features = (
        tfidf_word_vectorizer.transform(
            [cleaned_text]
        )
    )

    # Character TF-IDF
    char_features = (
        tfidf_char_vectorizer.transform(
            [cleaned_text]
        )
    )

    # Combine features
    features = hstack(
        [
            word_features,
            char_features
        ]
    )

    # Safety check
    expected_features = (
        tfidf_classifier.n_features_in_
    )

    if features.shape[1] != expected_features:

        raise ValueError(
            f"TF-IDF feature mismatch: "
            f"got {features.shape[1]}, "
            f"expected {expected_features}"
        )

    prediction = tfidf_classifier.predict(
        features
    )[0]

    return prediction


# SENTENCE TRANSFORMER + SVM PREDICTION

def predict_sentence_transformer(text):

    cleaned_text = clean_tweet(text)

    embedding = sentence_encoder.encode(
        [cleaned_text],
        normalize_embeddings=True
    )

    prediction_id = st_classifier.predict(
        embedding
    )[0]

    classes = st_artifact["classes"]

    prediction = classes[int(prediction_id)]

    return prediction

# BOTH MODELS

def predict(text):

    tfidf_prediction = predict_tfidf(
        text
    )

    st_prediction = predict_sentence_transformer(
        text
    )

    return {
        "text": text,

        "tfidf_logistic_regression":
            tfidf_prediction,

        "sentence_transformer_svm":
            st_prediction
    }


# DATASET PREDICTION

def predict_dataset(
    input_file,
    output_file="predicted_intents.csv"
):

    df = pd.read_csv(
        input_file
    )

    if "text" not in df.columns:

        raise ValueError(
            "CSV must contain a 'text' column."
        )

    print(
        f"Loaded {len(df)} rows."
    )

    # Preprocess

    cleaned_texts = [
        clean_tweet(text)
        for text in df["text"]
    ]

#..==
    # MODEL 1
    # TF-IDF + Logistic Regression
#..==

    print(
        "\nRunning TF-IDF + Logistic Regression..."
    )

    word_features = (
        tfidf_word_vectorizer.transform(
            cleaned_texts
        )
    )

    char_features = (
        tfidf_char_vectorizer.transform(
            cleaned_texts
        )
    )

    combined_features = hstack(
        [
            word_features,
            char_features
        ]
    )

    tfidf_predictions = (
        tfidf_classifier.predict(
            combined_features
        )
    )

#..==
    # MODEL 2
    # Sentence Transformer + Linear SVM
#..==

    print(
        "Running Sentence Transformer + Linear SVM..."
    )

    embeddings = sentence_encoder.encode(
        cleaned_texts,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    numerical_predictions = (
        st_classifier.predict(
            embeddings
        )
    )

    # --------------------------------------------------------
    # Decode ALL numerical predictions
    # --------------------------------------------------------

    st_predictions = [
        decode_sentence_prediction(
            prediction
        )
        for prediction in numerical_predictions
    ]

#..==
    # ADD RESULTS
#..==

    df[
        "tfidf_logistic_regression"
    ] = tfidf_predictions

    df[
        "sentence_transformer_svm"
    ] = st_predictions

    # SAVE

    df.to_csv(
        output_file,
        index=False
    )

    print(
        f"\nSaved predictions to: "
        f"{output_file}"
    )

    return df


# ============================================================
# COMMAND LINE INTERFACE
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("INTENT CLASSIFICATION SYSTEM")
    print("=" * 60)

    print("\n1. Single text")
    print("2. CSV dataset")

    choice = input(
        "\nSelect option (1/2): "
    ).strip()

    # SINGLE TEXT_input 

    if choice == "1":

        text = input(
            "\nEnter text:\n"
        )

        result = predict(
            text
        )

        print("\n" + "=" * 60)
        print("PREDICTION RESULTS")
        print("=" * 60)

        print(
            f"\nInput:\n{text}"
        )

        print(
            "\nTF-IDF + Logistic Regression:"
        )

        print(
            f"→ {result['tfidf_logistic_regression']}"
        )

        print(
            "\nSentence Transformer + Linear SVM:"
        )

        print(
            f"→ {result['sentence_transformer_svm']}"
        )

    # DATASET
    elif choice == "2":

        input_file = input(
            "\nEnter CSV path:\n"
        ).strip()

        output_file = input(
            "\nEnter output filename "
            "(Enter = predicted_intents.csv):\n"
        ).strip()

        if not output_file:

            output_file = (
                "predicted_intents.csv"
            )

        predict_dataset(
            input_file,
            output_file
        )

    else:

        print(
            "Invalid option."
        )