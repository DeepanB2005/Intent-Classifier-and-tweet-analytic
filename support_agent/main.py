
import os
import json

from .classifier import IntentClassifier
from .retriever import AppleRetriever
from .agent import SupportAgent


BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

ARTIFACT_DIR = os.path.join(
    BASE_DIR,
    "artifacts"
)


CLASSIFIER_PATH = os.path.join(
    ARTIFACT_DIR,
    "sentence_transformer_intent_classifier.pkl"
)

RAG_PATH = os.path.join(
    ARTIFACT_DIR,
    "apple_rag_with_intents.csv"
)

INDEX_PATH = os.path.join(
    ARTIFACT_DIR,
    "apple_rag.index"
)


# ---------------------------------------
# Load classifier
# ---------------------------------------

classifier = IntentClassifier(
    CLASSIFIER_PATH
)


# ---------------------------------------
# Load retriever
# ---------------------------------------

retriever = AppleRetriever(
    rag_path=RAG_PATH,
    index_path=INDEX_PATH,
    embedding_model=classifier.embedding_model,
    classifier=classifier
)


# ---------------------------------------
# Load Gemini agent
# ---------------------------------------

agent = SupportAgent(
    classifier=classifier,
    retriever=retriever,
    api_key=os.environ["GEMINI_API_KEY"]
)


def support_agent(customer_message):

    if not isinstance(
        customer_message,
        str
    ):
        raise TypeError(
            "customer_message must be a string"
        )

    if not customer_message.strip():
        raise ValueError(
            "customer_message cannot be empty"
        )

    result, _ = agent.generate(
        customer_message.strip()
    )

    return result.model_dump()


if __name__ == "__main__":

    customer_message = input(
        "Customer message: "
    )

    result = support_agent(
        customer_message
    )

    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False
        )
    )
