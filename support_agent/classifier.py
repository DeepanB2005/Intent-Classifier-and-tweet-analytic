
import joblib
from sentence_transformers import SentenceTransformer


class IntentClassifier:

    def __init__(self, model_path):

        model_data = joblib.load(model_path)

        self.classifier = model_data["classifier"]
        self.label_encoder = model_data["label_encoder"]

        self.model_name = model_data["sentence_model_name"]
        self.normalize_embeddings = model_data["normalize_embeddings"]

        self.embedding_model = SentenceTransformer(
            self.model_name
        )

    def predict(self, text):

        embedding = self.embedding_model.encode(
            [text],
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings
        )

        prediction = self.classifier.predict(
            embedding
        )

        intent = self.label_encoder.inverse_transform(
            prediction
        )[0]

        return intent

    def encode(self, texts):

        return self.embedding_model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
