
from google import genai

from .schemas import SupportAgentOutput


class SupportAgent:

    def __init__(
        self,
        classifier,
        retriever,
        api_key,
        model_name="gemini-3.5-flash-lite"
    ):

        self.classifier = classifier
        self.retriever = retriever

        self.client = genai.Client(
            api_key=api_key
        )

        self.model_name = model_name

    def build_context(self, results):

        # Retriever returns a dictionary.
        # Extract the DataFrame containing the retrieved examples.
        if isinstance(results, dict):
            results = results["results"]

        context = []

        for i, (_, row) in enumerate(
            results.iterrows()
        ):

            example = f"""
Historical Example {i + 1}

Customer:
{row["customer_message"]}

Apple Response:
{row["apple_response"]}

Historical Intent:
{row["intent"]}

Semantic Similarity:
{row["semantic_score"]:.3f}

Intent Match:
{bool(row["intent_match"])}
"""

            context.append(example)

        return "\n".join(context)
        
    def build_prompt(
        self,
        customer_message,
        predicted_intent,
        rag_context
    ):

        return f"""
You are an AI customer-support agent for Apple.

Your task is to analyze the customer message and produce:

1. The customer's intent
2. The recommended support action
3. A concise customer-facing response
4. Whether the issue should be automatically handled or escalated
5. A concise reason for that decision

IMPORTANT:

The dedicated intent classifier predicted:

{predicted_intent}

The `intent` field in your output MUST exactly equal this
predicted intent.

Do not change, reinterpret, or replace the classifier's intent.

Use the historical Apple support examples as evidence for
how similar issues were handled.

RULES:

1. Preserve the classifier's predicted intent exactly.
2. Ground the response in the historical Apple examples.
3. Do not invent Apple policies.
4. Do not invent refund or replacement guarantees.
5. Do not fabricate troubleshooting procedures.
6. Do not claim that an issue has been resolved when it has not.
7. If the issue requires account-specific investigation,
   private information, or unsupported action, choose ESCALATE.
8. If the issue can be safely addressed using the available
   historical evidence, choose AUTO_HANDLE.
9. Keep the response concise and professional.
10. Do not mention RAG, embeddings, classifiers, or the LLM.
11. Do not expose internal reasoning.
12. Do not reproduce shortened Twitter/X URLs such as t.co.
13. If a historical response contains a shortened URL,
    provide the useful guidance without reproducing the URL.

CUSTOMER MESSAGE:

{customer_message}

CLASSIFIER PREDICTED INTENT:

{predicted_intent}

HISTORICAL APPLE SUPPORT EXAMPLES:

{rag_context}

Return only the requested structured output.
"""

    def generate(self, customer_message):

        # --------------------------------
        # 1. Predict intent
        # --------------------------------

        predicted_intent = self.classifier.predict(
            customer_message
        )

        # --------------------------------
        # 2. Retrieve historical examples
        # --------------------------------

        retrieved = self.retriever.retrieve(
            query=customer_message,
            predicted_intent=predicted_intent,
            initial_k=20,
            final_k=5
        )

        # --------------------------------
        # 3. Build RAG context
        # --------------------------------

        rag_context = self.build_context(
            retrieved
        )

        # --------------------------------
        # 4. Build prompt
        # --------------------------------

        prompt = self.build_prompt(
            customer_message=customer_message,
            predicted_intent=predicted_intent,
            rag_context=rag_context
        )

        # --------------------------------
        # 5. Generate structured response
        # --------------------------------

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_schema": SupportAgentOutput,
            }
        )

        # --------------------------------
        # 6. Validate output
        # --------------------------------

        result = SupportAgentOutput.model_validate_json(
            response.text
        )

        # --------------------------------
        # 7. Safety check:
        #    LLM must not change intent
        # --------------------------------

        if result.intent != predicted_intent:

            result.intent = predicted_intent

        return result, retrieved
