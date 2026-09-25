from google import genai

from app.core.config import settings


class LLMService:
    MODEL_NAME = "gemini-3.6-flash"

    def __init__(self):
        self.client = genai.Client(
            api_key=settings.gemini_api_key
        )

    def generate(
        self,
        question: str,
        context: str,
    ) -> str:

        prompt = f"""
You are an engineering knowledge assistant.

Answer the user's question using only the provided context.

Rules:
1. Use only information present in the context.
2. Do not invent facts.
3. If the context does not contain enough information,
   clearly say that you do not have enough information.
4. Give a concise and technically accurate answer.
5. When possible, refer to the provided source numbers.

Context:
{context}

Question:
{question}
"""

        response = self.client.models.generate_content(
            model=self.MODEL_NAME,
            contents=prompt,
        )

        return response.text