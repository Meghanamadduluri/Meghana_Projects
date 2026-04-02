import os
from google import genai


class Generator:
    def __init__(self):
        api_key = os.getenv("GOOGLE_API_KEY")
        self.client = genai.Client(api_key=api_key)
        self.model_id = "gemini-2.0-flash"
        #for m in self.client.models.list():
         #   print(m.name)

    def generate(self, query: str, context: list):
        context_text = "\n".join(context)

        prompt = f"""
        You are a Stripe API documentation assistant.

        Answer the question using ONLY the context below.
        Do not mention sources.
        Respond in plain paragraph form.
        Do not use bullet points, markdown, or special formatting.
        
        Context:
        {context_text}
        
        Question:
        {query}
        
        Answer:
        """

        response = self.client.models.generate_content(
            model="gemini-2.5-flash", 
            contents=prompt
        )
        return response.text