# services/openai_service.py

import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

openai = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY_FOR_O1")
)

async def get_o1_preview_response(user_prompt: str, model="o1-preview"):
    messages = [
        {"role": "user", "content": user_prompt}
    ]
    response = openai.chat.completions.create(
        model=model,
        messages=messages
    )
    response_content = response.choices[0].message.content
    return response_content

get_o1_preview_response.__name__ = "o1_preview"