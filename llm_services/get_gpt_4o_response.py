# services/openai_service.py

import os
from dotenv import load_dotenv
from openai import OpenAI
from typing import Any
load_dotenv()

openai = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

async def get_gpt_4o_response(user_prompt: str, temperature=0.3):
    model="gpt-4o-2024-08-06"
    messages = [
        {"role": "system", "content": "You are a helpful assistant that translates sentences from Bemba to English."},
        {"role": "user", "content": user_prompt}
    ]
    response = openai.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature
    )
    response_content = response.choices[0].message.content
    return response_content

get_gpt_4o_response.__name__ = "gpt_4o"
# async def get_gpt4o_structured_response(messages: list, response_schema, model="gpt-4o-2024-08-06"):
#     response = openai.beta.chat.completions.parse(
#         model=model,
#         messages=messages,
#         response_format=response_schema
#     )
#     response_content = response.choices[0].message.parsed.model_dump()
#     return response_content