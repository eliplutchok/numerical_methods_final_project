# services/anthropic_service.py

import os
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()

anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
anthropic_client = Anthropic(api_key=anthropic_api_key)

async def get_sonnet_3_point_5_response(user_prompt, temperature=0.3):
    response = anthropic_client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=1024,
        messages=[
            {"role": "user", "content": user_prompt}
        ],
        system="You are a helpful assistant that translates sentences from Bemba to English.",
        temperature=temperature
    )
    return response.content[0].text

get_sonnet_3_point_5_response.__name__ = "sonnet_3_point_5"