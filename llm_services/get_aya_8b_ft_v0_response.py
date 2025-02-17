import cohere 
import os
import asyncio
from dotenv import load_dotenv
load_dotenv()

co = cohere.Client(
  api_key=os.getenv("COHERE_API_KEY")
) 

async def get_aya_8b_ft_v0_response(user_prompt: str, temperature=0.3):
    response = co.chat(
        model="0171c0af-f40c-43a0-bace-e47ab2ea2eb0-ft",
        message="You are a helpful assistant that translates sentences from Bemba to English.",
        temperature=temperature,
        chat_history=[
            {"role": "User", "message": user_prompt}
        ],
    )
    response_content = response.text
    return response_content

get_aya_8b_ft_v0_response.__name__ = "aya_8b_ft_v0"

# test
# print(asyncio.run(get_aya_8b_ft_v0_response("Hello, how are you?")))