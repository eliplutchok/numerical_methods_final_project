import cohere 
import os
import asyncio
from dotenv import load_dotenv
load_dotenv()

co = cohere.Client(
  api_key=os.getenv("COHERE_API_KEY")
) 

async def get_aya_8b_ft_v1_response(user_prompt: str, temperature=0.3):
    response = co.chat(
        model="d6d43987-f99b-4b2e-9a52-b0be49c248c1-ft",
        message="You are a helpful assistant that translates sentences from Bemba to English.",
        temperature=temperature,
        chat_history=[
            {"role": "User", "message": user_prompt}
        ],
    )
    response_content = response.text
    return response_content

get_aya_8b_ft_v1_response.__name__ = "aya_8b_ft_v1"

# test
# if __name__ == "__main__":
#     print(asyncio.run(get_aya_8b_ft_v1_response("Hello, how are you? what llm model are u, do u know?")))