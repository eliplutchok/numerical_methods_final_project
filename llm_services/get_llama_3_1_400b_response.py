from openai import OpenAI
import os
import asyncio
from dotenv import load_dotenv
load_dotenv()

client = OpenAI(
  base_url = "https://integrate.api.nvidia.com/v1",
  api_key = os.getenv("NVIDIA_API_KEY")
)



async def get_llama_3_1_400b_response(user_prompt: str, temperature=0.2):
    completion = client.chat.completions.create(
        model="meta/llama-3.1-405b-instruct",
        messages=[{"role": "user","content": user_prompt}],
        temperature=temperature,
        top_p=0.7,
        max_tokens=2024
    )
    return completion.choices[0].message.content

get_llama_3_1_400b_response.__name__ = "llama_3_1_400b"

# test
# if __name__ == "__main__":
#     print(asyncio.run(get_llama_3_1_400b_response("Hello, how are you?")))