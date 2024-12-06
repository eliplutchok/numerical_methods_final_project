import os
from dotenv import load_dotenv
load_dotenv()

import google.generativeai as genai
import asyncio



async def get_gemini_response(prompt: str, temperature=None):
    """Generates content using the Google Gemini model."""
    api_key = os.environ["GOOGLE_CLOUD_API_KEY"]
    genai.configure(api_key=api_key)
    
    # Since the genai library is synchronous, use an executor to avoid blocking.
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(None, _generate_content, prompt, temperature)
    return response

def _generate_content(prompt, temperature=0.5):
    generation_config = {
        "max_output_tokens": 8192,
        "temperature": temperature,
        "top_p": 0.95,
}
    model = genai.GenerativeModel("gemini-1.5-pro")
    response = model.generate_content(
        prompt, 
        generation_config=generation_config
    )
    return response.text

get_gemini_response.__name__ = "gemini_1_5_pro"

# Test
if __name__ == "__main__":
    async def main():
        prompt = ("""
        Translate the following text from the Bemba language to English:
        A: Ba malonda babili bale cekina nga isitima lilifye bwino.
        B: Bushe teili isitima lilingile ukwima nombaline ukutwala abantu kumusumba?
        A: Emukwai nililine cipalile kwati kwaciba fye akabwafya akanono eko bacila lolekeshapo.
        B: Pantu kwena balandile ukuti lilife bwino abantu bakacelele nomba lelo lyaonaika shani?
        A: Baleti lya cikwatakofye ubwafya ubunono lelo nababombelapo mukwai.
        """)
        result = await get_gemini_response(prompt, temperature=1.9)
        print(result)
    asyncio.run(main())
