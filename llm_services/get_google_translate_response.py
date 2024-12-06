# services/openai_service.py
import os
from dotenv import load_dotenv
load_dotenv()

import aiohttp
import asyncio

async def get_google_translate_response(text: str, temperature=None):
    """Translates text into English using the Google Translation REST API."""
    api_key = os.environ['GOOGLE_CLOUD_API_KEY']
    url = 'https://translation.googleapis.com/language/translate/v2'

    params = {
        'q': text,
        'target': 'en',
        'format': 'text',
        'key': api_key
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(url, data=params) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"Error: {response.status}, {error_text}")

            result = await response.json()
            translated_text = result['data']['translations'][0]['translatedText']
            return translated_text

get_google_translate_response.__name__ = "google_translate"

# Test
if __name__ == "__main__":
    async def main():
        translated_text = await get_google_translate_response("""
        A: Ba malonda babili bale cekina nga isitima lilifye bwino.
        B: Bushe teili isitima lilingile ukwima nombaline ukutwala abantu kumusumba?
        A: Emukwai nililine cipalile kwati kwaciba fye akabwafya akanono eko bacila lolekeshapo.
        B: Pantu kwena balandile ukuti lilife bwino abantu bakacelele nomba lelo lyaonaika shani?
        A: Baleti lya cikwatakofye ubwafya ubunono lelo nababombelapo mukwai.
        """)
        print(translated_text)

    asyncio.run(main())
