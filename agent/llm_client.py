import os

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

MODEL_NAME = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.8-flash"
)


def generate_response(user_prompt, system_instruction=None):

    if not isinstance(user_prompt, str) or not user_prompt.strip():
        raise ValueError("Prompt cannot be empty.")

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError("Gemini API key not found.")

    print("[1/3] Initializing Gemini client...", flush=True)

    client = genai.Client(
       http_options=types.HttpOptions(
    timeout=120000
)
    )

    try:

        print("[2/3] Sending request to Gemini...", flush=True)

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                max_output_tokens=256
            )
        )

        if not response.text:
            raise RuntimeError("Gemini returned an empty response.")

        print("[3/3] Response received!", flush=True)

        return response.text.strip()

    except Exception as error:

        raise RuntimeError(
            f"Gemini API request failed: "
            f"{type(error).__name__}: {error}"
        ) from error