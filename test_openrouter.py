import os
from dotenv import load_dotenv
from openai import OpenAI


# ============================================================
# OPENROUTER API TEST
# ============================================================

print("=" * 60)
print("OPENROUTER API TEST")
print("=" * 60)


# ============================================================
# 1. LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

openrouter_api_key = os.getenv("OPEN_ROUTER_API_KEY")

if not openrouter_api_key:
    print("\nERROR: OPENROUTER_API_KEY was not found!")
    print("Make sure your .env file contains:")
    print("OPENROUTER_API_KEY=sk-or-v1-xxxxxxxx")
    exit(1)

print("[OK] OPENROUTER_API_KEY found")


# ============================================================
# 2. CREATE OPENROUTER CLIENT
# ============================================================

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=openrouter_api_key,
)

print("[OK] OpenRouter client created")


# ============================================================
# 3. SEND TEST REQUEST
# ============================================================

print("[INFO] Sending request to Llama 3.1 8B...")

try:

    completion = client.chat.completions.create(
        model="meta-llama/llama-3.1-8b-instruct:free",
        messages=[
            {
                "role": "user",
                "content": "What is the capital of France?"
            }
        ],
        temperature=0,
        max_tokens=20
    )

    response = completion.choices[0].message.content

    print("\n" + "=" * 60)
    print("MODEL RESPONSE")
    print("=" * 60)

    print(response)

    print("\n" + "=" * 60)
    print("TEST SUCCESSFUL")
    print("=" * 60)

except Exception as e:

    print("\n" + "=" * 60)
    print("TEST FAILED")
    print("=" * 60)

    print(f"Error type: {type(e).__name__}")
    print(f"Error: {e}")