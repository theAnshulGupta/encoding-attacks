from dotenv import load_dotenv
load_dotenv()
import os
import anthropic
from anthropic import Anthropic

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

messages = [
    {
        "role": "user",
        "content": ""
    }
]

out = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        messages=messages,
        max_tokens=1000
    )
print(out)