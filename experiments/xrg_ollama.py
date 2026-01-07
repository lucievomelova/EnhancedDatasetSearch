# README
# 1) Install dependencies:
# pip install ollama
# 2) Connect to gpu server and tunnel port, change username from skoda:
# ssh skoda@xrg13.ms.mff.cuni.cz -p 1302 -L 11434:localhost:11434
#

# https://github.com/ollama/ollama-python
from ollama import chat
from ollama import ChatResponse

# We need to have ollama available at localhost port 11434.
response: ChatResponse = chat(model="llama3.2", messages=[
  {
    "role": "user",
    "content": "Why is the sky blue?",
  },
])

print(response["message"]["content"])