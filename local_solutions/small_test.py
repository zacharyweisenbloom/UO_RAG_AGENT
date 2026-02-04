from openai import OpenAI

client = OpenAI(base_url="http://192.168.0.19:11434/v1", api_key="ollama")

tools = [{
  "type": "function",
  "function": {
    "name": "echo",
    "description": "Echo back text",
    "parameters": {
      "type": "object",
      "properties": {"text": {"type": "string"}},
      "required": ["text"]
    }
  }
}]

r = client.chat.completions.create(
    model="llama3.2",  # you have this pulled
    messages=[{"role":"user","content":"Call the echo tool with text='hi'"}],
    tools=tools
)

print("tool_calls:", r.choices[0].message.tool_calls)
print("assistant_content:", r.choices[0].message.content)

