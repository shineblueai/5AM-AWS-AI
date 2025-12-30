# day_25.py
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, AIMessage

# Simulate a chatbot (no real API call for classroom)
def simple_chatbot(user_input):
    # In real use: use OpenAI chat model via LangChain
    return f"You said: '{user_input}'. This is a simulated response."

# Example interaction
user_msg = "Hello, how are you?"
response = simple_chatbot(user_msg)
print("User:", user_msg)
print("Bot:", response)

# Using LangChain prompt template (structure only)
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{user_input}")
])

messages = prompt.format_messages(user_input="What is AI?")
print("\nFormatted prompt:")
for msg in messages:
    print(msg)