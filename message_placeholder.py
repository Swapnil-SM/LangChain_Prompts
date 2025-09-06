from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from dotenv import load_dotenv

load_dotenv()
# chat template
chat_template = ChatPromptTemplate([
    ('system','You are a helpful customer support agent'),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human','{query}')
])

chat_history = []
# load chat history
with open('chat_history.txt') as f:
    chat_history.extend(f.readlines())

print(chat_history)

# create prompt
user_query = input("Enter your question: ")
prompt = chat_template.invoke({'chat_history':chat_history, 'query':user_query})

print(prompt)

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)
response = model.invoke(prompt)

print("Assistant:", response.content)

# optional: save the new exchange into chat_history
chat_history.append(f"Human: {user_query}")
chat_history.append(f"Assistant: {response.content}")

# write updated history back to file
with open('chat_history.txt', 'w') as f:
    f.write("\n".join(chat_history))
