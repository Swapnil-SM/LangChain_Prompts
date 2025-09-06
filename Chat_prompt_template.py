from langchain_core.prompts import ChatPromptTemplate

chat_template = ChatPromptTemplate([    #list of tuples :The string "system" means: make a SystemMessage.
    ('system', 'You are a helpful {domain} expert'),
    ('human', 'Explain in simple terms, what is {topic}')
])

prompt = chat_template.invoke({'domain':'cricket','topic':'Dusra'})

print(prompt)