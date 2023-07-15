from langchain import HuggingFaceHub
from langchain import PromptTemplate, LLMChain
import chainlit as cl
HUGGINGFACEHUB_API_TOKEN = "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"
repo_id = "tiiuae/falcon-7b-instruct"
llm = HuggingFaceHub(huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN, 
                     repo_id=repo_id, 
                     model_kwargs={"temperature":0.7, "max_new_tokens":500})


template = """
You are a helpful AI assistant and provide the answer for the question asked politely.

{question}
"""

@cl.langchain_factory(use_async=False)
def factory():
    prompt = PromptTemplate(template=template, input_variables=["question"])
    llm_chain = LLMChain(prompt=prompt, llm=llm)

    return llm_chain

