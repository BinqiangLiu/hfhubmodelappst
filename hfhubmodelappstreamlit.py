#可正常运行！
#195070887@QQ.com
#不会反复发送登录信息到邮箱

#https://python.langchain.com/docs/integrations/llms/huggingface_hub

# get a token: https://huggingface.co/docs/api-inference/quicktour#get-your-api-token

#from getpass import getpass

#HUGGINGFACEHUB_API_TOKEN = getpass()

from langchain import HuggingFaceHub
from langchain import PromptTemplate, LLMChain
import streamlit as st
from huggingface_hub import InferenceClient
from langchain import HuggingFaceHub
import requests# Internal usage
import os
from dotenv import load_dotenv
from time import sleep
import streamlit as st

load_dotenv()
#yourHFtoken = os.getenv("HUGGINGFACEHUB_API_TOKEN")
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
#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

#question = "Who won the FIFA World Cup in the year 1994? "
question = st.text_input("Enter your question:")

template = """Question: {question}

Answer: Let's think step by step."""

prompt = PromptTemplate(template=template, input_variables=["question"])

#repo_id = "google/flan-t5-xxl"  # See https://huggingface.co/models?pipeline_tag=text-generation&sort=downloads for some other options
#repo_id = "HuggingFaceH4/starchat-beta" #能够运行，但app会自动天添加一些完全无关的问题并进行回答？
#repo_id = "meta-llama/Llama-2-70b-chat-hf" #Visit https://huggingface.co/meta-llama/Llama-2-70b-chat-hf to ask for access
#repo_id = "tiiuae/falcon-7b" #这个LLM模型似乎是将问题逐字拆解进行QA问答？？？
#https://huggingface.co/models位于huggingface的模型LLM
#repo_id = "databricks/dolly-v2-3b"
repo_id = "mosaicml/mpt-7b"
#https://huggingface.co/blog/llama2#why-llama-2介绍了用于QA Chat的一些模型

llm = HuggingFaceHub(repo_id=repo_id,
                     model_kwargs={"min_length":100,
                                   "max_new_tokens":1024,
                                   "temperature":0.2,
#                                   "top_k":50,
#                                   "top_p":0.95, "eos_token_id":49155})
                                   })

llm_chain = LLMChain(prompt=prompt, llm=llm)
response=llm_chain.run(question)

st.write("Your question:\n"+question)
print("Your question:\n"+question)

st.write(response)
print(llm_chain.run(question))
