import torch
import transformers
from transformers import AutoTokenizer, pipeline
import gradio as gr
import os

token = os.getenv('HF_TOKEN')

model = "meta-llama/Llama-2-7b-chat-hf"

tokenizer = AutoTokenizer.from_pretrained(model, token=token)

llama_pipeline = pipeline(
    "text-generation",
    model=model,
    torch_dtype = torch.bfloat16,
    device_map="auto"
)

BOS = "<s>"
EOS = "</s>"
BINS = "[INST] "
EINS = " [/INST]"
BSYS = "<<SYS>>\n"
ESYS = "\n<</SYS>>\n\n"

SYSTEM_PROMPT = BOS + BINS + BSYS + """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content.
Please ensure that your responses are socially unbiased and positive in nature.
If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct.
If you don't know the answer to a question, just say you don't know, please don't share false information.""" + ESYS

def message_format(msg: str, history: list, history_lim: int = 5):
  history = history[-max(len(history), history_lim):]

  if len(history) == 0:
    return SYSTEM_PROMPT + f"{msg} {EINS}"

  # history is list of (user_query, model_response)
  query = SYSTEM_PROMPT + f"{history[0][0]} {EINS} {history[0][1]} {EOS}"

  for user_query, model_response in history[1:]:
    query += f"{BOS}{BINS} {user_query} {EINS} {model_response} {EOS}"

  query += f"{BOS}{BINS} {msg} {EINS}"

  return query

def response(msg: str, history: list):
  query = message_format(msg, history)

  response = ""

  sequences = llama_pipeline(
      query,
      do_sample=True, #randomly sample from the most likely tokens for diversity in generated text.
      top_k=10, #consider the top 10 likely tokens at each step
      num_return_sequences=1, # return the most likely answer at last generation step.
      eos_token_id=tokenizer.eos_token_id, # when reaching end-of-sentence token, it will stop generating
      max_length=1024 # set the max length if the answers is too long
  )

  generated_text = sequences[0]["generated_text"]
  response = generated_text[len(query):].strip() # removing prompt

  print(f"AI Agent: {response}")
  return response


app = gr.ChatInterface(fn=response, examples=["Hello", 
                                              "Bonjour", 
                                              "Xin chào"],
                      title="Llama 2 Chat")
app.launch(share=True)