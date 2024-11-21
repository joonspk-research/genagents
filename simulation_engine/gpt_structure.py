import openai
import time
import base64
from typing import List, Union

from simulation_engine.settings import *

# Conditional import for GPT4All
gpt4all_instance = None
if LLM_VERS == "gpt4all":
  try:
    from gpt4all import GPT4All, Embed4All
  except ImportError:
    raise ImportError(
      "The 'gpt4all' library is not installed. Please install it with 'pip install gpt4all' to use GPT4All models."
    )
  try:
    gpt4all_instance = GPT4All(LLM_MODEL, n_ctx=28672)
    gpt4all_embeddings = Embed4All("nomic-embed-text-v1.5.f16.gguf")
  except Exception as e:
    raise RuntimeError(
      f"Failed to initialize GPT4All with the model '{LLM_MODEL}'. "
      "Ensure the model file exists and is correctly configured."
    ) from e
elif LLM_VERS.startswith("claude"):
  try:
    import anthropic
  except ImportError:
    raise ImportError(
      "The 'anthropic' library is not installed. Please install it with 'pip install anthropic' to use anthropic models."
    )
  try:
    anthropic_client = anthropic.Client(api_key=ANTHROPIC_API_KEY)
  except Exception as e:
    raise RuntimeError(
      f"Failed to initialize anthropic with the model '{LLM_VERS}'. "
      "Ensure the model file exists and is correctly configured."
    ) from e
    
openai.api_key = OPENAI_API_KEY

# ============================================================================
# #######################[SECTION 1: HELPER FUNCTIONS] #######################
# ============================================================================

def print_run_prompts(prompt_input: Union[str, List[str]], 
                      prompt: str, 
                      output: str) -> None:
  print (f"=== START =======================================================")
  print ("~~~ prompt_input    ----------------------------------------------")
  print (prompt_input, "\n")
  print ("~~~ prompt    ----------------------------------------------------")
  print (prompt, "\n")
  print ("~~~ output    ----------------------------------------------------")
  print (output, "\n") 
  print ("=== END ==========================================================")
  print ("\n\n\n")


def generate_prompt(prompt_input: Union[str, List[str]], 
                    prompt_lib_file: str) -> str:
  """Generate a prompt by replacing placeholders in a template file with 
     input."""
  if isinstance(prompt_input, str):
    prompt_input = [prompt_input]
  prompt_input = [str(i) for i in prompt_input]

  with open(prompt_lib_file, "r") as f:
    prompt = f.read()

  for count, input_text in enumerate(prompt_input):
    prompt = prompt.replace(f"!<INPUT {count}>!", input_text)

  if "<commentblockmarker>###</commentblockmarker>" in prompt:
    prompt = prompt.split("<commentblockmarker>###</commentblockmarker>")[1]

  return prompt.strip()


# ============================================================================
# ####################### [SECTION 2: SAFE GENERATE] #########################
# ============================================================================

def gpt_request(prompt: str, 
                model: str = "gpt-4o", 
                max_tokens: int = 1500) -> str:
  """Make a request to OpenAI or GPT4All based on LLM_VERS."""
  if model == "o1-preview": 
    try:
      client = openai.OpenAI(api_key=OPENAI_API_KEY)
      response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
      )
      return response.choices[0].message.content
    except Exception as e:
      return f"GENERATION ERROR: {str(e)}"
  if LLM_VERS == "gpt4all":
    try:
      response = gpt4all_instance.generate(
        prompt=prompt, 
        max_tokens=max_tokens,
        temp=0.7
      )

      return response
    except Exception as e:
      raise ImportError(
        f"GENERATION ERROR GPT4ALL: {str(e)}"
      )
  elif LLM_VERS.startswith("claude"):
    try:
      response = anthropic_client.messages.create(
        model=LLM_VERS, 
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.7,
      )

      return response.content[0].text
    except Exception as e:
      raise ImportError(
        f"GENERATION ERROR ANTTHROPIC: {str(e)}"
      )
    
  try:
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
      model=model,
      messages=[{"role": "user", "content": prompt}],
      max_tokens=max_tokens,
      temperature=0.7
    )
    return response.choices[0].message.content
  except Exception as e:
    raise ImportError(
      f"GENERATION ERROR OPENAI: {str(e)}"
    )
  

def gpt4_vision(messages: List[dict], max_tokens: int = 1500) -> str:
  """Make a request to OpenAI's GPT-4 Vision model."""
  try:
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
      model="gpt-4o",
      messages=messages,
      max_tokens=max_tokens,
      temperature=0.7
    )
    return response.choices[0].message.content
  except Exception as e:
    return f"GENERATION ERROR: {str(e)}"


def chat_safe_generate(prompt_input: Union[str, List[str]], 
                       prompt_lib_file: str,
                       gpt_version: str = "gpt-4o", 
                       repeat: int = 1,
                       fail_safe: str = "error", 
                       func_clean_up: callable = None,
                       verbose: bool = False,
                       max_tokens: int = 1500,
                       file_attachment: str = None,
                       file_type: str = None) -> tuple:
  """Generate a response using GPT models with error handling & retries."""
  if file_attachment and file_type:
    prompt = generate_prompt(prompt_input, prompt_lib_file)
    messages = [{"role": "user", "content": prompt}]

    if file_type.lower() == 'image':
      with open(file_attachment, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')
      messages.append({
        "role": "user",
        "content": [
            {"type": "text", "text": "Please refer to the attached image."},
            {"type": "image_url", "image_url": 
              {"url": f"data:image/jpeg;base64,{base64_image}"}}
        ]
      })
      response = gpt4_vision(messages, max_tokens)

    elif file_type.lower() == 'pdf':
      pdf_text = extract_text_from_pdf_file(file_attachment)
      pdf = f"PDF attachment in text-form:\n{pdf_text}\n\n"
      instruction = generate_prompt(prompt_input, prompt_lib_file)
      prompt = f"{pdf}"
      prompt += f"<End of the PDF attachment>\n=\nTask description:\n{instruction}"
      response = gpt_request(prompt, gpt_version, max_tokens)

  else:
    prompt = generate_prompt(prompt_input, prompt_lib_file)
    for i in range(repeat):
      response = gpt_request(prompt, model=gpt_version)
      if response != "GENERATION ERROR":
        break
      time.sleep(2**i)
    else:
      response = fail_safe

  if func_clean_up:
    response = func_clean_up(response, prompt=prompt)

  if verbose or DEBUG:
    print_run_prompts(prompt_input, prompt, response)

  return response, prompt, prompt_input, fail_safe


# ============================================================================
# #################### [SECTION 3: OTHER API FUNCTIONS] ######################
# ============================================================================

def get_text_embedding(text: str, 
                       model: str = "text-embedding-3-small") -> List[float]:
  """Generate an embedding for the given text using OpenAI's API."""
  if not isinstance(text, str) or not text.strip():
    raise ValueError("Input text must be a non-empty string.")

  text = text.replace("\n", " ").strip()

  if LLM_VERS == "gpt4allx":
    # Temporal solution to get the same embedding twice
    response = list(gpt4all_embeddings.embed(text=[text], dimensionality=768)[0]) + list(gpt4all_embeddings.embed(text=[text], dimensionality=768)[0]) 

  else:
    response = openai.embeddings.create(
      input=[text], model=model).data[0].embedding
    
  return response









