#!/usr/bin/env python
# coding: utf-8



# Some utils to read images

import io
from PIL import Image




import textgrad as tg
from textgrad import get_engine
from textgrad.autograd import MultimodalLLMCall
from textgrad.loss import ImageQALoss




import os

# Assuming the OpenAI API key is stored in an environment variable named 'OPENAI_API_KEY'
openai_api_key = os.getenv('OPENAI_API_KEY')
assert openai_api_key is not None and len(openai_api_key) > 0, "Please set the OPENAI_API_KEY environment variable"




tg.set_backward_engine("gpt-4o", override=True)


# # Simply answering questions about images



# import httpx

# image_url = "https://upload.wikimedia.org/wikipedia/commons/a/a7/Camponotus_flavomarginatus_ant.jpg"
# image_data = httpx.get(image_url).content

# Import necessary library
import httpx

# Path to the local image
image_path = ".assets/176.png"

# Read the local image file in binary mode
with open(image_path, 'rb') as file:
    image_data = file.read()

# Print the first few bytes of the image data to verify (optional)
print(image_data[:10])




image_variable = tg.Variable(image_data, role_description="image to answer a question about", requires_grad=False)

# MathVista-176, ground truth = (D) 2
question_text = """
Hint: Please answer the question and provide the correct option letter, e.g., A, B, C, D, at the end. 
Question: Find $RS$ if $\triangle QRS$ is an equilateral triangle. 
Choices: (A) 0.5 (B) 1 (C) 1.5 (D) 2
"""
question_text = question_text.strip()
question_text




question_variable = tg.Variable(question_text, role_description="question", requires_grad=False)
question_variable




response = MultimodalLLMCall("gpt-4o")([image_variable, question_variable])
response




Image.open(io.BytesIO(image_data))




loss_fn = ImageQALoss(
    evaluation_instruction="Does this seem like a complete and good answer for the image? Criticize heavily.",
    # engine="claude-3-5-sonnet-20240620"
    engine="gpt-4o",
)
loss_fn




loss = loss_fn(question=question_variable, image=image_variable, response=response)
loss




optimizer = tg.TGD(parameters=[response])
optimizer




loss.backward()
optimizer.step()
print(response.value)






