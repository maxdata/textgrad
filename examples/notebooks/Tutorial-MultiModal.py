#!/usr/bin/env python
# coding: utf-8

# # TextGrad Tutorials: MultiModal Optimization
# 
# ![TextGrad](https://github.com/vinid/data/blob/master/logo_full.png?raw=true)
# 
# An autograd engine -- for textual gradients!
# 
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zou-group/TextGrad/blob/main/examples/notebooks/Prompt-Optimization.ipynb)
# [![GitHub license](https://img.shields.io/badge/License-MIT-blue.svg)](https://lbesson.mit-license.org/)
# [![Arxiv](https://img.shields.io/badge/arXiv-2406.07496-B31B1B.svg)](https://arxiv.org/abs/2406.07496)
# [![Documentation Status](https://readthedocs.org/projects/textgrad/badge/?version=latest)](https://textgrad.readthedocs.io/en/latest/?badge=latest)
# [![PyPI - Python Version](https://img.shields.io/pypi/pyversions/textgrad)](https://pypi.org/project/textgrad/)
# [![PyPI](https://img.shields.io/pypi/v/textgrad)](https://pypi.org/project/textgrad/)
# 
# **Objectives for this tutorial:**
# 
# * Introduce you to multimodal optimization with TextGrad
# 
# **Requirements:**
# 
# * You need to have an OpenAI API key to run this tutorial. This should be set as an environment variable as OPENAI_API_KEY.
# 



# Some utils to read images

import io
from PIL import Image




import textgrad as tg

# differently from the past tutorials, we now need a multimodal LLM call instead of a standard one!
from textgrad.autograd import MultimodalLLMCall
from textgrad.loss import ImageQALoss




tg.set_backward_engine("gpt-4o")


# # Simply answering questions about images

# We now downlaod an image from the web.



import httpx

image_url = "https://upload.wikimedia.org/wikipedia/commons/a/a7/Camponotus_flavomarginatus_ant.jpg"
image_data = httpx.get(image_url).content


# As usual, in TextGrad we now have to transform our object of interest into a Variable object. In the previous tutorials, we were doing this with text data, now we are going to do this with Images.



image_variable = tg.Variable(image_data, role_description="image to answer a question about", requires_grad=False)


# Let's now ask as question!



question_variable = tg.Variable("What do you see in this image?", role_description="question", requires_grad=False)
response = MultimodalLLMCall("gpt-4o")([image_variable, question_variable])
response




Image.open(io.BytesIO(image_data))




loss_fn = ImageQALoss(
    evaluation_instruction="Does this seem like a complete and good answer for the image? Criticize. Do not provide a new answer.",
    engine="gpt-4o"
)
loss = loss_fn(question=question_variable, image=image_variable, response=response)
loss




optimizer = tg.TGD(parameters=[response])
loss.backward()
optimizer.step()
print(response.value)











