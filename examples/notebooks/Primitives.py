#!/usr/bin/env python
# coding: utf-8

# # TextGrad Tutorials: Primitives
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
# * Introduce you to the primitives in TextGrad
# 
# **Requirements:**
# 
# * You need to have an OpenAI API key to run this tutorial. This should be set as an environment variable as OPENAI_API_KEY.
# 



from textgrad.engine import get_engine
from textgrad import Variable
from textgrad.optimizer import TextualGradientDescent
from textgrad.loss import TextLoss
from dotenv import load_dotenv
load_dotenv()


# ## Introduction: Variable
# 
# Variables in TextGrad are the metaphorical equivalent of tensors in PyTorch. They are the primary data structure that you will interact with when using TextGrad. 
# 
# Variables keep track of gradients and manage the data.
# 
# Variables require two arguments (and there is an optional third one):
# 
# 1. `data`: The data that the variable will hold
# 2. `role_description`: A description of the role of the variable in the computation graph
# 3. `requires_grad`: (optional) A boolean flag that indicates whether the variable requires gradients



x = Variable("A sntence with a typo", role_description="The input sentence", requires_grad=True)




x.gradients


# ## Introduction: Engine
# 
# When we talk about the engine in TextGrad, we are referring to an LLM. The engine is an abstraction we use to interact with the model.



engine = get_engine("azure-gpt-4o")


# This object behaves like you would expect an LLM to behave: You can sample generation from the engine using the `generate` method. 



engine.generate("Hello how are you?")


# ## Introduction: Loss
# 
# Again, Loss in TextGrad is the metaphorical equivalent of loss in PyTorch. We use Losses in different form in TextGrad but for now we will focus on a simple TextLoss. TextLoss is going to evaluate the loss wrt a string.



system_prompt = Variable("Evaluate the correctness of this sentence", role_description="The system prompt")
loss = TextLoss(system_prompt, engine=engine)


# 

# ## Introduction: Optimizer
# 
# Keeping on the analogy with PyTorch, the optimizer in TextGrad is the object that will update the parameters of the model. In this case, the parameters are the variables that have `requires_grad` set to `True`.
# 
# **NOTE** This is a text optimizer! It will do all operations with text! 



optimizer = TextualGradientDescent(parameters=[x], engine=engine)


# ## Putting it all together
# 
# We can now put all the pieces together. We have a variable, an engine, a loss, and an optimizer. We can now run a single optimization step.



l = loss(x)
l.backward(engine)
optimizer.step()




x.value


# While here it is not going to be useful, we can also do multiple optimization steps in a loop! Do not forget to reset the gradients after each step!



optimizer.zero_grad()






