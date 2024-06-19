#!/usr/bin/env python
# coding: utf-8

# ## Tutorial: Running Solution Optimization
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
# **Objectives:**
# 
# * In this tutorial, we will implement a solution optimization pipeline.
# 
# **Requirements:**
# 
# * You need to have an OpenAI API key to run this tutorial. This should be set as an environment variable as OPENAI_API_KEY.
# 



get_ipython().system('pip install textgrad # you might need to restart the notebook after installing textgrad')


import textgrad as tg
tg.set_backward_engine(tg.get_engine("gpt-4o"))

initial_solution = """To solve the equation 3x^2 - 7x + 2 = 0, we use the quadratic formula:
x = (-b ± √(b^2 - 4ac)) / 2a
a = 3, b = -7, c = 2
x = (7 ± √((-7)^2 + 4(3)(2))) / 6
x = (7 ± √73) / 6
The solutions are:
x1 = (7 + √73)
x2 = (7 - √73)"""

solution = tg.Variable(initial_solution,
                       requires_grad=True,
                       role_description="solution to the math question")

loss_system_prompt = tg.Variable("""You will evaluate a solution to a math question. 
Do not attempt to solve it yourself, do not give a solution, only identify errors. Be super concise.""",
                                 requires_grad=False,
                                 role_description="system prompt")
                              
loss_fn = tg.TextLoss(loss_system_prompt)
optimizer = tg.TGD([solution])




loss = loss_fn(solution)
loss




loss.backward()
optimizer.step()
print(solution.value)






