#!/usr/bin/env python
# coding: utf-8

# # Setup

# ## Stock Imports

# ### IDE Stuff

# #### Installing New Packages

# In[ ]:


# Jupyter Plugins that make things much nicer:
# * Collapsible_Headings
# * hide_code
# * jupyterlab-code-cell-collapser
# * jupyterlab_templates


# Depending on how many installations of Python you have on your system, doing a simple `conda install` or `pip install` will put the module in the wrong installation.  This makes sure it lands in the installation corresponding to the current notebook.

# In[ ]:


# Code for installing packages
# !conda install --yes --prefix {sys.prefix} shapely
# !{sys.executable} -m pip install plotly


# #### Notebook Customization

# This will change the fonts.  I find the rendered mardown heading fonts (H1, H2, H3, etc) to be too similar.  This will alternate between italicized and normal fonts for headings.  Additionally, it adds a small indent that grows with heading level.  H1 is centered.

# In[ ]:


import IPython


# In[ ]:


css_str = """
<link rel="preconnect" href="https://fonts.gstatic.com">

<link href="https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@300;400;600&family=Playfair+Display+SC&display=swap" rel="stylesheet">
<link rel="stylesheet" href="https://fonts.googleapis.com/css?family=Lora">
<link href="https://fonts.googleapis.com/css2?family=IM+Fell+Double+Pica:ital@1&display=swap" rel="stylesheet">
    <style>
h1 { color: #7c795d; font-family: 'Playfair Display SC', serif; text-indent: 00px; text-align: center;}
h2 { color: #7c795d; font-family: 'Lora', serif;                text-indent: 00px; text-align: left; }
h3 { color: #7c795d; font-family: 'IM Fell Double Pica', serif; text-indent: 15px; text-align: left; }
h4 { color: #7c795d; font-family: 'Lora', Arial, serif;         text-indent: 30px; text-align: left}
h5 { color: #71a832; font-family: 'IM Fell Double Pica', serif; text-indent: 45px; text-align: left}

"""
IPython.display.HTML(css_str)


# Allows for changes in packages to be detected and immediately incorporated into the present notebook without resetting the kernel.

# In[ ]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# By default, Jupyter returns the last expression (and won't return this if it is a statement).  Sometimes we want different behavior that is more similar to Mathematica

# In[ ]:


from IPython.core.interactiveshell import InteractiveShell

# pretty print only the last output of the cell
InteractiveShell.ast_node_interactivity = 'last_expr' # ['all', 'last', 'last_expr', 'none', 'last_expr_or_assign']


# In[ ]:


x = 2
y = 3


# In[ ]:


del x, y


# In[ ]:


x = 2


# ### Python Libraries

# Standard Python imports

# In[ ]:


import os, sys, time, glob


# It is a good idea to record the starting working directory before it gets changed around.  Note that this can be problematic depending on how you open the notebook, but it works most of the time.

# In[ ]:


baseDir = os.getcwd()
baseDir


# It is also nice to know if what is running is a notebook, or a python script generated from the notebook.

# In[ ]:


mainQ = (__name__ == '__main__')
if mainQ:
    print("This is the main file")


# JupyterLab doesn't support navigation to other drives.  This is a handy trick to make folders in other drives "appear" as if they're local.  It even works on network shares.  
# 
# PowerShell Command to Map network drives:
# ```powershell
# New-Item -ItemType SymbolicLink -Path "c:\users\brianedw\group_share" -Target "\\158.130.53.35\_Group Share"
# ```
# 

# ### Notebook Interactivity

# Adds a nice progress bar for visualizing a loop iterator.  See snippets.

# In[ ]:


from tqdm import tqdm


# Useful for monitoring a calculation's progress.

# In[ ]:


from IPython.display import clear_output


# In[ ]:


for i in range(5):
    print(i)
    time.sleep(0.1)
    clear_output(wait=True)


# A nice sound to play when long calculations are completed.

# In[ ]:


import winsound

def soundDone():
    soundfile = "C:/Windows/Media/ring01.wav"
    winsound.PlaySound(soundfile, winsound.SND_FILENAME | winsound.SND_ASYNC)


# ### Functional Programming

# It is very common in data analysis to run data through a series of transformations, often called a "pipe".  The advantage of this is the arguments are contained with the functions and it is more readable (once you get used to it!).  For instance, in `Mathematica`, 
# 
# `H(#, 4)& @ G(#,3)& @ F(#,2)& @ 1  =>  H(G(F(1,2),3),4).`
# 
# In the former, it is clear that `3` belongs to `G`.  In the latter, you need to count parenthesis.  This typically makes use of "lambda functions" or "pure functions".  As a primarily Object Oriented Programming (OOP) language, Python doesn't natively support much of this Functional paradigm.  However, it does treat functions as objects which can be manipulated.  Given the utility of Functional Programming, there are several packages that attempt to bring it into the language, each with varying success.

# In[ ]:


# from pipetools import where, X, pipe
# 10 > (pipe | range | where(X % 2) | sum)


# In[ ]:


# from pipey import Pipeable

# Print = Pipeable(print)
# @Pipeable
# def add(a,b): return a + b
# @Pipeable
# def sqr(b): return b*b

# np.array([3, 4]) >> sqr >> add(1000)


# The `toolz` library has a lot of great functions for performing common operations on iterables, functions, and dictionaries.

# In[ ]:


# from toolz.itertoolz import ()
from toolz.functoolz import (curry, pipe, thread_first)
# from toolz.dicttoolz import ()


# In[ ]:


@curry
def add(x, y): return x + y
@curry
def pow(x, y): return x**y
thread_first(1, add(y=4), pow(y=2))  # pow(add(1, 4), 2)


# In[ ]:


from mini_lambda import InputVar, as_function
_ = as_function
X = InputVar('X')


# In[ ]:


_(X+3)(10)


# In[ ]:


thread_first(1, add(y=4), _(pow(x=2, y=X)))  # pow(2, add(1, 4))


# In[ ]:


thread_first(1, _(X+4), _(2**X))  # pow(add(1, 4), 2)


# ### Scientific Programming

# In[ ]:


from math import *
deg = radians(1)    # so that we can refer to 90*deg
I = 1j              # potentially neater imaginary nomenclature.


# In[ ]:


import numpy as np  # Does high performance dense array operations
np.set_printoptions(edgeitems=30, linewidth=100000,
                    formatter=dict(float=lambda x: "%.3g" % x))
import scipy as sp
import pandas as pd
import PIL 


# In[ ]:


import skimage


# In[ ]:


# Python function compilization.  Makes things very fast.  Function must only include Numpy and basic Python.  No custom classes.
import numba
from numba import njit


# In[ ]:


import sympy as sp
# sp.init_printing(pretty_print=True)
# sp.init_printing(pretty_print=False)


# ### Plotting

# In[ ]:


import bokeh
from bokeh.io import output_notebook
from bokeh.plotting import figure, show
output_notebook()
bokeh.io.curdoc().theme = 'dark_minimal'


# ## Custom Imports

# In[ ]:


from UtilityMath import (plotComplexArray, 
                         RandomComplexCircularMatrix, RandomComplexGaussianMatrix,
                         PolarPlot,
                         RescaleToUnitary,
                         ReIm,
                         MatrixSqError, MatrixError, MatrixErrorNormalized)
from Logger import Logger


# ## Code Snippets

# ### Cell Updating

# In[ ]:


if mainQ and False:
    for f in range(10):
        clear_output(wait=True)
        print(f)
        time.sleep(0.5)


# In[ ]:


if mainQ and False:
    for i in tqdm(range(100000000)):
        pass


# ### Bokeh Simple Line Plot

# In[ ]:


ts = np.linspace(0, 4, num=300)


# In[ ]:


xs = 5.0 * ts


# In[ ]:


ys = -9.8*ts**2 + 50*ts + 0


# In[ ]:


fig = figure(x_range=(min(xs), max(xs)), y_range=(min(ys), max(ys)), 
             plot_width=800, plot_height=400,
             title='Trajectory')
fig.xaxis.axis_label = "x (m)"
fig.yaxis.axis_label = "y (m)"
fig.line(x=xs, y=ys)
if mainQ: show(fig)


# In[ ]:


del ts, xs, ys, fig


# # Work

# In[ ]:




