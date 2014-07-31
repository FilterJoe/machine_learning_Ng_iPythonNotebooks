# Andrew Ng's Spring 2014 Machine Learning Course Rewritten in Python

> Just let me see it! Preview these notebooks using NBViewer:
[ex1](http://nbviewer.ipython.org/github/FilterJoe/machine_learning_Ng_iPythonNotebooks/tree/master/ex1)
[ex2](http://nbviewer.ipython.org/github/FilterJoe/machine_learning_Ng_iPythonNotebooks/tree/master/ex2)
[ex3](http://nbviewer.ipython.org/github/FilterJoe/machine_learning_Ng_iPythonNotebooks/tree/master/ex3)

## Introduction

I took Andrew Ng's machine learning course on Coursera from March to May of 2014. It was a great intro to data science, using Octave, an open source Matlab clone. However, Python is my favorite language these days.

I wanted to learn the Python/numpy/scipy/matplotlib way of doing the same things. I started by slavishly imitating all exercises in Python, completing programming exercises ex1, ex2, ex3.

But Python is not Matlab or Octave, the PyCharm editor is not optimized for scientific computing, and the resulting code did not look all that pythonic. It was slow going. I couldn't even conviently embed plots with text output with my setup. There had to be a better way - and poking around I realized that iPython Notebook was it.

## Why iPython Notebook

iPython Notebook is a natural match for coursework, especially data science coursework, that combines math, coding, and graphical output. I can learn and use Python/numpy/scipy. I can quickly prototype and annotate my experiments. I can get output that is cleaner, clearer, and more flexible, including LaTex formatted equations and embedded plots. And the ability to rerun a small part of the code without having to start over from the beginning saves time.

With iPython Notebook, slavishly imitating course exercises does not make sense. The same material is covered, but formatted to take advantage of iPython's capabilities. I already wrote Python code for ex1, ex2, and ex3, so it's an opporuntity to learn iPython Notebook.

## A Note About the Honor Code

The Honor Code for Andrew Ng's Machine Learning course on Coursera requests to not show programming solutions to other students. The Spring 2014 session is over, but I assume the Honor Code applies.

However, the assignments and solutions for the course are in Octave (or Matlab, which is very similar).  I do not believe posting Python Code is particularly helpful to students trying to write Octave code. Translating to Python requires considerable understanding of the material. To translate back into Octave would require more work and understanding of the material than completing the assignments in Octave with the material given and the help available in the forums.

If any staff associated with this course believes this is a violation of the Honor Code, please contact me and I will remove the material immediately.

## How to Launch iPython Notebook

iPython notebook's default behavior does not allow inline plots, nor does it allow matplotlib's default behavior of keeping the active plot open so that you can draw it multiple times. Most exercises require this behavior. It can be enabled by starting iPython notebook as follows:

ipython notebook --InlineBackend.close_figures=False --IPKernelApp.matplotlib='inline'

I am not using pylab --inline, because: [No Pylab Thanks] (http://carreau.github.io/posts/10-No-PyLab-Thanks.ipynb.html)

## 
