# This file containss some very basic code to illustrate how  to import 
# .py files from github in some Google Colab .ipynb file.

import numpy as np
import matplotlib.pyplot as plt

myvar = "Hello in there"

def myfunction(n):
  return -n**2+3*n+1
  
class MyClass:
  def __init__(self, arg):
    self.arg = arg
    return
    
  def display(self):
    print("Here is the class")
    print("You gave the arg {}".format(self.arg))
    return
