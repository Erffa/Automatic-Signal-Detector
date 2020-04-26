import numpy as np
import matplotlib.pyplot as plt

myvar = "Hello in there"

def myfunction(n):
  return 3*n+1
  
class MyClass:
  def __init__(self, arg):
    self.arg = arg
    return
    
  def display(self):
    print("Here is the class")
    print("You gave the arg {}".format(self.arg))
    return
