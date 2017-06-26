# These are called modules
import random
import sys
import os

# Hello world is just one line of code
# print() outputs data to the screen
print("Hello World")

'''
This is a multi-line comment
'''

# A variable is a place to store values
# Its name is like a label for that value
name = "Derek"
print(name)

# A variable name can contain letters, numbers, or _
# but can't start with a number

# There are 5 data types Numbers, Strings, List, Tuple, Dictionary
# You can store any of them in the same variable

name = 15
print(name)

# The arithmetic operators +, -, *, /, %, **, //
# ** Exponential calculation
# // Floor Division
print("5 + 2 =", 5+2)
print("5 - 2 =", 5-2)
print("5 * 2 =", 5*2)
print("5 / 2 =", 5/2)
print("5 % 2 =", 5%2)
print("5 ** 2 =", 5**2)
print("5 // 2 =", 5//2)

# Order of Operation states * and / is performed before + and -

print("1 + 2 - 3 * 2 =", 1 + 2 - 3 * 2)
print("(1 + 2 - 3) * 2 =", (1 + 2 - 3) * 2)

# A string is a string of characters surrounded by " or '
# If you must use a " or ' between the same quote escape it with \
quote = "\"Always remember your unique,"

# A multi-line quote
multi_line_quote = ''' just
like everyone else" '''

print(quote + multi_line_quote)

# To embed a string in output use %s
print("%s %s %s" % ('I like the quote', quote, multi_line_quote))

# To keep from printing newlines use end=""
print("I don't like ",end="")
print("newlines")

# You can print a string multiple times with *
print('\n' * 5)