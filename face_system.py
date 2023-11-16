import math
import random
from matplotlib import pyplot as plt 
import numpy as np
import os 
import cv2
import tkinter as tk
from tkinter import PhotoImage

"""
Aims:
- decent user interface
- user can scan their face, this is saved and identified as theirs
- user can save their face (MAYBE, just add it by default)
- user can remove their faces from the set (MAYBE)
- best attempt at identification displayed
- some measurement of confidence/proximity displayed (MAYBE)
"""

e_faces = np.array([])

def initialize():
    # pass
    return None

def get_closest_face(img):
    # pass
    return None

def add_face(img):
    # pass
    return None

def remove_face(name):
    # pass
    return None


# making root GUI
root = tk.Tk()
root.title('Demo Facial Recognition System')
width = 1000
height = (192 * 2) + 240
root.geometry(f'{width}x{height}')

# top frame - banner, title, version, etc.
top_frame = tk.Frame(root, width=width, height=100)
top_frame.configure(bg='black')
top_frame.pack(side='top')

# bottom frame - current id, save image button, etc.
bottom_frame = tk.Frame(root, width=width, height=100)
bottom_frame.configure(highlightbackground='black', highlightthickness=3)
bottom_frame.pack(side='bottom')

# center frame - left, center, right panels
center_frame = tk.Frame(root, width=width, height=root.winfo_height() - 200)
center_frame.configure(highlightbackground='blue', highlightthickness=3)

# left panel - see existing saved faces
left_panel = tk.Frame(center_frame, width=300, height=height - 200)
left_panel.configure(highlightbackground='red', highlightthickness=3)
left_panel.pack(side='left')

# center panel - see current image
center_panel = tk.Frame(center_frame, width=width - 500, height=height - 200)
center_panel.pack(side='left')
center_panel.configure(highlightbackground='green', highlightthickness=3)

# right panel - closest match, guessed name, confidence (?), etc.
right_panel = tk.Frame(center_frame, width=200, height=height - 200)
right_panel.configure(highlightbackground='red', highlightthickness=3)
right_panel.pack(side='left')

center_frame.pack(side='top')


# # banner at top of root with blue background
# banner = tk.Label(root, bg='blue', width=width, height=100)
# banner.pack()

root.mainloop()