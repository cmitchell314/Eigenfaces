import math
import random
from matplotlib import pyplot as plt 
import numpy as np
import os 
import cv2
import tkinter as tk
from tkinter import PhotoImage
from PIL import Image, ImageTk

"""
Aims:
- decent user interface
- user can scan their face, this is saved and identified as theirs
- user can save their face (MAYBE, just add it by default)
- user can remove their faces from the set (MAYBE)
- best attempt at identification displayed
- some measurement of confidence/proximity displayed (MAYBE)
"""

class App:
    def __init__(self, root, source=0):
        # INSTANCE STATE VARIABLES
        self.user_id = None # name of current user 
        self.log_next_img = False

        # INITIALIZATION of algorithms
        initialize()

        # SETUP of GUI
        self.root = root
        self.root.title('Demo Facial Recognition System')
        width = 1000
        height = (192 * 2) + 240
        self.root.geometry(f'{width}x{height}')

        self.video = cv2.VideoCapture(source)

        # top frame - banner, title, version, etc.
        top_frame = tk.Frame(self.root, width=width, height=100)
        top_frame.configure(bg='black')
        top_frame.pack(side='top')

        # bottom frame - current id, save image button, etc.
        bottom_frame = tk.Frame(self.root, width=width, height=100)
        bottom_frame.configure(highlightbackground='black', highlightthickness=3)
        bottom_frame.pack(side='bottom')

        # center frame - left, center, right panels
        center_frame = tk.Frame(self.root, width=width, height=root.winfo_height() - 200)
        center_frame.configure(highlightbackground='blue', highlightthickness=3)

        # left panel - see existing saved faces
        left_panel = tk.Frame(center_frame, width=300, height=height - 200)
        left_panel.configure(highlightbackground='red', highlightthickness=3)
        left_panel.pack(side='left')

        # center panel - see current image
        self.center_panel = tk.Frame(center_frame, width=width - 500, height=height - 200)
        self.center_panel.pack(side='left')
        self.center_panel.configure(highlightbackground='gray', highlightthickness=10)

        # right panel - closest match, guessed name, confidence (?), etc.
        right_panel = tk.Frame(center_frame, width=200, height=height - 200)
        right_panel.configure(highlightbackground='red', highlightthickness=3)
        right_panel.pack(side='left')

        center_frame.pack(side='top')

        # img feed
        self.img_can = tk.Canvas(self.center_panel, width=168*2, height=192 * 2)
        self.img_can.pack(anchor=tk.CENTER)

        # bottom frame contents
        self.id_tag = tk.Label(bottom_frame, text='Current User ID: ', font=('Calibri', 30))
        self.id_tag.pack(side='left', padx=10, pady=10)

        self.user_id_box = tk.Entry(bottom_frame, font=('Calibri', 30))
        self.user_id_box.pack(side='left', padx=10, pady=10)

        # self.save_img_button = tk.Button(bottom_frame, text='Save Image', font=('Calibri', 20), command=(lambda: self.log_next_img=True))
        self.clear_all_button = tk.Button(bottom_frame, text='Reset All', font=('Calibri', 20), command= erase_all_user_faces)
        self.clear_all_button.pack(side='right', padx=10, pady=10)
        self.save_img_button = tk.Button(bottom_frame, text='Save Image', font=('Calibri', 20), command= self.set_grab_next)
        self.save_img_button.pack(side='right', padx=10, pady=10)

        self.update()

    def set_grab_next(self):
        self.log_next_img = True

    def update(self):
        ### Update camera panel
        ret, frame = self.video.read()
        frame = frame[frame.shape[0]//2 - 192:frame.shape[0]//2 + 192, frame.shape[1]//2 - 168:frame.shape[1]//2 + 168]

        if ret:
            # print('got frame')
            # convert to PIL format
            photo = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(np.flip(frame, axis=1), cv2.COLOR_BGR2RGB)))
            self.img_can.create_image(0, 0, image=photo, anchor=tk.NW)
            self.img_can.image = photo

        if (self.log_next_img):
            if ret:
                # convert frame to grayscale
                g_frm = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[::2, ::2]
                add_face(g_frm, self.user_id_box.get())
                # print(g_frm.shape)
                # plt.imshow(g_frm, cmap='gray')
                # plt.show()
            self.log_next_img = False


        # schedule next update
        self.root.after(20, self.update)

e_faces = np.array([])
mean_face = np.array([])
user_faces = np.array([])
user_labels = []

def initialize():
    global e_faces, mean_face, user_faces, user_labels

    try:
        e_faces = np.load('e_faces_set_all_faces.npy')
        print('Loaded eigenface dataset. Shape: ', e_faces.shape)
    except:
        print('Cannot load eigenface dataset. Please retrain model.')

    try:
        mean_face = np.load('mean_face_all_faces.npy')
        print('Loaded mean face. Shape: ', mean_face.shape)
    except:
        print('Cannot load mean face. Please retrain model.')

    try:
        user_faces = np.load('user_faces_app.npy')
        print('Loaded user faces. Shape: ', user_faces.shape)
    except:
        user_faces = np.array([])
        np.save('user_faces_app.npy', user_faces)

    # get user id names from user_labels.csv
    user_labels = []
    try:
        with open('user_labels.csv', 'r') as f:
            for line in f:
                user_labels.append(line.strip())
    except:
        with open('user_labels.csv', 'w') as f:
            pass
    print('Loaded user labels. Found ', len(user_labels), ' labels.')


def get_closest_face(img):
    global e_faces, mean_face, user_faces, user_labels

    if img.shape[0] != 192 or img.shape[1] != 168:
        return None
    
    face_eform = np.matmul(e_faces.T, img.flatten() - mean_face)

    # find closest face - use KNN

    s

    
    return None

def add_face(img, user_name):
    global e_faces, mean_face, user_faces, user_labels

    if img.shape[0] == 192 and img.shape[1] == 168 and user_name != '':
        face_eform = np.matmul(e_faces.T, img.flatten() - mean_face)
        if user_faces.shape[0] == 0:
            user_faces = np.array([img.flatten()])
        else:
            user_faces = np.append(user_faces, [img.flatten()], axis=0)
        np.save('user_faces_app.npy', user_faces)

        if user_name in user_labels:
            clones = [nm for nm in user_labels if nm.startswith(user_name)]
            user_name += '_' + str(len(clones))

        user_labels.append(user_name)
        with open('user_labels.csv', 'a') as f:
            f.write(user_name + '\n')

        print('Adding face ', user_faces.shape[0], ' with label', user_name, 'to user faces.')
        plt.imshow((np.matmul(e_faces, face_eform) + mean_face).reshape(192, 168), cmap='gray')
        plt.show()
        return True

    return False


def remove_face(name):
    # pass
    return None

def erase_all_user_faces():
    global user_faces, user_labels
    user_faces = np.array([])
    np.save('user_faces_app.npy', user_faces)
    print('deleted all user faces')

    user_labels = []
    with open('user_labels.csv', 'w') as f:
        pass


# making root GUI
root = tk.Tk()

app = App(root)

root.mainloop()