import math
import random
from matplotlib import pyplot as plt 
import numpy as np
import os 
import scipy.io as sio

def get_data():
    e_faces = np.load('e_faces_set_all_faces.npy')
    mean_face = np.load('mean_face_all_faces.npy')

    # get the test faces - split by their subject
    data = sio.loadmat('allFaces.mat')
    test_faces = []
    faces = data['faces']
    n = data['n'][0][0]
    m = data['m'][0][0]
    nfaces = np.ndarray.flatten(data['nfaces'])
    print(nfaces.shape[0], 'people with indiv img count:', nfaces)
    print('images:', len(faces), 'n:', n, 'm:', m)
    imgs = np.array(faces)
    test_faces = []
    for j in range(len(nfaces)-3, len(nfaces)):
        subj = []
        for i in range(nfaces[j]):
            img = np.array(imgs[:, i + np.sum(nfaces[:j])]).reshape(m, n).T
            subj.append(img)
        test_faces.append(subj)

    return e_faces, mean_face, test_faces

def get_closest_face(img, e_faces, mean_face, pop_faces, user_owner):    
    if img.shape[0] != 192 or img.shape[1] != 168:
        return None
    
    face_eform = np.matmul(e_faces.T, img.flatten() - mean_face)

    # find closest face - use KNN
    faces = [i for i in range(len(pop_faces))]
    eucl_dst = {}
    for i in faces:
        eucl_dst[i] = np.linalg.norm(face_eform - pop_faces[i])
    faces.sort(key=lambda x: eucl_dst[x])
    # print(faces)
    # print([eucl_dst[i] for i in faces])
    u_fac = np.matmul(e_faces, pop_faces[faces[0]]) + mean_face
    return user_owner[faces[0]], np.sum(pop_faces[faces[0]]), u_fac, eucl_dst[faces[0]]

e_faces, mean_face, test_faces = get_data()

print(len(test_faces))

user_owner = {}
pop_faces = []
j = 0
for i, u in enumerate(test_faces):
    for img in u:
        pop_faces.append(img)
        user_owner[j] = i
        j += 1 

print(len(pop_faces))

tr_fac = []
tes_fac = []
for i in range(0, len(pop_faces)):
    if random.random() < 0.9:
        tr_fac.append(i)
    else:
        tes_fac.append(i)

print('tr_fac:', tr_fac)
print('tes_fac:', tes_fac)

tr_pop_face = [np.matmul(e_faces.T, pop_faces[i].flatten() - mean_face) for i in tr_fac]

user_faces = [0, 0, 0]
user_correct = [0, 0, 0]

for i in tes_fac:
    exp_owner, _, _, _ = get_closest_face(pop_faces[i], e_faces, mean_face, tr_pop_face, user_owner)
    if exp_owner is not None:
        user_faces[user_owner[i]] += 1
        print(exp_owner, ', ', user_owner[i])
        if exp_owner == user_owner[i]:
            user_correct[user_owner[i]] += 1
    else:
        print('None')

print(len(tr_fac))
print(user_faces)
print(user_correct)

# show the first face of each user
fig = plt.figure()
for i, u in enumerate(test_faces):
    plt.subplot(1, len(test_faces), i+1)
    plt.imshow(u[0], cmap='gray')
    plt.title('User ' + str(i))

plt.show()

total_faces = sum(user_faces)
total_correct = sum(user_correct)
print('Total Faces:', total_faces, ' Total Correct:', total_correct, ' Accuracy:', float(total_correct)/total_faces)

# total_faces = 0
# total_correct = 0
# for num, u in enumerate(test_faces):
#     user_faces = 0
#     user_correct = 0
#     for img in u:
#         # print(img.shape)
#         _, pix_sum, _, _ = get_closest_face(img, e_faces, mean_face, pop_faces, user_owner)
#         # print(closest_face.shape)
#         if pix_sum is not None:
#             user_faces += 1
#             if pix_sum == np.sum(np.matmul(e_faces.T, img.flatten() - mean_face)):
#                 user_correct += 1
#             # print('Closest face:', closest_face[0], 'Distance:', closest_face[3])
#             # plt.imshow(closest_face[1].reshape(192, 168), cmap='gray')
#             # plt.show()
#             # plt.imshow(closest_face[2].reshape(192, 168), cmap='gray')
#             # plt.show()
#     print(num)
#     print(user_faces)
#     print(float(user_correct)/user_faces)
#     print('User', num, ' Total User Faces:', user_faces, ' Accuracy:', float(user_correct)/user_faces)
#     total_faces += user_faces
#     total_correct += user_correct