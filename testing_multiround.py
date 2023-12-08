import math
import random
from matplotlib import pyplot as plt 
import numpy as np
import os 
import scipy.io as sio

def get_data():
    # e_faces = np.load('e_faces_set_all_faces.npy')
    # mean_face = np.load('mean_face_all_faces.npy')

    # get the test faces - split by their subject
    data = sio.loadmat('allFaces.mat')
    faces = data['faces']
    n = data['n'][0][0]
    m = data['m'][0][0]
    nfaces = np.ndarray.flatten(data['nfaces'])
    print(nfaces.shape[0], 'people with indiv img count:', nfaces)
    print('images:', len(faces), 'n:', n, 'm:', m)
    imgs = np.array(faces)
    all_faces = []
    for j in range(len(nfaces)):
        subj = []
        for i in range(nfaces[j]):
            img = np.array(imgs[:, i + np.sum(nfaces[:j])]).reshape(m, n).T
            subj.append(img)
        all_faces.append(subj)

    return all_faces

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

def train_on_set(trn_faces, e_cnt = 900):
    trn_vecs = [fac.flatten() for fac in trn_faces]
    raw_trn_fac = np.array(trn_vecs).T
    mean_face = np.mean(raw_trn_fac, axis=1)
    raw_trn_fac = raw_trn_fac - np.tile(mean_face, (raw_trn_fac.shape[1], 1)).T

    e_faces, _, _ = np.linalg.svd(raw_trn_fac, full_matrices=False)
    return e_faces[:, :e_cnt], mean_face

def run_with_test_set(all_faces, test_face_nums, iter):
    training_faces = []
    test_faces = []
    for i, u in enumerate(all_faces):
        if i in test_face_nums:
            test_faces.append(u)
        else:
            training_faces += u
    
    print('\tTraining...')
    e_faces, mean_face = train_on_set(training_faces)

    user_owner = {}
    pop_faces = []
    j = 0
    for i, u in enumerate(test_faces):
        for img in u:
            pop_faces.append(img)
            user_owner[j] = i
            j += 1

    user_faces = [0 for i in range(len(test_face_nums))]
    user_correct = [0 for i in range(len(test_face_nums))]
    
    for i in range(iter):
        tr_te_fac = []
        tes_fac = []
        for j in range(0, len(pop_faces)):
            if random.random() < 0.9:
                tr_te_fac.append(j)
            else:
                tes_fac.append(j)

        tr_pop_face = [np.matmul(e_faces.T, pop_faces[i].flatten() - mean_face) for i in tr_te_fac]

        for j in tes_fac:
            exp_owner, _, _, _ = get_closest_face(pop_faces[j], e_faces, mean_face, tr_pop_face, user_owner)
            if exp_owner is not None:
                user_faces[user_owner[j]] += 1
                # print(exp_owner, ', ', user_owner[j])
                if exp_owner == user_owner[j]:
                    user_correct[user_owner[j]] += 1
            else:
                print('None type in closest')
        
        print('\t\tIteration:', i, 'Faces:', user_faces, 'Correct:', user_correct)

    return user_faces, user_correct

def run_multi_round():
    all_faces = get_data()
    # print(len(all_faces))

    subj_tested = [0 for i in range(len(all_faces))]
    subj_correct = [0 for i in range(len(all_faces))]
    epochs_as_test = [0 for i in range(len(all_faces))]

    epochs = 50
    for epoch in range(epochs):
        test_face_nums = []
        user_face_nums = random.sample(range(len(all_faces)), 3)
        user_face_nums.sort()

        user_tstd, user_corr = run_with_test_set(all_faces, user_face_nums, 5)

        for i, user in enumerate(user_face_nums):
            subj_tested[user] += user_tstd[i]
            subj_correct[user] += user_corr[i]
            epochs_as_test[user] += 1

        print('Epoch [', epoch, '/', epochs, '] Complete \n\tSubject Tested: ', subj_tested, '\n\tSubject Correct: ', subj_correct, sep='')


    print('\n\nFinal Results:')
    print('\tSubject Tested: ', subj_tested, '\n\tSubject Correct: ', subj_correct, '\n\tEpochs Tested: ', epochs_as_test, sep='')
    print('\tSubject Accuracy: ', [float(subj_correct[i])/subj_tested[i] for i in range(len(subj_correct))], sep='')
    print('\tTotal Accuracy: ', float(sum(subj_correct))/sum(subj_tested), sep='')

    print('User | Epochs Tested | Correct | Total | Accuracy')
    for i in range(len(subj_correct)):
        print(i, ' | ', epochs_as_test[i], ' | ', subj_correct[i], ' | ', subj_tested[i], ' | ', float(subj_correct[i])/subj_tested[i], '%', sep='')


    # total_faces = sum(user_faces)
    # total_correct = sum(user_correct)
    # print('Total Faces:', total_faces, ' Total Correct:', total_correct, ' Accuracy:', float(total_correct)/total_faces)

def graph_pre_res():
    subj_test = [157, 60, 62, 174, 83, 131, 139, 127, 167, 186, 114, 83, 88, 57, 182, 31, 92, 112, 205, 230, 184, 113, 64, 118, 127, 188, 150, 70, 191, 121, 57, 122, 126, 163, 42, 121, 249, 85]
    subj_corr =  [141, 57, 59, 147, 81, 97, 109, 100, 141, 131, 100, 77, 82, 40, 136, 29, 69, 83, 169, 179, 134, 78, 48, 105, 102, 138, 131, 50, 130, 76, 41, 88, 81, 116, 26, 87, 171, 61]
    subj_epochs = [5, 2, 2, 5, 3, 4, 4, 4, 5, 5, 4, 3, 3, 2, 6, 1, 3, 3, 6, 7, 6, 3, 2, 4, 4, 6, 5, 2, 6, 4, 2, 4, 4, 5, 1, 4, 8, 3]

    subj_accu = [float(subj_corr[i])/subj_test[i] for i in range(len(subj_test))]

    plt.hist(subj_accu, bins=np.linspace(0, 1, 41), color='red', alpha=0.5, edgecolor='black' )
    plt.title('Recognition Accuracy by Subject')
    plt.xlabel('Accuracy')
    plt.ylabel('Frequency')
    plt.show()

    plt.clf()
    plt.hist(subj_test, bins=np.linspace(0, 250, 41), color='blue', alpha=0.5, edgecolor='black')
    plt.title('Recognition Attempts by Subject')
    plt.xlabel('Attempts')
    plt.ylabel('Frequency')
    plt.show()

    comps = [(i, subj_accu[i]) for i in range(len(subj_accu))]
    comps.sort(key=lambda x: x[1])
    print('Sorted by Accuracy:')
    for (i, accu) in comps:
        print('Subject', i, 'Accuracy:', accu)

    print('Overall Accuracy:', float(sum(subj_corr))/sum(subj_test), 'Total Comparisons:', sum(subj_test))

graph_pre_res()