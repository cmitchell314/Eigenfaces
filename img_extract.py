import tarfile

with tarfile.open('./CALTECH_faces/faces.tar') as tar:
    tar.extractall(path='./CALTECH_faces/imgs')

print('all imgs extracted')