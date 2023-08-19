import os

def mkdir(path):
    path = path.split('/')
    for iter in range(1, len(path) + 1):
        temp_dir = '/'.join(path[:iter])
        if not os.path.exists(temp_dir):
            os.mkdir(temp_dir)