import shutil
import os
def copy_path(path_1, path_2) :
    arr = path_1.split('\\')
    name = arr[len(arr) - 1]
    os.makedirs(f'{path_2}/{name}')
    files = os.listdir(path_1)
    for file in files :
        shutil.copy(f'{path_1}/{file}', f'{path_2}/{name}')

copy_path(r'2sem\\results\\1\\input',
          r'4sem\\results\\4.9')