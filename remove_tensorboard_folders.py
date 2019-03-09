
import os
import shutil

root_folder = 'run_A/'

for path, dirs, files in os.walk(root_folder):
#    print(path)
#    print(dirs)
#    print(files)
    if path[-11:] == 'tensorboard':
        print(path)
        shutil.rmtree(path)


