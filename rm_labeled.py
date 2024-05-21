
import os

json_dir = '/home/jinling/Documents/data/picked_hands/json_files'
check_dir = '/home/jinling/Documents/data/picked_hands/thick_strap'

json_names = [file.removesuffix('.json') for file in os.listdir(json_dir) if '.json' in file]

for file in os.listdir(check_dir):
    name = file.removesuffix('.png')
    if name not in json_names:
        continue
    command = 'rm ' + os.path.join(check_dir, file)
    os.system(command)
    print(command)
