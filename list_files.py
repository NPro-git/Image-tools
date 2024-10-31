
import os
dir = '/home/jinling/Documents/data/full-seg-datasets/models_performance/fingers_v4.18_alpha_p0_ep01_natural_angles/test/images'

files = [file for file in os.listdir(dir) if '.png' in file]

for file in files[:12000]:
    print(file.removesuffix('.png'))

print('Num: ', len(files))
