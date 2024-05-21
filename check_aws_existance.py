
import os

img_dirs = [
'/home/jinling/Documents/data/tmpdir/batch20231130171013_48998',
'/home/jinling/Documents/data/tmpdir/batch20231130170442_48819',
'/home/jinling/Documents/data/tmpdir/batch20231130171557_49206'
]

json_dirs = [            '/home/jinling/Documents/data/Buckets/deep-learning-datasets-10b/fingers-segmentation/p0_batches/annotations/devebec/batch_p0_06_31OCT2023_devebec/LabelMe_2/',
'/home/jinling/Documents/data/Buckets/deep-learning-datasets-10b/fingers-segmentation/p0_batches/annotations/devebec/batch_p0_06_31OCT2023_devebec/Lableme_1/',
'/home/jinling/Documents/data/Buckets/deep-learning-datasets-10b/fingers-segmentation/p0_batches/annotations/devebec/batch_p0_06_31OCT2023_devebec/LableMe/',
'/home/jinling/Documents/data/Buckets/deep-learning-datasets-10b/fingers-segmentation/p0_batches/annotations/devebec/batch_p0_06_31OCT2023_devebec/LableMe_3/',]

json_dict = {}
for json_dir in json_dirs:
    for file in os.listdir(json_dir):
        if '.json' not in file:
            continue
        name = file.removesuffix('.json')
        json_dict[name] = os.path.join(json_dir, file)

for img_dir in img_dirs:
    for file in os.listdir(img_dir):
        if '.png' not in file:
            continue
        name = file.removesuffix('.png')
        if name not in json_dict.keys():
            print(os.path.join(img_dir, name)+'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        else:
            print(os.path.join(img_dir, name))

