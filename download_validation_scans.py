
import os

ValidationListFile = '/home/jinling/Documents/data/Buckets/EP01/10b-capture-backup/ALLLEDs/EP1_ValidationScans/ValidationList.txt'
save_dir = '/home/jinling/Documents/data/Buckets/EP01/10b-capture-backup/ALLLEDs/EP1_ValidationScans/scans'

with open(ValidationListFile, 'r') as fd:
    lines = fd.readlines()

scans = [line.replace('\n', '') for line in lines]
for scan in scans:
    's3://10b-dataset/AnishaValidationLeft131/'
    uri = os.path.join('s3://10b-dataset', scan)
    cmd = 'aws s3 cp ' + uri + ' ' + os.path.join(save_dir, scan) + ' --recursive'
    os.system(cmd)
    print(cmd)

print('Completed. ')
