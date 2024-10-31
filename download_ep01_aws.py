
import boto3
import re
import json
import os
import time

bucket_name = '10b-capture-backup'
save_dir = '/home/jinling/Documents/data/Buckets/EP01/10b-capture-backup/ALLLEDs/hands_EPxx_download_20241009'

s3 = boto3.resource('s3')
bucket = s3.Bucket(bucket_name)
all_objs = bucket.objects.all()
all_objs = all_objs.filter(Prefix='10000') #Only download the ones start with '1000'

# For getting the objs size:
# https://stackoverflow.com/questions/51032100/how-to-get-size-of-filtered-objectscollection-in-boto3
obj_num = 100

# From the rig infomation, we can know the version of the image
gantry_version_dict = dict()
for obj_idx, obj in enumerate(all_objs):
    file_name = obj.key.split('/')[-1]
    if 'RigMetaData.json' not in file_name or '10000' not in file_name:
        continue
    gantry_id = file_name.split('_')[0]
    if gantry_id in gantry_version_dict.keys():
        continue
    s3_clientobj = boto3.client('s3').get_object(Bucket=bucket_name, Key=obj.key)
    s3_clientdata = s3_clientobj['Body'].read().decode('utf-8')
    json_data = json.loads(s3_clientdata)
    version = json_data['hostname'].split('-')[0]
    gantry_version_dict[gantry_id] = version
obj_num = obj_idx + 1

for obj_idx, obj in enumerate(all_objs):
    if 'image_All.jpg' not in obj.key:
        continue
    infos = obj.key.split('/')
    if len(infos) == 5:
        gantry_id, date, clock, camera, led = infos
    elif len(infos) == 3:
        vv, camera, led = infos
        gantry_id, date, clock = vv.split('_')
    else:
        continue
    if date <= '2024-09-21': # The images before this day will not be downloaded.
        continue
    # For the gantries have not json files, version will be EPxx
    version = gantry_version_dict[gantry_id] if gantry_id in gantry_version_dict.keys() else 'EPxx'
    version = version.upper()
    save_name = version + '_' + date+'_'+clock+'_'+gantry_id+'_XXX'+'_'+camera+'_handL'+'_'+'ALLLEDs.jpg'
    uri = 's3://'+bucket_name+'/'+obj.key
    command = 'aws s3 cp '+ uri+' '+os.path.join(save_dir, save_name)
    os.system(command)
    time.sleep(0.1)
    print('%d/%d: '%(obj_idx+1, obj_num), obj.key)

print('Completed. ')
