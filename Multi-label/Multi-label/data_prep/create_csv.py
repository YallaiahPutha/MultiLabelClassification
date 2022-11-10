import json
from csv_dict import rows_dict
from create_dct import lst
import pandas as pd

f=open('assert_filter.json')
data=json.load(f)
annot_dict={}
cat_dict={}
for annotation in data['annotations']:
    if annotation['image_id'] not in annot_dict:
        annot_dict[annotation['image_id']]=set()
    annot_dict[annotation['image_id']].add(annotation['category_id'])

for category in data['categories']:
    cat_dict[category['name']]=category['id']

cnt=0
for img in data['images']:
    rows_dict['ImageId'].append(img['file_name'])
    label_lst=[]
    for cat in lst:
        if cat_dict[cat.lower()] in annot_dict[img['id']]:
            label_lst.append(cat.lower())
            rows_dict[cat.lower()].append(1)
        else:
            rows_dict[cat.lower()].append(0)
    rows_dict['Labels'].append(label_lst)

print(rows_dict)


data_df = pd.DataFrame.from_dict(rows_dict, orient = 'columns').to_csv('annotate.csv')




    
