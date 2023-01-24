import os
import pandas as pd

category_folders = [
    'business',
    'entertainment',
    'politics',
    'sport',
    'tech',
]

directory = 'data/raw/bbc/'

texts = []
classifications = []

for category in category_folders:

    for file_name in os.listdir(directory+category):
    
        with open(directory+category+'/'+file_name) as file:
            texts.append(file.read())
            classifications.append(category)
        

df = pd.DataFrame(zip(texts, classifications),
                    columns = ['Text', 'Category'])
                    

df.to_csv('data/raw/'+'BBC_data.csv', index_label = 'Index')