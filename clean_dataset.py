import sys
import pandas as pd

from tools import undersample_dataframe
from tools import clean_text
from tools import preprocess_text

directory = 'data/'

FILE_NAME = sys.argv[1]
TEXT_COL = 'Text'
CAT_COL = 'Category'
CUSTOM_STOPWORDS = ['mr']


data_df = pd.read_csv(directory + 'raw/' + FILE_NAME)

# Drop rows with null values, duplicates
data_df = data_df.dropna()
data_df = data_df.drop_duplicates(subset = TEXT_COL)

# Undersample to balance categories
data_df = undersample_dataframe(data_df, CAT_COL, 0)



# Clean text
data_df[TEXT_COL] = data_df[TEXT_COL].apply(clean_text)

name = FILE_NAME.split('.')[0]
data_df.to_csv(directory + 'clean/' + name + '_CLEAN.csv', index = False)

# Preprocess text
data_df[TEXT_COL] = data_df[TEXT_COL].apply(preprocess_text, args = [CUSTOM_STOPWORDS])

data_df.to_csv(directory + 'clean/' + name + '_CLEAN_P.csv', index = False)