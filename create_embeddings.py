import joblib
from sentence_transformers import SentenceTransformer

from tools import read_in_data


read_directory = 'data/clean/'
documents, _, _ =  read_in_data(read_directory+'BBC_data_CLEAN.csv', x_col = 1, y_col = 2)


#load sentence_model
model_name = "all-MiniLM-L6-v2"
sentence_model = SentenceTransformer(model_name)

#embed/vectorize documents
embeddings = sentence_model.encode(documents)

#write out embeddings
joblib.dump(embeddings, read_directory + 'BBC_embeddings.z')
