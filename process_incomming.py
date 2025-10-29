import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import joblib
import requests


def create_embedding(text_list):
    r = requests.post("http://localhost:11434/api/embed", json={
        "model": "bge-m3",
        "input": text_list
    })
    
    embedding = r.json()["embeddings"]
    return embedding

def  inference(prompt):
    r = requests.post("http://localhost:11434/api/generate", json={
        # "model": "deepseek-r1:1.5b",
        "model": "llama3.2",
        "prompt": prompt,
        "stream": False
    })

    response = r.json()
    print(response)
    return response

    
df = joblib.load('embeddings.joblib')

incoming_query = input("Ask the question: ")
question_embedding = create_embedding([incoming_query])[0]


# Find the similarities of the question_embedding with other embeddings
# print(np.vstack(df['embedding'].values))
# print(np.vstack(df['embedding']).shape)
similarities = cosine_similarity(np.vstack(df['embedding']), [question_embedding]).flatten()
# print(similarities)
top_results = 5
max_indx = similarities.argsort()[::-1][0:top_results]
# print(max_indx)
new_df = df.loc[max_indx]
# print(new_df[["title", "number", "text"]])

prompt = f'''I am teaching web development in my sigma web development course. Here are video subtitle chunks containing video title, video number, start time in seconds, end time in seconds, the text at that time:

{new_df[["title", "number", "start", "end", "text"]].to_json(orient="records")}
----------------------------------

"{incoming_query}"
User asked this question realted to the video chunks, you have answer in a human way(don't mention the above format, its just for you) where and how much content is tought in which video (in which video and at what timestamp) and guide the user to go to that particular video. If user ask unrelated question, tell him that you can only answer questions related to the course
'''

with open ("prompt.text", "w") as f:
    f.write(prompt)

response = inference(prompt)["response"]
print(response)

with open("response.txt", "w") as f:
    f.write(response)
