import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


df = pd.read_csv("Question_Answers.csv", index_col=None, encoding='latin1')
df.dropna(inplace=True)

vector = TfidfVectorizer()

vector.fit(np.concatenate((df.Questions, df.Answers)))

vector_questions = vector.transform(df.Questions)

#print(vector_questions)

while True:
    
    user_input = input()
    
    print(user_input)
    
    vector_user_input = vector.transform([user_input])
    
    similarities = cosine_similarity(vector_user_input, vector_questions)
    
    closest_question = np.argmax(similarities, axis=1)
    
    print("Similarities: ", similarities)
    
    print("Closet Question: ", closest_question)
    
    answer = df.Answers.iloc[closest_question].values[0]
    
    print("Answer: ", answer)
