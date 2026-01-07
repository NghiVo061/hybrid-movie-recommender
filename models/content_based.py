import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class ContentBasedModel:
    def __init__(self, movies_path):
        self.movies = pd.read_csv(movies_path)
        self.tfidf_matrix = None
        self.cosine_sim = None
        
    def fit(self):
        tfidf = TfidfVectorizer(token_pattern=r'(?u)\b\w+\b')
        self.tfidf_matrix = tfidf.fit_transform(self.movies['genres_text'])
        self.cosine_sim = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)
        print("CB Model: Đã huấn luyện xong TF-IDF và Cosine Similarity.")

    def recommend(self, movie_title, top_n=10):
        if movie_title not in self.movies['title'].values:
            return "Không tìm thấy phim này trong dữ liệu."
            
        idx = self.movies[self.movies['title'] == movie_title].index[0]
        sim_scores = list(enumerate(self.cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        movie_indices = [i[0] for i in sim_scores[1:top_n+1]]
        return self.movies[['title', 'genres_text']].iloc[movie_indices]

cb_model = ContentBasedModel('movies_cleaned.csv')
cb_model.fit()
print("\nGợi ý cho phim 'Toy Story (1995)':")
print(cb_model.recommend('Toy Story (1995)'))