from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class SemanticScorer:
    def __init__(self):

        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def calculate_similarity(self, student_answer, model_answer):

        # Generate embeddings
        embeddings = self.model.encode([student_answer, model_answer])

        # Compute cosine similarity
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

        return round(float(similarity), 2)
