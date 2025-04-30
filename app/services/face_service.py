from facenet_pytorch import MTCNN, InceptionResnetV1
from keras import Model
import numpy as np

class FaceVerificationService:

    def __init__(self):
        self.detector = MTCNN(keep_all=False, image_size=160, margin=15, min_face_size=20, device='cpu')
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval()

    def extract_face(self, image):
        # Detect face
        face = self.detector(image)
        return face if face is not None else None

    def get_embedding(self, face_tensor):
        if face_tensor is None:
            return None

        # Calculate embedding (unsqueeze to add batch dimension)
        embedding = self.resnet(face_tensor.unsqueeze(0))

        # Convert to numpy array and normalize
        embedding = embedding.detach().numpy()[0]
        return embedding / np.linalg.norm(embedding)

    def cosine_similarity(self, a, b):
        """Calculate cosine similarity between two embeddings"""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
