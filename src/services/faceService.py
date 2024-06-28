from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import numpy as np
import os
import pickle
import traceback

class FaceService:
    instance = None

    def __init__(self):
        self.mtcnn = MTCNN(image_size=160, margin=0)
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval()
        self.known_face_encoding = None
        self.known_face_name = None
        self.known_faces_dir = "src/saveFaces"
        self.load_known_faces()

    def load_known_faces(self):
        if not os.path.exists(self.known_faces_dir):
            os.makedirs(self.known_faces_dir)
        
        known_faces_path = os.path.join(self.known_faces_dir, 'known_face.pkl')
        if os.path.exists(known_faces_path):
            with open(known_faces_path, 'rb') as f:
                self.known_face_encoding, self.known_face_name = pickle.load(f)

    async def save_known_faces(self):

        if not os.path.exists(self.known_faces_dir):
            os.makedirs(self.known_faces_dir)

        known_faces_path = os.path.join(self.known_faces_dir, 'known_face.pkl')
        with open(known_faces_path, 'wb') as f:
            pickle.dump((self.known_face_encoding, self.known_face_name), f)

    async def add_face(self, name, image_file):
        try:
            image_file.file.seek(0)
            image = Image.open(image_file.file)
            img_cropped = self.mtcnn(image)
            if img_cropped is not None:
                img_embedding = self.resnet(img_cropped.unsqueeze(0)).detach().cpu().numpy()[0]
                self.known_face_encoding = img_embedding
                self.known_face_name = name
                await self.save_known_faces()
                return {"success": True, "message": "Face added successfully"}
            else:
                return {"success": False, "message": "No face found in the image"}
        except Exception as e:
            error_message = traceback.format_exc()
            return {"success": False, "message": error_message}

    async def recognize_face(self, image_file):
        if self.known_face_encoding is None:
            return {"success": False, "message": "No known face to compare with"}

        try:
            image_file.file.seek(0)
            image = Image.open(image_file.file)
            img_cropped = self.mtcnn(image)
            if img_cropped is not None:
                img_embedding = self.resnet(img_cropped.unsqueeze(0)).detach().cpu().numpy()[0]
                distance = np.linalg.norm(self.known_face_encoding - img_embedding)
                if distance < 1.0:
                    name = self.known_face_name
                else:
                    name = "Unknown"
                return {"success": True, "name": name}
            else:
                return {"success": False, "message": "No face found in the image"}
        except Exception as e:
            error_message = traceback.format_exc()
            return {"success": False, "message": error_message}

    async def detect_face(self, image_file):
        try:
            image_file.file.seek(0)
            image = Image.open(image_file.file)
            img_cropped = self.mtcnn(image)
            if img_cropped is not None:
                return {"success": True, "message": "Face detected"}
            else:
                return {"success": False, "message": "No face detected"}
        except Exception as e:
            error_message = traceback.format_exc()
            return {"success": False, "message": error_message}
    @staticmethod
    def getInstance():
        if FaceService.instance is None:
            FaceService.instance = FaceService()
        return FaceService.instance


faceServiceInstance = FaceService.getInstance()
