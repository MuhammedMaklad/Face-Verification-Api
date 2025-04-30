

from flask import jsonify, request, render_template
from app.api import main_blueprint
import logging
from app.services.face_service import FaceVerificationService
from io import BytesIO
from PIL import Image
from torchvision import transforms
import base64

logger = logging.getLogger(__name__)

face_verification_service = FaceVerificationService()

@main_blueprint.route('/face', methods=['GET'])
def index():
    logger.info(f"request {request.path}")
    return render_template('index.html')

@main_blueprint.route('/face/detect', methods=['GET'])
def face():
    return render_template("upload.html")

@main_blueprint.route('/face/detect', methods=['POST'])
def detect_face():
    if 'image' not in request.files:
        return jsonify({"message": "No image file provided"}), 400

    image_file = request.files['image']
    try:
        img_bytes = BytesIO(image_file.read())
        img = Image.open(img_bytes).convert('RGB')

        face_tensor = face_verification_service.extract_face(img)  # returns tensor: [3, H, W]
        if face_tensor is None:
            return jsonify({"message": "No face detected"}), 400

        face_img = transforms.ToPILImage()(face_tensor)

        buffer = BytesIO()
        face_img.save(buffer, format='JPEG')
        encoded_face = base64.b64encode(buffer.getvalue()).decode('utf-8')

        return render_template('result.html', face_base64=encoded_face)

    except Exception as e:
        logger.warning(f"Exception error => {request.path}\n {e}")
        raise


@main_blueprint.route('/face/get-embedding', methods=['POST'])
def get_embedding():
    if 'image' not in request.files:
        return jsonify({"message": "No image file provided"}), 400

    image_file = request.files['image']
    try:
        img_bytes = BytesIO(image_file.read())
        img = Image.open(img_bytes).convert('RGB')

        face_tensor = face_verification_service.extract_face(img)  # returns tensor: [3, H, W]
        if face_tensor is None:
            return jsonify({"message": "No face detected"}), 400

        embedding = face_verification_service.get_embedding(face_tensor)

        return (
            jsonify({
                'message':"get embedding successfuly",
                'embedding':embedding.tolist()
            }, 200)
        )

    except Exception as e:
        logger.warning(f"Exception error => {request.path}\n {e}")
        raise


@main_blueprint.route('/face/verify', methods=['POST'])
def verify_face():
    data = request.get_json()
    if data is None:
        return jsonify({"message": "No data provided"}), 400
    try:
        embedding1 = data['embedding1']
        embedding2 = data['embedding2']

        if 'embedding1' not in data or 'embedding2' not in data:
            return jsonify({"message": "Missing embedding data"}), 400

        similarity = face_verification_service.cosine_similarity(embedding1, embedding2)
        if similarity > 0.7:
            return jsonify({"verify": True}), 200
        else:
            return jsonify({"verify": False}), 200
    except Exception as e:
        logger.warning(f"Exception error => {request.path}\n {e}")
        raise