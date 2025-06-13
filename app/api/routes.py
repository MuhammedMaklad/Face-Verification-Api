from flask import jsonify, request, render_template
from app.api import main_blueprint
import logging
import numpy as np
from app.services.card_service import IDCardProcessor
from app.services.face_service import FaceVerificationService
from io import BytesIO
from PIL import Image
from torchvision import transforms
import base64
from werkzeug.utils import secure_filename
import cv2


logger = logging.getLogger(__name__)

face_verification_service = FaceVerificationService()
processor = IDCardProcessor()

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

    logger.info(f"image --> {image_file}")
    try:
        img_bytes = BytesIO(image_file.read())
        logger.info(f"image bytes --> {img_bytes}")
        img = Image.open(img_bytes).convert('RGB')

        face_tensor = face_verification_service.extract_face(img)  # returns tensor: [3, H, W]
        logger.info(face_tensor)
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
    logger.info(f"Received image file: {image_file.filename}")

    try:
        # Read the image file into memory
        img_bytes = image_file.read()

        # Convert bytes to numpy array
        nparr = np.frombuffer(img_bytes, np.uint8)

        # Decode image using OpenCV (automatically converts to BGR format)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img_bgr is None:
            return jsonify({"message": "Invalid image file"}), 400

        # Convert BGR to RGB (most face recognition models expect RGB)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Optional: Resize if needed (maintain aspect ratio)
        max_dim = 1024
        h, w = img_rgb.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            img_rgb = cv2.resize(img_rgb, (0, 0), fx=scale, fy=scale)

        # Extract face - convert to format your service expects
        # Option 1: If service expects numpy array (RGB format)
        face_tensor = face_verification_service.extract_face(img_rgb)

        # Option 2: If service expects PIL Image
        # pil_img = Image.fromarray(img_rgb)
        # face_tensor = face_verification_service.extract_face(pil_img)

        if face_tensor is None:
            return jsonify({"message": "No face detected"}), 400

        embedding = face_verification_service.get_embedding(face_tensor)
        logger.info("Successfully generated embedding")
        # logger.info(embedding)

        return jsonify({
            'message': "Embedding generated successfully",
            'embedding': embedding.tolist()
        }), 200

    except Exception as e:
        logger.error(f"Error processing image: {str(e)}", exc_info=True)
        return jsonify({
            "message": "Error processing image",
            "error": str(e)
        }), 500

@main_blueprint.route('/face/verify', methods=['POST'])
def verify_face():
    data = request.get_json()

    if data is None:
        return jsonify({"message": "No data provided"}), 400

    if 'embedding1' not in data or 'embedding2' not in data:
        return jsonify({"message": "Missing embedding data"}), 400

    try:
        # Convert embeddings to numpy arrays
        embedding1 = np.array(data['embedding1'], dtype=np.float32)
        embedding2 = np.array(data['embedding2'], dtype=np.float32)

        logger.info(f"embedding shapes --> {embedding1.shape}, {embedding2.shape}")
        logger.info(f"embedding types --> {type(embedding1)}, {type(embedding2)}")
        logger.info(f"embedding dtypes --> {embedding1.dtype}, {embedding2.dtype}")

        similarity = face_verification_service.cosine_similarity(embedding1, embedding2)

        logger.info(f"similarity --> {similarity}")
        if similarity > 0.7:
            return jsonify({"verify": True, "similarity": float(similarity)}), 200
        else:
            return jsonify({"verify": False, "similarity": float(similarity)}), 200
    except Exception as e:
        logger.warning(f"Exception error => {request.path}\n {str(e)}", exc_info=True)
        return jsonify({"message": "Error processing request", "error": str(e)}), 500


@main_blueprint.route('/face/process-id', methods=['POST'])
def process_id():
    try:
        print("=== DEBUG: process_id called ===")
        print(f"Request files: {list(request.files.keys())}")
        print(f"Request form: {dict(request.form)}")

        if 'front' not in request.files or 'back' not in request.files:
            print("ERROR: Missing files")
            return jsonify({"message": "Both images required"}), 400

        print("Loading images...")
        front = processor.load_image(request.files['front'])
        back = processor.load_image(request.files['back'])
        print("Images loaded successfully")

        print("Processing front image...")
        info = processor.process_image(front)
        print(f"Front processing result: {info}")

        print("Processing back image...")
        info['job'] = processor.process_job(back)
        print(f"Back processing result: {info['job']}")

        return jsonify({"status": "success", "data": info})
    except Exception as e:
        print(f"ERROR in process_id: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500



@main_blueprint.route('/face/upload-id')
def upload_id():
    return render_template('upload_id.html')

@main_blueprint.route('/face/display-id-info')
def display_id_info():
    return render_template('display_id_info.html')