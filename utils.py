import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import os
import google.generativeai as genai
# from dotenv import load_dotenv

# load_dotenv()


GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

interpreter = None
mp_face_detection = None

def load_detect_face():
    """Load the MediaPipe face detection model."""
    global mp_face_detection
    if not mp_face_detection:
        mp_face_detection = mp.solutions.face_detection

def load_tflite_model():
    """Load the TensorFlow Lite model for NSFW detection."""
    global interpreter
    if not interpreter:
        interpreter = tf.lite.Interpreter(model_path='./models/nsfw_saved_model_quantized.tflite')
        interpreter.allocate_tensors()

def process_image(image_bytes):
    """Process the input image for both NSFW and face detection."""
    npimg = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    width_px = 512
    height_px = int(img.shape[0] / img.shape[1] * width_px)
    img = cv2.resize(img, (width_px, height_px))

    # Prepare the image for both detections
    rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tf_image = cv2.resize(rgb_image, (224, 224))
    tf_image = np.expand_dims(tf_image, axis=0).astype(np.float32) / 255.0

    result_nsfw_detected = {}
    result_face_detected = []

    # NSFW Detection
    try:
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        interpreter.set_tensor(input_details[0]['index'], tf_image)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])[0]
        result_nsfw_detected = {
            'drawing': float(output[0]),
            'hentai': float(output[1]),
            'neutral': float(output[2]),
            'porn': float(output[3]),
            'sexy': float(output[4])
        }
    except Exception as e:
        result_nsfw_detected = {'error': str(e)}

    # Face Detection
    try:
        with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
            results = face_detection.process(rgb_image)
            if results.detections:
                for detection in results.detections:
                    loc = detection.location_data
                    rel_kp = detection.location_data.relative_keypoints
                    extra_features = [{'x': kp.x, 'y': kp.y} for kp in rel_kp]
                    result_face_detected.append({
                        'score': detection.score[0],
                        'relative_bounding_box': {
                            'xmin': loc.relative_bounding_box.xmin,
                            'ymin': loc.relative_bounding_box.ymin,
                            'width': loc.relative_bounding_box.width,
                            'height': loc.relative_bounding_box.height,
                        },
                        'extra_features': extra_features
                    })
    except Exception as e:
        result_face_detected = [{'error': str(e)}]

    return {
        'nude_detection': result_nsfw_detected,
        'face_detection': result_face_detected
    }


# Google Gemini Extraction
async def extract_via_google_gemini(query):
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(
            f"""  
                {query}
            """
        )
        return response.text
    except Exception as e:
        return f"Unexpected error occurred {e} "


