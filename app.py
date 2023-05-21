from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import cv2
import numpy as np
import base64
import time
import random
import string
from textblob import TextBlob
import spacy
import tensorflow as tf
import mediapipe as mp


nlp = None
interpreter = None
mp_face_detection = None

def load_detect_face():
    global mp_face_detection
    mp_face_detection = mp.solutions.face_detection


def load_nlp_model():
    global nlp
    nlp = spacy.load('en_core_web_sm')

def load_tflite_model():
    global interpreter
    interpreter = tf.lite.Interpreter(model_path='./models/nsfw_saved_model_quantized.tflite')
    interpreter.allocate_tensors()


app = Flask(__name__)
CORS(app, origins='*')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/analyze/image', methods=['POST'])
def upload_image():
    try:
        global interpreter
        global mp_face_detection

        if mp_face_detection is None:
            load_detect_face()

        if interpreter is None:
            load_tflite_model()

        filestr = request.files['image'].read()
        npimg = np.frombuffer(filestr, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        width_px = 512
        height_px = int(img.shape[0] / img.shape[1] * width_px)
        img = cv2.resize(img, (width_px, height_px))
        rand = ''.join(random.choices(string.digits, k=6))
        tstr = str(int(str(time.time()).replace('.', '')))
        filename = 'uploads/uploaded_image'+tstr+"_"+rand+'.jpg'
        result_nsfw_detected = []
        result_face_detected = []
        _, img_encoded = cv2.imencode('.jpg', img)
        base64_image = base64.b64encode(img_encoded).decode('utf-8')
        image = tf.image.decode_jpeg(base64.b64decode(base64_image), channels=3)
        image = tf.image.resize(image, (224, 224))
        image = np.expand_dims(image, axis=0)
        image = image.astype(np.float32) / 255.0
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        interpreter.set_tensor(input_details[0]['index'], image)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])[0]
        result_nsfw_detected = {
            'drawing':str(output[0]),
            'hentai':str(output[1]),
            'neutral':str(output[2]),
            'porn':str(output[3]),
            'sexy':str(output[4])
        }

        rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        with mp_face_detection.FaceDetection(
            min_detection_confidence=0.5) as face_detection:
            results = face_detection.process(rgb_image)
            if results.detections:
                for detection in results.detections:
                    loc = detection.location_data
                    rel_kp = detection.location_data.relative_keypoints
                    extra_features = []
                    for k in rel_kp:
                        extra_features.append({'x':k.x,'y':k.y})
                    d1 = {
                       'score':detection.score[0],
                       'relative_bounding_box':{
                           'xmin':loc.relative_bounding_box.xmin,
                           'ymin':loc.relative_bounding_box.ymin,
                           'width':loc.relative_bounding_box.width,
                           'height':loc.relative_bounding_box.height,
                       },
                       'extra_features':extra_features
                    }
                    result_face_detected.append(d1)

        response = jsonify({
            'status':'success',
            'filename': filename,
            'image':base64_image,
            'nude_detection':result_nsfw_detected,
            'face_detection':result_face_detected
        })
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response
    except Exception as e:
        response = jsonify({'status': 'error', 'message': str(e)})
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response

@app.route('/analyze/text/job', methods=['POST'])
def upload_text_job():
    try:
        if nlp is None:
            load_nlp_model()
        skills_provided = request.form['skills_provided']
        job_description = request.form['job_description']
        job_descriptionTb = TextBlob(job_description)
        doc = nlp(job_description)
        skills_extracted = []
        for entity in doc.ents:
            if entity.label_ in ["ORG", "TECH", "SKILL"]:
                skills_extracted.append(entity.text)

        skills_similarity_score = nlp(', '.join(skills_extracted)).similarity(nlp(skills_provided))
        
        response = jsonify({
            'job_description':{
                'sentiment':{
                    'polarity':job_descriptionTb.sentiment.polarity,
                    'subjectivity':job_descriptionTb.sentiment.subjectivity,
                },
                'skills':{
                    'skills_provided': skills_provided,
                    'skills_extracted': skills_extracted,
                    'skills_similarity_score': skills_similarity_score,
                },
            },
        })

        response.headers.add('Access-Control-Allow-Origin', '*')
        return response
    except Exception as e:
        response = jsonify({'status': 'error', 'message': str(e)})
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response

@app.route('/analyze/text/post', methods=['POST'])
def upload_text_post():
    try:
        if nlp is None:
            load_nlp_model()
        parent_post_or_comment = request.form['parent_post_or_comment']
        child_comment = request.form['child_comment']
        parent_post_or_commentTb = TextBlob(parent_post_or_comment)
        child_commentTb = TextBlob(child_comment)
        context_similarity_score = nlp(parent_post_or_comment).similarity(nlp(child_comment))

        response = jsonify({
            'post_description':{
                'parent_post_or_comment':{
                    'sentiment':{
                        'polarity':parent_post_or_commentTb.sentiment.polarity,
                        'subjectivity':parent_post_or_commentTb.sentiment.subjectivity,
                    },
                },
                'child_comment':{
                    'sentiment':{
                        'polarity':child_commentTb.sentiment.polarity,
                        'subjectivity':child_commentTb.sentiment.subjectivity,
                    },
                },
                'context_similarity_score': context_similarity_score,
            },
        })

        response.headers.add('Access-Control-Allow-Origin', '*')
        return response
    except Exception as e:
        response = jsonify({'status': 'error', 'message': str(e)})
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response




if __name__ == '__main__':
    app.run(port=5000, threaded=True)
