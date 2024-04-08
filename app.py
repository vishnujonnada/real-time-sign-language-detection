from flask import Flask, render_template, Response, request
import base64
import cv2
import os
import pickle
import mediapipe as mp
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

app = Flask(__name__)


# Load the trained model for gesture recognition
model_dict = pickle.load(open('./model.p', 'rb'))
gesture_model = model_dict['model']

# Initialize MediaPipe Hands module
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Dictionary to map class labels to characters for gesture recognition
labels_dict = {"good":"good","hi":"hi","three":"three","two":"two"}

# Directory to store captured images for dataset creation
DATA_DIR = 'captured_images'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Route for capturing images to create dataset
@app.route('/capture_dataset', methods=['GET'])
def capture_dataset():
    return render_template('capture_dataset.html')

# Route for capturing images
@app.route('/capture_images', methods=['POST'])
def capture_images():
    # Get the class name and number of images to capture from the form
    class_name = request.form['class_name']
    num_images = int(request.form['num_images'])
    
    # Create directory for the class if it doesn't exist
    class_dir = os.path.join(DATA_DIR, class_name)
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    # Loop through the submitted images
    for i in range(num_images):
        # Get the image data from the form
        image_data = request.form.getlist('image[]')[i]

        # Convert the data URL to OpenCV image format
        data = image_data.split(',')[1]
        nparr = np.frombuffer(base64.b64decode(data), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Save the captured image to the class directory
        cv2.imwrite(os.path.join(class_dir, f'image_{i+1}.jpg'), img)

    return "Image capture completed."



# Route for training the model
@app.route('/train_model', methods=['GET'])
def train_model():
    # Load dataset
    data = []
    labels = []
    for dir_ in os.listdir(DATA_DIR):
        for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
            data_aux = []
            x_ = []
            y_ = []
            img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            results = hands.process(img_rgb)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        x_.append(x)
                        y_.append(y)

                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        data_aux.append(x - min(x_))
                        data_aux.append(y - min(y_))

                data.append(data_aux)
                labels.append(dir_)

    

    f = open('data.pickle', 'wb')
    pickle.dump({'data': data, 'labels': labels}, f)
    f.close()
    data_dict = pickle.load(open('./data.pickle', 'rb'))

    data = np.asarray(data_dict['data'])
    labels = np.asarray(data_dict['labels'])

    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

    model = RandomForestClassifier()

    model.fit(x_train, y_train)

    y_predict = model.predict(x_test)

    score = accuracy_score(y_predict, y_test)

    print('{}% of samples were classified correctly !'.format(score * 100))

    f = open('model.p', 'wb')
    pickle.dump({'model': model}, f)
    f.close()

    return "training completed"
# Route for gesture recognition
@app.route('/gesture_recognition')
def gesture_recognition():
    return render_template('gesture_recognition.html')

# Route for streaming webcam with hand gesture recognition
def gen(camera):
    while True:
        ret, frame = camera.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,  
                    hand_landmarks,  
                    mp_hands.HAND_CONNECTIONS,  
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

            data_aux = []
            x_ = []
            y_ = []
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            x1 = int(min(x_) * frame.shape[1]) - 10
            y1 = int(min(y_) * frame.shape[0]) - 10
            x2 = int(max(x_) * frame.shape[1]) - 10
            y2 = int(max(y_) * frame.shape[0]) - 10

            prediction = gesture_model.predict([np.asarray(data_aux)])
            print("prediction",prediction)
            
            if str(prediction[0]) in labels_dict:
                predicted_character = labels_dict[prediction[0]]
                print("Predicted character:", predicted_character)
            else:
                print("Predicted label not found in labels_dict.")
                predicted_character = "Unknown" 
                print("Predicted character:", predicted_character)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                        cv2.LINE_AA)

        ret, jpeg = cv2.imencode('.jpg', frame)
        frame_bytes = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(cv2.VideoCapture(0)), mimetype='multipart/x-mixed-replace; boundary=frame')

# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
