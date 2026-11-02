from flask import Flask, render_template, request, redirect, url_for, session
from PIL import Image
import cv2
import numpy as np
import random
import os

app = Flask(__name__)
app.secret_key = 'your_secret_key'

users = {'admin': 'admin'}

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if username in users and users[username] == password:
            session['username'] = username  # Set the session variable
            return redirect(url_for('index'))
        else:
            return "Login failed. Please check your username and password."

    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if username not in users:
            users[username] = password
            return redirect(url_for('login'))
        else:
            return "Registration failed. User already exists."

    return render_template('register.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/segment', methods=['POST', 'GET'])
def segment():
    if 'file' not in request.files:
        return "No file part"
    
    file = request.files['file']
    
    if file.filename == '':
        return "No selected file"
    
    if file:
        original_image = Image.open(file)
        original_image_cv2 = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)
        
        # Perform segmentation with different methods
        original_segmented_image, detected_classes = segment_image(original_image_cv2)
        threshold_segmented = segment_image(original_image_cv2, 'threshold')
        edge_segmented = segment_image(original_image_cv2, 'edge')
        kmeans_segmented = segment_image(original_image_cv2, 'kmeans')
        
        # Save the images temporarily
        original_path = 'static/original_image.jpg'
        original_segmented_path = 'static/original_segmented_image.jpg'
        threshold_path = 'static/threshold_segmented_image.jpg'
        edge_path = 'static/edge_segmented_image.jpg'
        kmeans_path = 'static/kmeans_segmented_image.jpg'
        
        cv2.imwrite(original_path, original_image_cv2)
        cv2.imwrite(original_segmented_path, original_segmented_image)
        cv2.imwrite(threshold_path, threshold_segmented)
        cv2.imwrite(edge_path, edge_segmented)
        cv2.imwrite(kmeans_path, kmeans_segmented)
        
        return render_template('result.html', 
                               detected_classes=detected_classes,
                               original_image=original_path, 
                               original_segmented_image=original_segmented_path,
                               threshold_image=threshold_path, 
                               edge_image=edge_path, 
                               kmeans_image=kmeans_path)

@app.route('/threshold_result', methods=['GET'])
def threshold_result():
    # You can fetch the path of the images you want to display from the session or another storage method
    original_image_path = 'static/original_image.jpg'
    threshold_image_path = 'static/threshold_segmented_image.jpg'
    
    return render_template('threshold_result.html', 
                           original_image=original_image_path, 
                           threshold_image=threshold_image_path)


def segment_image(image, method='original'):
    if method == 'original':
        return original_segmentation(image)
    elif method == 'threshold':
        return threshold_segmentation(image)
    elif method == 'edge':
        return edge_detection_segmentation(image)
    elif method == 'kmeans':
        return kmeans_segmentation(image)

def original_segmentation(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
    background = np.full_like(image, (255, 0, 0))  # BGR for blue
    yellow = np.full_like(image, (0, 255, 255))  # BGR for yellow
    segmented_image = np.where(mask[:, :, None] == 255, yellow, background)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(segmented_image, (x, y), (x+w, y+h), (0, 0, 255), 2)  # Red rectangle
    detected_classes = generate_random_classes()
    return segmented_image, detected_classes

def threshold_segmentation(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
    return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

def edge_detection_segmentation(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_image, 100, 200)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

def kmeans_segmentation(image):
    Z = image.reshape((-1, 3))
    Z = np.float32(Z)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 3
    _, labels, centers = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    segmented_image = centers[labels.flatten()]
    segmented_image = segmented_image.reshape((image.shape))
    return segmented_image

def generate_random_classes():
    classes = ["Successfully Segmented"]
    num_classes = random.randint(1, len(classes))
    random_classes = random.sample(classes, num_classes)
    return random_classes

if __name__ == "__main__":
    app.run(debug=True)

