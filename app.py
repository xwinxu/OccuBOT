import csv
import sys
import os

from PIL import Image
import io

from flask import Flask, render_template, request

from imutils import paths
from imutils.object_detection import non_max_suppression
#import numpy as np
#nimport argparse
import imutils
import tempfile
import cv2

from detect import detectHuman


ROOT_DIR = os.path.abspath(os.path.dirname(__file__))

reload(sys)
sys.setdefaultencoding('utf8')

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join(ROOT_DIR, "static/uploads")
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    restaurants = open_file()
    return render_template('index.html', restaurants=restaurants)

@app.route('/upload/')
def upload():
	return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_file():
	file = request.files['image']
	temp_dir = tempfile.mkdtemp()
	if file:
		f = os.path.join(temp_dir, file.filename)
		file.save(f)

		newImage = detectHuman(f, temp_dir)
		
		return newImage
	else:
		return render_template('upload.html')

def open_file():
    with open('tripadvisor_in-restaurant_sample.csv', 'rb') as infile:
        data = [row for row in csv.DictReader(infile)]
    return data


if __name__ == '__main__':
    app.run(debug=True)

