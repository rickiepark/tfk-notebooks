"""
 스크립트 구동 방법
 $ export FLASK_APP=urban_sound_classifier.py
 $ flask run
 ... Running on http://127.0.0.1:5000
"""

import numpy as np
import tensorflow as tf
import librosa
from flask import Flask, request
import os


def extract_feature(file_name):
    X, sample_rate = librosa.load(file_name)
    stft = np.abs(librosa.stft(X))
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0)
    return mfccs,chroma,mel,contrast,tonnetz

# 텐서플로우 모델 생성
training_epochs = 5000
n_dim = 193
n_classes = 10
learning_rate = 0.001

X = tf.placeholder(tf.float32, [None, n_dim])
Y = tf.placeholder(tf.float32, [None, n_classes])

c1 = tf.layers.conv2d(tf.reshape(X, [-1, 1, n_dim, 1]), 50, (1, 5), padding='same',
                      activation=tf.nn.sigmoid, name="c1")
p1 = tf.layers.max_pooling2d(inputs=c1, pool_size=[1, 2], strides=2)
c2 = tf.layers.conv2d(tf.reshape(p1, [-1, 1, 96, 50]), 100, (1, 5), padding='same',
                      activation=tf.nn.sigmoid, name="c2")
p2 = tf.layers.max_pooling2d(inputs=c2, pool_size=[1, 2], strides=2)

h_p = tf.reshape(p2, [-1, 48*100])

h_1 = tf.layers.dense(inputs=h_p, units=1000, activation=tf.nn.sigmoid,
                      kernel_initializer=tf.contrib.layers.xavier_initializer(), name="fc1")

y_hat = tf.layers.dense(inputs=h_1, units=n_classes,
                        kernel_initializer=tf.contrib.layers.xavier_initializer(),
                        name="h4")

y_sigmoid = tf.nn.sigmoid(y_hat)
y_ = tf.nn.softmax(y_hat)

init = tf.global_variables_initializer()

# 모델 파라메타 로드
saver = tf.train.Saver()
sess = tf.Session()
sess.run(init)
saver.restore(sess, 'model_adam.ckpt')

# 플래스크 앱 생성
app = Flask(__name__)
# 업로드 폴더 설정
app.config['UPLOAD_FOLDER'] = './upload'
# 최대 업로드 설정
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024


@app.route('/upload', methods=['POST'])
def upload_file():
    # 파일 업로드 필드 체크
    if 'file' not in request.files:
        return ''
    file = request.files['file']
    # 파일 이름이 없다면 리턴
    if file.filename == '':
        return ''
    # 업로드 폴더에 저장
    audio_file = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(audio_file)
    # 모델 실행
    mfccs, chroma, mel, contrast,tonnetz = extract_feature(audio_file)
    x_data = np.hstack([mfccs,chroma,mel,contrast,tonnetz])
    y_hat, sigmoid = sess.run([y_, y_sigmoid], feed_dict={X: x_data.reshape(1,-1)})
    index = np.argmax(y_hat)
    print(sigmoid)
    # 소리 인덱스 리턴
    return '%d' % (index)