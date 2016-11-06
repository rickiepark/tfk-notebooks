import requests

url = 'http://127.0.0.1:5000/upload'
files = {'file': open('UrbanSound8K/audio/fold1/7061-6-0-0.wav', 'rb')}
r = requests.post(url, files=files)
print(r.text)
