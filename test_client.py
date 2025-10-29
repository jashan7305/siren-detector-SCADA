import requests

URL = "http://127.0.0.1:5000/predict"
AUDIO_FILE_NAME = "example_sounds/police_siren_us.wav"

def test_predict_endpoint():
    with open(AUDIO_FILE_NAME, "rb") as f:
        files = {"audio": f}
        res = requests.post(URL, files=files)
    
    print(res.json())

if __name__ == "__main__":
    test_predict_endpoint()