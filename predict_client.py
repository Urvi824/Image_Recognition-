import requests, sys

url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:5000/predict"
img = sys.argv[2] if len(sys.argv) > 2 else "sample.jpg"

with open(img, "rb") as f:
    files = {"file": (img, f, "image/jpeg")}
    r = requests.post(url, files=files, timeout=60)
    print(r.status_code, r.text)
