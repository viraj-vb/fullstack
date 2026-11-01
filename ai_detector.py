# Python AI-video detector: pip install -r requirements.txt && python ai_detector.py
import os, base64, json, requests, cv2, tempfile
from flask import Flask, request, jsonify
app = Flask(__name__)
KEY = os.getenv("GEMINI_API_KEY"); MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
PROMPT = ("You are a forensic video analyst. These are frames from one video. Judge whether it is AI-generated "
          "(diffusion/GAN/deepfake). Check warped hands/faces, inconsistent lighting, over-smooth texture, odd physics, garbled text. "
          'Return JSON {"ai_percent":0-100,"reasons":["short reason"]}.')

def frames(path, n=6):
    cap = cv2.VideoCapture(path); total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); out = []
    for i in range(n):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(total * (i + .5) / n)); ok, f = cap.read()
        if ok:
            f = cv2.resize(f, (640, int(640 * f.shape[0] / f.shape[1])))
            out.append(base64.b64encode(cv2.imencode(".jpg", f)[1]).decode())
    return out

@app.post("/detect-video")
def detect():
    fd, path = tempfile.mkstemp(suffix=".mp4"); os.close(fd)
    request.files["video"].save(path)
    try: imgs = frames(path)
    finally: os.unlink(path)
    if not imgs: return jsonify(error="Could not read video"), 400
    parts = [{"text": PROMPT}] + [{"inline_data": {"mime_type": "image/jpeg", "data": i}} for i in imgs]
    r = requests.post(f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent?key={KEY}",
                      json={"contents": [{"parts": parts}], "generationConfig": {"responseMimeType": "application/json"}}, timeout=90)
    r.raise_for_status()
    return jsonify(json.loads(r.json()["candidates"][0]["content"]["parts"][0]["text"]))

if __name__ == "__main__": app.run(port=5001)
