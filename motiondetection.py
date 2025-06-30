from flask import Flask, Response, render_template_string, request, jsonify
import cv2
import numpy as np
import threading
import base64
import io
from PIL import Image
import os

app = Flask(__name__)

# Variables globales
fgbg = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)
counter = 0
frame_width = 640
frame_height = 480
line_position = frame_width // 2
tracking_object = None
latest_frame = None
frame_lock = threading.Lock()

def process_frame_data(frame_data):
    """Traite les données d'image reçues du client"""
    global counter, tracking_object, latest_frame, fgbg
    
    try:
        # Décoder l'image base64
        image_data = base64.b64decode(frame_data.split(',')[1])
        image = Image.open(io.BytesIO(image_data))
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Redimensionner l'image
        frame = cv2.resize(frame, (frame_width, frame_height))
        
        # Dessiner la ligne de comptage
        cv2.line(frame, (line_position, 0), (line_position, frame_height), (255, 0, 0), 3)
        
        # Appliquer la soustraction d'arrière-plan
        fgmask = fgbg.apply(frame)
        
        # Supprimer le bruit
        kernel = np.ones((5, 5), np.uint8)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
        
        # Trouver les contours
        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        biggest_area = 0
        biggest_center = None
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 1000:  # Seuil adapté pour les images plus petites
                if area > biggest_area:
                    x, y, w, h = cv2.boundingRect(cnt)
                    cx = x + w // 2
                    cy = y + h // 2
                    biggest_area = area
                    biggest_center = (cx, cy)
        
        if biggest_center is not None:
            cx, cy = biggest_center
            cv2.circle(frame, (cx, cy), 10, (0, 0, 255), -1)
            
            if tracking_object is not None:
                prev_cx, prev_cy = tracking_object
                
                if abs(prev_cy - cy) < 50:
                    if prev_cx > line_position and cx <= line_position:
                        counter += 1
                        print("Right to Left! Counter:", counter)
                        tracking_object = None
                    elif prev_cx < line_position and cx >= line_position:
                        counter = max(0, counter - 1)
                        print("Left to Right! Counter:", counter)
                        tracking_object = None
            
            tracking_object = (cx, cy)
        else:
            tracking_object = None
        
        # Ajouter le texte du compteur
        cv2.putText(frame, f"Counter: {counter}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Encoder l'image traitée
        _, buffer = cv2.imencode('.jpg', frame)
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        
        with frame_lock:
            latest_frame = frame_base64
        
        return True
    except Exception as e:
        print(f"Erreur lors du traitement de l'image: {e}")
        return False

@app.route('/')
def index():
    return render_template_string('''
<!DOCTYPE html>
<html>
<head>
    <title>Motion Detection - Camera Stream</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body {
            margin: 0;
            padding: 20px;
            font-family: Arial, sans-serif;
            background-color: #1a1a1a;
            color: white;
            text-align: center;
        }
        .container {
            max-width: 800px;
            margin: 0 auto;
        }
        .video-container {
            position: relative;
            margin: 20px 0;
            background-color: #333;
            border-radius: 10px;
            overflow: hidden;
        }
        #localVideo, #processedImage {
            width: 100%;
            max-width: 640px;
            height: auto;
            display: block;
        }
        #processedImage {
            border: 2px solid #4CAF50;
        }
        .controls {
            margin: 20px 0;
        }
        button {
            background-color: #4CAF50;
            color: white;
            border: none;
            padding: 10px 20px;
            margin: 5px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
        }
        button:hover {
            background-color: #45a049;
        }
        button:disabled {
            background-color: #666;
            cursor: not-allowed;
        }
        .counter {
            font-size: 24px;
            font-weight: bold;
            margin: 20px 0;
            color: #4CAF50;
        }
        .status {
            margin: 10px 0;
            padding: 10px;
            border-radius: 5px;
        }
        .status.success {
            background-color: #4CAF50;
        }
        .status.error {
            background-color: #f44336;
        }
        .hidden {
            display: none;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Détection de Mouvement</h1>
        
        <div class="counter">
            Compteur: <span id="counterValue">0</span>
        </div>
        
        <div class="controls">
            <button id="startBtn" onclick="startCamera()">Démarrer la Caméra</button>
            <button id="stopBtn" onclick="stopCamera()" disabled>Arrêter</button>
            <button onclick="resetCounter()">Reset Compteur</button>
        </div>
        
        <div id="status" class="status hidden"></div>
        
        <div class="video-container">
            <video id="localVideo" autoplay muted playsinline class="hidden"></video>
            <img id="processedImage" class="hidden" alt="Image traitée">
        </div>
    </div>

    <script>
        let localVideo = document.getElementById('localVideo');
        let processedImage = document.getElementById('processedImage');
        let stream = null;
        let processing = false;
        let intervalId = null;

        function showStatus(message, type) {
            const statusDiv = document.getElementById('status');
            statusDiv.textContent = message;
            statusDiv.className = `status ${type}`;
            statusDiv.classList.remove('hidden');
            setTimeout(() => {
                statusDiv.classList.add('hidden');
            }, 3000);
        }

        async function startCamera() {
            try {
                stream = await navigator.mediaDevices.getUserMedia({ 
                    video: { 
                        width: { ideal: 640 },
                        height: { ideal: 480 },
                        facingMode: 'environment'
                    } 
                });
                
                localVideo.srcObject = stream;
                localVideo.classList.remove('hidden');
                
                document.getElementById('startBtn').disabled = true;
                document.getElementById('stopBtn').disabled = false;
                
                showStatus('Caméra démarrée avec succès', 'success');
                
                // Commencer le traitement des images
                startProcessing();
                
            } catch (err) {
                console.error('Erreur d\'accès à la caméra:', err);
                showStatus('Erreur d\'accès à la caméra: ' + err.message, 'error');
            }
        }

        function stopCamera() {
            if (stream) {
                stream.getTracks().forEach(track => track.stop());
                stream = null;
            }
            
            localVideo.classList.add('hidden');
            processedImage.classList.add('hidden');
            
            document.getElementById('startBtn').disabled = false;
            document.getElementById('stopBtn').disabled = true;
            
            stopProcessing();
            showStatus('Caméra arrêtée', 'success');
        }

        function startProcessing() {
            if (processing) return;
            processing = true;
            
            intervalId = setInterval(() => {
                captureAndProcess();
            }, 200); // Traiter 5 images par seconde
        }

        function stopProcessing() {
            processing = false;
            if (intervalId) {
                clearInterval(intervalId);
                intervalId = null;
            }
        }

        function captureAndProcess() {
            if (!stream || !processing) return;
            
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');
            
            canvas.width = localVideo.videoWidth || 640;
            canvas.height = localVideo.videoHeight || 480;
            
            ctx.drawImage(localVideo, 0, 0, canvas.width, canvas.height);
            
            const imageData = canvas.toDataURL('image/jpeg', 0.8);
            
            // Envoyer l'image au serveur pour traitement
            fetch('/process_frame', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ frame: imageData })
            })
            .then(response => response.json())
            .then(data => {
                if (data.success && data.processed_frame) {
                    processedImage.src = 'data:image/jpeg;base64,' + data.processed_frame;
                    processedImage.classList.remove('hidden');
                    document.getElementById('counterValue').textContent = data.counter;
                }
            })
            .catch(error => {
                console.error('Erreur lors du traitement:', error);
            });
        }

        function resetCounter() {
            fetch('/reset_counter', { method: 'POST' })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    document.getElementById('counterValue').textContent = '0';
                    showStatus('Compteur remis à zéro', 'success');
                }
            });
        }

        // Arrêter la caméra lors de la fermeture de la page
        window.addEventListener('beforeunload', stopCamera);
    </script>
</body>
</html>
    ''')

@app.route('/process_frame', methods=['POST'])
def process_frame():
    try:
        data = request.get_json()
        frame_data = data.get('frame')
        
        if not frame_data:
            return jsonify({'success': False, 'error': 'No frame data'})
        
        success = process_frame_data(frame_data)
        
        if success:
            with frame_lock:
                processed_frame = latest_frame
            
            return jsonify({
                'success': True,
                'processed_frame': processed_frame,
                'counter': counter
            })
        else:
            return jsonify({'success': False, 'error': 'Frame processing failed'})
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/reset_counter', methods=['POST'])
def reset_counter():
    global counter
    counter = 0
    return jsonify({'success': True, 'counter': counter})

@app.route('/get_counter')
def get_counter():
    return jsonify({'counter': counter})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)