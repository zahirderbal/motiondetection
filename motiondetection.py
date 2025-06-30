from flask import Flask, Response, render_template_string, request, jsonify
import numpy as np
import threading
import base64
import io
from PIL import Image, ImageDraw, ImageFont
import os

app = Flask(__name__)

# Variables globales
counter = 0
frame_width = 640
frame_height = 480
line_position = frame_width // 2
tracking_object = None
latest_frame = None
frame_lock = threading.Lock()
background_frames = []
background_model = None

def create_background_model(frames):
    """Crée un modèle d'arrière-plan simple"""
    if len(frames) < 5:
        return None
    
    # Convertir en arrays numpy
    arrays = [np.array(frame) for frame in frames]
    
    # Calculer la médiane pour chaque pixel
    background = np.median(arrays, axis=0).astype(np.uint8)
    return Image.fromarray(background)

def detect_motion(current_frame, background):
    """Détecte le mouvement en comparant avec l'arrière-plan"""
    if background is None:
        return None, None
    
    # Convertir en arrays numpy
    current_array = np.array(current_frame.convert('L'))  # Convertir en niveaux de gris
    background_array = np.array(background.convert('L'))
    
    # Calculer la différence
    diff = np.abs(current_array.astype(np.int16) - background_array.astype(np.int16))
    
    # Seuillage
    threshold = 30
    motion_mask = diff > threshold
    
    # Trouver le centre de masse du mouvement
    if np.sum(motion_mask) > 500:  # Seuil minimum de pixels en mouvement
        y_coords, x_coords = np.where(motion_mask)
        if len(x_coords) > 0:
            center_x = int(np.mean(x_coords))
            center_y = int(np.mean(y_coords))
            area = np.sum(motion_mask)
            return (center_x, center_y), area
    
    return None, None

def process_frame_data(frame_data):
    """Traite les données d'image reçues du client"""
    global counter, tracking_object, latest_frame, background_frames, background_model
    
    try:
        # Décoder l'image base64
        image_data = base64.b64decode(frame_data.split(',')[1])
        image = Image.open(io.BytesIO(image_data))
        frame = image.resize((frame_width, frame_height))
        
        # Mettre à jour le modèle d'arrière-plan
        background_frames.append(frame.copy())
        if len(background_frames) > 10:
            background_frames.pop(0)
        
        if len(background_frames) >= 5 and background_model is None:
            background_model = create_background_model(background_frames)
        elif len(background_frames) >= 10:
            # Mettre à jour occasionnellement le modèle d'arrière-plan
            if len(background_frames) % 5 == 0:
                background_model = create_background_model(background_frames[-5:])
        
        # Créer une copie pour dessiner
        draw_frame = frame.copy()
        draw = ImageDraw.Draw(draw_frame)
        
        # Dessiner la ligne de comptage
        draw.line([(line_position, 0), (line_position, frame_height)], fill=(255, 0, 0), width=3)
        
        # Détecter le mouvement
        motion_center, motion_area = detect_motion(frame, background_model)
        
        if motion_center is not None and motion_area > 1000:
            cx, cy = motion_center
            
            # Dessiner le centre du mouvement
            draw.ellipse([cx-10, cy-10, cx+10, cy+10], fill=(0, 0, 255))
            
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
        try:
            font = ImageFont.load_default()
            draw.text((20, 20), f"Counter: {counter}", fill=(0, 0, 255), font=font)
        except:
            draw.text((20, 20), f"Counter: {counter}", fill=(0, 0, 255))
        
        # Encoder l'image traitée
        img_buffer = io.BytesIO()
        draw_frame.save(img_buffer, format='JPEG', quality=80)
        frame_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
        
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
    <title>Motion Detection - Simple Version</title>
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
        <h1>Détection de Mouvement - Version Simple</h1>
        
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
            }, 300);
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
            
            const imageData = canvas.toDataURL('image/jpeg', 0.7);
            
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
    global counter, background_model, background_frames
    counter = 0
    background_model = None
    background_frames = []
    return jsonify({'success': True, 'counter': counter})

@app.route('/get_counter')
def get_counter():
    return jsonify({'counter': counter})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)