import cv2
import mediapipe as mp
import numpy as np
import time
from datetime import datetime
from flask import Flask, Response, render_template_string, jsonify

app = Flask(__name__)

# MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=5,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Estructura de estadísticas para emociones y atributos
analytics_data = {
    'total_detections': 0,
    'current_faces': 0,
    'history_timeline': [],
    'emotions': {'Feliz': 0, 'Neutral': 0, 'Sorprendido': 0, 'Serio': 0},
    'skin_tone': {'Clara': 0, 'Media': 0, 'Oscura': 0},
    'hair_color': {'Negro/Castaño': 0, 'Rubio/Claro': 0, 'Pelirrojo/Otro': 0},
    'eye_color': {'Oscuros': 0, 'Claros/Azules/Verdes': 0}
}

def euclidean_dist(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def get_hsv_crop(frame, landmarks, indices, h, w):
    pts = np.array([(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in indices], np.int32)
    x, y, bw, bh = cv2.boundingRect(pts)
    x, y = max(0, x), max(0, y)
    crop = frame[y:y+bh, x:x+bw]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    return np.mean(hsv, axis=(0,1))

# Índices para Malla Facial
CHEEK_INDICES = [117, 118, 101, 205, 50]
HAIR_INDICES = [10, 338, 297, 67, 109]
IRIS_INDICES = [468, 469, 470, 471, 472]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
LEFT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [61, 185, 40, 39, 291, 269, 270, 267]

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

def process_facial_analytics():
    global analytics_data
    last_log_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.03)
            continue
            
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        h, w, _ = frame.shape
        current_count = 0
        
        if results.multi_face_landmarks:
            current_count = len(results.multi_face_landmarks)
            for face_landmarks in results.multi_face_landmarks:
                lms = face_landmarks.landmark
                
                # 1. Dibujar Malla
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                )
                
                # 2. Análisis de Emociones mediante ratios geométricos
                # MAR (Mouth Aspect Ratio) y Eyebrow Ratio
                pts_m = [(int(lms[i].x * w), int(lms[i].y * h)) for i in MOUTH]
                p2_p8 = euclidean_dist(pts_m[1], pts_m[7])
                p1_p5 = euclidean_dist(pts_m[0], pts_m[4])
                mar = p2_p8 / p1_p5 if p1_p5 > 0 else 0
                
                if mar > 0.45:
                    emotion = 'Sorprendido'
                elif mar > 0.25:
                    emotion = 'Feliz'
                elif mar < 0.10:
                    emotion = 'Serio'
                else:
                    emotion = 'Neutral'
                
                analytics_data['emotions'][emotion] += 1
                
                # 3. Tonalidad de Piel (Análisis HSV en la mejilla)
                skin_hsv = get_hsv_crop(frame, lms, CHEEK_INDICES, h, w)
                if skin_hsv is not None:
                    v_val = skin_hsv[2]
                    if v_val > 170:
                        skin_tone = 'Clara'
                    elif v_val > 110:
                        skin_tone = 'Media'
                    else:
                        skin_tone = 'Oscura'
                    analytics_data['skin_tone'][skin_tone] += 1
                else:
                    skin_tone = 'Media'
                
                # 4. Color de Cabello (Análisis HSV en la frente superior)
                hair_hsv = get_hsv_crop(frame, lms, HAIR_INDICES, h, w)
                if hair_hsv is not None:
                    h_val, s_val, v_val = hair_hsv[0], hair_hsv[1], hair_hsv[2]
                    if v_val < 80:
                        hair_color = 'Negro/Castaño'
                    elif h_val < 30 and s_val > 50:
                        hair_color = 'Pelirrojo/Otro'
                    else:
                        hair_color = 'Rubio/Claro'
                    analytics_data['hair_color'][hair_color] += 1
                    
                # 5. Color de Ojos
                eye_hsv = get_hsv_crop(frame, lms, IRIS_INDICES, h, w)
                if eye_hsv is not None:
                    eye_v = eye_hsv[2]
                    eye_color = 'Claros/Azules/Verdes' if eye_v > 130 else 'Oscuros'
                    analytics_data['eye_color'][eye_color] += 1
                
                # Renderizado de Bounding Box y Texto
                x_coords = [int(lm.x * w) for lm in lms]
                y_coords = [int(lm.y * h) for lm in lms]
                xmin, xmax = min(x_coords), max(x_coords)
                ymin, ymax = min(y_coords), max(y_coords)
                
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 127), 2)
                lbl = f"{emotion} | Piel: {skin_tone}"
                cv2.putText(frame, lbl, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 127), 2)

        analytics_data['current_faces'] = current_count
        
        if time.time() - last_log_time >= 3:
            analytics_data['total_detections'] += current_count
            now = datetime.now().strftime("%H:%M:%S")
            analytics_data['history_timeline'].append({'time': now, 'count': current_count})
            if len(analytics_data['history_timeline']) > 20:
                analytics_data['history_timeline'].pop(0)
            last_log_time = time.time()
            
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

INDEX_HTML = """
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8"><title>Face Analytics System</title>
    <style>
        body { font-family: 'Segoe UI', Arial; background: #0f172a; color: #f8fafc; margin: 0; padding: 20px; }
        .header { display: flex; justify-content: space-between; border-bottom: 2px solid #334155; padding-bottom: 15px; }
        .nav-btn { background: #3b82f6; color: white; padding: 10px 15px; border-radius: 8px; text-decoration: none; font-weight: bold; margin-left: 10px; }
        .container { display: flex; gap: 20px; margin-top: 20px; flex-wrap: wrap; }
        .video-box { flex: 2; min-width: 320px; background: #1e293b; padding: 15px; border-radius: 12px; }
        .video-box img { width: 100%; border-radius: 8px; }
    </style>
</head>
<body>
    <div class="header">
        <h2>Face Analytics - Malla & Reconocimiento Facial</h2>
        <div>
            <a href="/traits" class="nav-btn" style="background:#8b5cf6;">Ver Estadísticas de Rasgos</a>
        </div>
    </div>
    <div class="container">
        <div class="video-box">
            <img src="/video_feed" alt="Transmision en vivo">
        </div>
    </div>
</body>
</html>
"""

TRAITS_HTML = """
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8"><title>Estadísticas de Rasgos & Emociones</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body { font-family: 'Segoe UI', Arial; background: #0f172a; color: #f8fafc; margin: 0; padding: 20px; }
        .header { display: flex; justify-content: space-between; border-bottom: 2px solid #334155; padding-bottom: 15px; }
        .nav-btn { background: #64748b; color: white; padding: 10px 15px; border-radius: 8px; text-decoration: none; font-weight: bold; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px; margin-top: 20px; }
        .chart-box { background: #1e293b; padding: 20px; border-radius: 12px; }
    </style>
</head>
<body>
    <div class="header">
        <h2>Dashboard de Emociones y Atributos Faciales (Barras y Puntos)</h2>
        <a href="/" class="nav-btn">Regresar al Monitoreo</a>
    </div>
    <div class="grid">
        <div class="chart-box">
            <h3>Distribución de Emociones (Gráfica de Barras)</h3>
            <canvas id="emotionsChart"></canvas>
        </div>
        <div class="chart-box">
            <h3>Tonalidad de Piel (Gráfica de Barras)</h3>
            <canvas id="skinChart"></canvas>
        </div>
        <div class="chart-box">
            <h3>Color de Cabello (Gráfica de Barras)</h3>
            <canvas id="hairChart"></canvas>
        </div>
        <div class="chart-box">
            <h3>Color de Ojos (Gráfica de Barras)</h3>
            <canvas id="eyeChart"></canvas>
        </div>
    </div>
    <script>
        function createBarChart(ctx, label, colors) {
            return new Chart(ctx, {
                type: 'bar',
                data: { labels: [], datasets: [{ label: label, data: [], backgroundColor: colors }] },
                options: { responsive: true, scales: { y: { beginAtZero: true } } }
            });
        }

        const emotionsChart = createBarChart(document.getElementById('emotionsChart'), 'Conteo', ['#22c55e', '#38bdf8', '#eab308', '#ef4444']);
        const skinChart = createBarChart(document.getElementById('skinChart'), 'Conteo', ['#fde047', '#f97316', '#78350f']);
        const hairChart = createBarChart(document.getElementById('hairChart'), 'Conteo', ['#1e293b', '#facc15', '#b91c1c']);
        const eyeChart = createBarChart(document.getElementById('eyeChart'), 'Conteo', ['#334155', '#38bdf8']);

        setInterval(() => {
            fetch('/api/data')
                .then(r => r.json())
                .then(d => {
                    // Actualizar Emociones
                    emotionsChart.data.labels = Object.keys(d.emotions);
                    emotionsChart.data.datasets[0].data = Object.values(d.emotions);
                    emotionsChart.update();

                    // Actualizar Piel
                    skinChart.data.labels = Object.keys(d.skin_tone);
                    skinChart.data.datasets[0].data = Object.values(d.skin_tone);
                    skinChart.update();

                    // Actualizar Cabello
                    hairChart.data.labels = Object.keys(d.hair_color);
                    hairChart.data.datasets[0].data = Object.values(d.hair_color);
                    hairChart.update();

                    // Actualizar Ojos
                    eyeChart.data.labels = Object.keys(d.eye_color);
                    eyeChart.data.datasets[0].data = Object.values(d.eye_color);
                    eyeChart.update();
                });
        }, 1500);
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(INDEX_HTML)

@app.route('/traits')
def traits():
    return render_template_string(TRAITS_HTML)

@app.route('/video_feed')
def video_feed():
    return Response(process_facial_analytics(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/data')
def api_data():
    return jsonify(analytics_data)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
