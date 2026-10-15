cat << 'EOF' > ~/docker/face_analytics/README.md
# Face Analytics System - Edge AI (Raspberry Pi 5)

Sistema avanzado de analítica facial, detección de gestos, emociones y métricas biométricas en tiempo real ejecutado en Raspberry Pi 5 usando Docker, MediaPipe Face Mesh y Flask.

## Características
- **Malla Facial en Tiempo Real:** 468 puntos clave 3D.
- **Biometría y Atributos:** Estimación de tono de piel, color de cabello, color de ojos y emociones (Feliz, Neutral, Sorprendido, Serio).
- **Interfaz Web Triple:** Monitoreo (`/`), Analytics de Proximidad (`/analytics`) y Dashboard de Rasgos (`/traits`).

## Despliegue con Docker
```bash
cd ~/docker/face_analytics
docker compose up -d --build