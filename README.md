# SnapScout AI Backend

This is the FastAPI backend for SnapScout.
It provides high-accuracy image labeling using Google's Vision Transformer.

## Endpoints

### `GET /`
Check that the server is running.

### `POST /detect`
Send an image and receive top 5 predicted labels.

Example (cURL):

```bash
curl -X POST "https://your-render-url/detect" \
  -F "file=@example.jpg"
