# MuseumLangID 🏛️

MuseumLangID is a REST API designed for automatic language recognition of museum-related texts. It uses a Machine Learning model (Scikit-Learn) deployed via FastAPI to provide low-latency predictions.

## 🌟 Features
- **FastAPI Backend**: Modern, fast (high-performance) framework for building APIs.
- **Museum-Specific Focus**: Trained on a specialized dataset of museum descriptions and labels.
- **Production Ready**: Includes professional logging, lifecycle management, and Docker support.
- **Pre-trained Model**: Uses a serialized Scikit-learn pipeline for efficient inference.

## 📁 Project Structure
```text
MuseumLangID/
├── data/           # Dataset files (TSV)
├── logs/           # Application execution logs
├── models/         # Pre-trained ML models (.pkl)
├── src/            # Application source code
│   └── main.py     # FastAPI main entry point
├── tests/          # Unit and integration tests
├── Dockerfile      # Containerization configuration
└── requirements.txt # Python dependencies
```

## 🚀 Getting Started

### Prerequisites
- Python 3.12+
- (Optional) Docker

### Local Installation
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the server:
   ```bash
   python src/main.py
   ```
   The API will be available at `http://127.0.0.1:8000`.

### Using Docker
1. Build the image:
   ```bash
   docker build -t museum-lang-id .
   ```
2. Run the container:
   ```bash
   docker run -p 8000:8000 museum-lang-id
   ```

## 🛠️ API Endpoints
- `GET /`: API welcome and version info.
- `POST /identify-language`: Predict language for a given text.
- `GET /health`: Health check endpoint.
- `GET /docs`: Interactive API documentation (Swagger UI).

## 📄 License
MIT
