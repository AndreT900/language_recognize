"""
MuseumLangAPI - API REST per il riconoscimento della lingua di testi museali
"""

import os
import pickle
import logging
from datetime import datetime
from typing import Dict
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import numpy as np

# Configurazione dei percorsi
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "language_detection_pipeline.pkl")
LOG_DIR = os.path.join(BASE_DIR, "logs")
LOG_PATH = os.path.join(LOG_DIR, "museum_lang_api.log")

# Assicura che la directory dei log esista
os.makedirs(LOG_DIR, exist_ok=True)

# Configurazione del logging
logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("museum_lang_api")

# Variabile globale per il modello
model = None

# Funzione per caricare il modello
def load_model() -> object:
    try:
        if not os.path.exists(MODEL_PATH):
            error_msg = f"File del modello non trovato in: {MODEL_PATH}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
            
        with open(MODEL_PATH, 'rb') as model_file:
            loaded_model = pickle.load(model_file)
            logger.info("Modello caricato con successo")
            return loaded_model
    except Exception as e:
        logger.error(f"Errore durante il caricamento del modello: {str(e)}")
        raise RuntimeError(f"Impossibile caricare il modello: {str(e)}")

# Context manager per il ciclo di vita dell'applicazione
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    model = load_model()
    logger.info("API inizializzata e pronta")
    yield  
    logger.info("Spegnimento dell'API")

# Creazione dell'applicazione FastAPI con il context manager lifespan
app = FastAPI(
    title="MuseumLangAPI",
    description="API per il riconoscimento automatico della lingua dei testi museali",
    version="1.0.0",
    lifespan=lifespan
)

# Modelli di dati per l'API
class LanguageRequest(BaseModel):
    text: str = Field(..., example="Questo è un esempio di testo.")


class LanguageResponse(BaseModel):
    """Modello per la risposta con la lingua identificata."""
    language_code: str = Field(..., example="IT")
    confidence: float = Field(..., example=0.98)


# Endpoint principale per la pagina root
@app.get("/")
async def root():
    return {
        "message": "Benvenuto all'API di MuseumLangID",
        "version": "1.0.0",
        "endpoints": {
            "identify_language": "/identify-language",
            "health": "/health",
            "docs": "/docs"
        }
    }


# Endpoint per l'identificazione della lingua
@app.post("/identify-language", response_model=LanguageResponse)
async def identify_language(request: LanguageRequest) -> Dict:

    logger.info(f"Richiesta ricevuta: {request.text[:100]}...")
    

    if not request.text or request.text.strip() == "":
        error_msg = "Il testo non può essere vuoto"
        logger.warning(error_msg)
        raise HTTPException(status_code=400, detail=error_msg)
    
    try:
        text_to_predict = [request.text]
        
        predicted_language = model.predict(text_to_predict)
        
        confidence = 0.9  
        try:
            probabilities = model.predict_proba(text_to_predict)
            confidence = float(max(probabilities[0]))
        except (AttributeError, NotImplementedError):
            pass
        
        response = {
            "language_code": predicted_language[0],
            "confidence": round(confidence, 2)
        }
        
        logger.info(f"Risposta: {response}")
        
        return response
    
    except Exception as e:
        error_msg = f"Errore durante l'identificazione della lingua: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)


# Endpoint per il controllo dello stato di salute dell'API
@app.get("/health")
async def health_check() -> Dict:
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)