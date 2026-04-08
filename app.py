from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from TTS.api import TTS
import tempfile
import uuid
import os
import requests

app = FastAPI()

API_KEY = os.getenv("COQUI_TTS_API_KEY", "")
MODEL_NAME = os.getenv("COQUI_MODEL_NAME", "tts_models/multilingual/multi-dataset/xtts_v2")

tts = TTS(MODEL_NAME).to("cpu")

class GenerateRequest(BaseModel):
    text: str
    language: str = "pt"
    speaker_audio_url: str
    memory_id: str
    user_id: str

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/generate")
def generate(req: GenerateRequest, x_api_key: str | None = Header(default=None)):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="invalid api key")

    ref_path = None
    out_path = None

    try:
        ref_fd, ref_path = tempfile.mkstemp(suffix=".wav")
        os.close(ref_fd)

        audio_resp = requests.get(req.speaker_audio_url)
        with open(ref_path, "wb") as f:
            f.write(audio_resp.content)

        out_path = f"{req.user_id}-{uuid.uuid4()}.wav"

        tts.tts_to_file(
            text=req.text,
            speaker_wav=ref_path,
            language=req.language,
            file_path=out_path,
        )

        return FileResponse(out_path, media_type="audio/wav")

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
