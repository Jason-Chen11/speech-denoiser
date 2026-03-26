from fastapi import FastAPI, File, UploadFile
from fastapi.responses import Response
import noisereduce as nr
import soundfile as sf
import io

app = FastAPI(title="RNNoise Denoising Service")

@app.get("/")
def root():
    return {
        "service": "RNNoise Denoising API",
        "status": "running",
        "version": "1.0.0",
        "model": "RNNoise (noisereduce)"
    }

@app.get("/health")
def health():
    return {"status": "healthy", "model": "RNNoise"}

@app.post("/denoise")
async def denoise(file: UploadFile = File(...)):
    try:
        # 讀取上傳的音訊
        audio_bytes = await file.read()
        data, sr = sf.read(io.BytesIO(audio_bytes))
        
        # 降噪處理
        reduced = nr.reduce_noise(y=data, sr=sr, prop_decrease=1.0)
        
        # 轉換回 bytes
        output_io = io.BytesIO()
        sf.write(output_io, reduced, sr, format='WAV')
        output_io.seek(0)
        
        return Response(
            content=output_io.read(),
            media_type="audio/wav",
            headers={
                "Content-Disposition": "attachment; filename=denoised.wav"
            }
        )
    except Exception as e:
        return {"error": str(e)}, 500

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)