from fastapi import FastAPI, File, UploadFile
from fastapi.responses import Response
import torch
import torchaudio
import io
import subprocess
import tempfile
import os

app = FastAPI(title="AudioSR Upsampling Service")

model = None

def get_model():
    global model
    if model is None:
        from audiosr import build_model
        model = build_model(model_name="speech", device="cpu")
    return model

@app.get("/")
def root():
    return {
        "service": "AudioSR Upsampling API",
        "status": "running",
        "version": "1.0.0",
        "model": "AudioSR-Speech"
    }

@app.get("/health")
def health():
    return {"status": "healthy", "model": "AudioSR-Speech"}

@app.post("/upsample")
async def upsample(file: UploadFile = File(...)):
    try:
        m = get_model()
        audio_bytes = await file.read()

        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_in:
            tmp_in_path = tmp_in.name
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_out:
            tmp_out_path = tmp_out.name

        try:
            with open(tmp_in_path, 'wb') as f:
                f.write(audio_bytes)

            subprocess.run([
                'ffmpeg', '-i', tmp_in_path,
                '-acodec', 'pcm_s16le',
                '-ar', '16000',
                '-ac', '1',
                '-y', tmp_out_path
            ], capture_output=True, check=True)

            from audiosr import super_resolution
            output_waveform = super_resolution(
                m,
                tmp_out_path,
                seed=42,
                guidance_scale=3.5,
                ddim_steps=25,
                latent_t_per_second=12.8
            )

            output_io = io.BytesIO()
            torchaudio.save(output_io, torch.tensor(output_waveform).unsqueeze(0), 48000, format="wav")
            output_io.seek(0)

        finally:
            os.unlink(tmp_in_path)
            os.unlink(tmp_out_path)

        return Response(
            content=output_io.read(),
            media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=upsampled.wav"}
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}, 500

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004, timeout_keep_alive=1800)