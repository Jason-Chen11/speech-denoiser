from fastapi import FastAPI, File, UploadFile
from fastapi.responses import Response
from df.enhance import enhance, init_df, load_audio, save_audio
import io
import subprocess
import tempfile
import os
import torchaudio
import torch

app = FastAPI(title="DeepFilterNet Denoising Service")

model, df_state, _ = init_df()

@app.get("/")
def root():
    return {
        "service": "DeepFilterNet Denoising API",
        "status": "running",
        "version": "1.0.0",
        "model": "DeepFilterNet3"
    }

@app.get("/health")
def health():
    return {"status": "healthy", "model": "DeepFilterNet3"}

@app.post("/denoise")
async def denoise(file: UploadFile = File(...)):
    try:
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
                '-ar', '48000',
                '-ac', '1',
                '-y', tmp_out_path
            ], capture_output=True, check=True)

            audio, _ = load_audio(tmp_out_path, sr=df_state.sr())
            enhanced = enhance(model, df_state, audio)

            output_io = io.BytesIO()
            if enhanced.dim() == 1:
                enhanced = enhanced.unsqueeze(0)
            torchaudio.save(output_io, enhanced, df_state.sr(), format="wav")
            output_io.seek(0)

        finally:
            os.unlink(tmp_in_path)
            os.unlink(tmp_out_path)

        return Response(
            content=output_io.read(),
            media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=denoised.wav"}
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}, 500

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)