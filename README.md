# Speech Denoiser

## 快速開始

### 環境需求

- macOS / Windows / Linux
- Docker Desktop
- Python 3.10+
  
### 安裝步驟

1. **Clone 專案**
```bash
git clone https://github.com/JC-9311/speech-denoiser.git
cd speech-denoiser
```

2. **安裝 Python 套件**
```bash
pip3 install PyQt6 requests
```

3. **建立 Docker 映像檔**
```bash
cd rnnoise
docker build -t rnnoise-service:v1 .
```

4. **啟動容器**
```bash
docker run -d -p 8001:8001 --name rnnoise rnnoise-service:v1
```

5. **啟動 GUI**
```bash
cd ../gui
python3 demo.py
```