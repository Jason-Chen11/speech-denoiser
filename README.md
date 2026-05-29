# Speech Denoiser

基於深度學習的語音降噪桌面應用系統，整合三種開源降噪模型，採用 Docker 容器化微服務架構，並以 PyQt6 開發跨平台圖形化介面。

開發目的為提供實驗室語音合成訓練資料的前處理工具，支援批次處理大量語料。

---

## 功能

- 三種降噪模型切換（RNNoise、DeepFilterNet、Demucs）
- 單檔處理與批次處理
- 降噪前後 Spectrogram 視覺化對比
- AB 試聽（Original / Denoised）
- 容器狀態即時監控

---

## 系統需求

- Docker Desktop
- Python 3.10+
- ffmpeg

---

## 安裝

### 1. 安裝 ffmpeg

**macOS**
```bash
brew install ffmpeg
```

**Windows**

至 https://ffmpeg.org/download.html 下載並加入 PATH。

**Linux**
```bash
sudo apt install ffmpeg
```

### 2. 安裝 Python 套件

```bash
pip3 install -r requirements.txt
```

### 3. 啟動 Docker 容器

```bash
docker compose up -d
```

第一次啟動會下載並建立三個容器映像，需要約 10～20 分鐘。

確認容器狀態：

```bash
curl http://localhost:8001/health   # RNNoise
curl http://localhost:8002/health   # DeepFilterNet
curl http://localhost:8003/health   # Demucs
```

---

## 執行

```bash
cd gui
python3 demo.py
```

---

## 專案結構

```
speech-denoiser/
├── rnnoise/            # RNNoise 容器（Port 8001）
│   ├── Dockerfile
│   └── app.py
├── deepfilternet/      # DeepFilterNet 容器（Port 8002）
│   ├── Dockerfile
│   └── app.py
├── demucs/             # Demucs 容器（Port 8003）
│   ├── Dockerfile
│   └── app.py
├── audiosr/            # AudioSR 升頻容器（需 GPU，暫未啟用）
│   ├── Dockerfile
│   └── app.py
├── gui/
│   └── demo.py         # PyQt6 主程式
├── add_noise.py        # 加噪腳本（產生測試資料）
├── evaluate.py         # 客觀指標評估腳本
├── evaluation_results.csv
├── docker-compose.yml
└── requirements.txt
```

---

## 評估腳本使用方式

準備測試資料：

```
test-audio/
├── clean/        # 乾淨語音（ground truth）
├── noisy_white/  # 白噪音版本（由 add_noise.py 產生）
├── noisy_pink/   # 粉紅噪音版本（由 add_noise.py 產生）
└── music/        # 背景音樂版本
```

執行加噪：

```bash
python3 add_noise.py
```

執行評估（需容器正在運行）：

```bash
python3 evaluate.py
```

輸出 `evaluation_results.csv`，包含 SNR 改善量、PESQ、STOI。

---

## 模型說明

| 模型 | Port | 特性 | 適用場景 |
|------|------|------|----------|
| RNNoise | 8001 | 輕量快速 | 一般環境噪音 |
| DeepFilterNet | 8002 | 原生 48kHz，高品質 | 通用語音降噪 |
| Demucs | 8003 | 音源分離架構 | 背景音樂去除 |

---

## AudioSR 升頻模組

AudioSR 升頻模組（16kHz → 48kHz）已完成容器化整合，基於 Diffusion Model，需要 NVIDIA GPU 才能正常運行。

啟用方式（需 GPU 環境）：

1. 將 `docker-compose.yml` 加入 audiosr service
2. 修改 `audiosr/app.py` 中的 `device="cpu"` 為 `device="cuda"`
3. 執行 `docker compose up -d audiosr`

---

## 跨平台注意事項

**Windows**
- 需啟用 WSL2
- GUI 中開啟輸出檔案的指令需將 `os.system(f'open ...')` 改為 `os.system(f'start ...')`

**Linux**
- 安裝 Docker Engine 即可，不需要 Docker Desktop
- 開啟檔案指令改為 `xdg-open`

---

## 參考文獻

- Valin, J.-M. (2018). A Hybrid DSP/Deep Learning Approach to Real-Time Full-Band Speech Enhancement. arXiv:1709.08243
- Defossez, A. et al. (2020). Real Time Speech Enhancement in the Waveform Domain. Interspeech 2020.
- Schröter, H. et al. (2022). DeepFilterNet. ICASSP 2022.

---

## 作者

陳俊佑（411286015）  
國立臺北大學通訊工程學系  
指導教授：江振宇