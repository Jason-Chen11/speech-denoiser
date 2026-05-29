"""
evaluate.py
對三個模型跑評估，輸出 SNR / PESQ / STOI 對比表

使用方式：
    python3 evaluate.py

前置條件：
    - Docker 容器正在運行（docker compose up -d）
    - pip3 install pesq pystoi

目錄結構：
    test-audio/clean/         乾淨語音（ground truth）
    test-audio/noisy_white/   白噪音版本
    test-audio/noisy_pink/    粉紅噪音版本
    test-audio/music/         背景音樂版本
"""

import os
import sys
import numpy as np
import soundfile as sf
import librosa
import requests
import io
import subprocess
import tempfile
from pesq import pesq
from pystoi import stoi

# ── 設定 ────────────────────────────────────────────────────
MODELS = {
    "RNNoise":       "http://localhost:8001",
    "DeepFilterNet": "http://localhost:8002",
    "Demucs":        "http://localhost:8003",
}

CLEAN_DIR = "test-audio/clean"
TEST_SETS = {
    "White Noise": "test-audio/noisy_white",
    "Pink Noise":  "test-audio/noisy_pink",
    "Music BG":    "test-audio/music",
}


# ── 工具函式 ─────────────────────────────────────────────────
def load_wav(path, target_sr=16000):
    y, sr = librosa.load(path, sr=target_sr, mono=True)
    return y, sr


def compute_snr(clean, enhanced):
    min_len = min(len(clean), len(enhanced))
    clean, enhanced = clean[:min_len], enhanced[:min_len]
    noise = clean - enhanced
    snr = 10 * np.log10(np.mean(clean ** 2) / (np.mean(noise ** 2) + 1e-10))
    return snr


def compute_pesq(clean, enhanced, sr=16000):
    try:
        min_len = min(len(clean), len(enhanced))
        score = pesq(sr, clean[:min_len], enhanced[:min_len], 'wb')
        return score
    except Exception as e:
        return None


def compute_stoi(clean, enhanced, sr=16000):
    try:
        min_len = min(len(clean), len(enhanced))
        score = stoi(clean[:min_len], enhanced[:min_len], sr, extended=False)
        return score
    except Exception as e:
        return None


def denoise_file(filepath, api_url):
    """送檔案到 API，回傳降噪後的 numpy array"""
    tmp_wav = None
    try:
        process_file = filepath
        if not filepath.lower().endswith('.wav'):
            tmp_wav = filepath.rsplit('.', 1)[0] + '_eval_temp.wav'
            subprocess.run([
                'ffmpeg', '-i', filepath,
                '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1',
                '-y', tmp_wav
            ], capture_output=True, check=True)
            process_file = tmp_wav

        with open(process_file, 'rb') as f:
            resp = requests.post(f"{api_url}/denoise",
                                 files={"file": f}, timeout=180)

        if resp.status_code != 200:
            return None

        audio_io = io.BytesIO(resp.content)
        y, sr = sf.read(audio_io)
        if y.ndim > 1:
            y = y.mean(axis=1)
        return y, sr

    except Exception as e:
        print(f"    降噪失敗：{e}")
        return None
    finally:
        if tmp_wav and os.path.exists(tmp_wav):
            try: os.remove(tmp_wav)
            except: pass


def find_clean(noisy_filename, clean_dir):
    """根據帶噪檔案名稱找對應的乾淨檔案"""
    # noisy 命名規則：{stem}_white_5dB.wav 或 {stem}_pink_5dB.wav
    # music 命名規則：直接比對 stem
    stem = noisy_filename
    for suffix in ['_white_5dB', '_pink_5dB', '_white', '_pink']:
        if stem.endswith(suffix + '.wav'):
            stem = stem[:-len(suffix + '.wav')]
            break
    else:
        stem = os.path.splitext(stem)[0]

    for ext in ['.wav', '.mp3', '.m4a', '.flac']:
        candidate = os.path.join(clean_dir, stem + ext)
        if os.path.exists(candidate):
            return candidate
    return None


# ── 主程式 ───────────────────────────────────────────────────
def main():
    print("Speech Denoiser — 客觀指標評估\n")
    print("=" * 60)

    # 檢查容器
    print("檢查容器狀態...")
    available = []
    for name, url in MODELS.items():
        try:
            r = requests.get(f"{url}/health", timeout=2)
            if r.status_code == 200:
                print(f"  ✓ {name}")
                available.append(name)
            else:
                print(f"  ✗ {name} (離線)")
        except:
            print(f"  ✗ {name} (無法連接)")

    if not available:
        print("\n錯誤：沒有可用的容器，請執行 docker compose up -d")
        sys.exit(1)

    print()

    # 結果儲存
    # results[test_set][model] = {snr_before, snr_after, pesq, stoi, count}
    results = {}

    for set_name, noisy_dir in TEST_SETS.items():
        if not os.path.exists(noisy_dir):
            print(f"跳過 {set_name}：目錄不存在 ({noisy_dir})")
            continue

        noisy_files = [f for f in sorted(os.listdir(noisy_dir))
                       if f.lower().endswith(('.wav', '.mp3', '.m4a', '.flac'))]

        if not noisy_files:
            print(f"跳過 {set_name}：目錄為空")
            continue

        print(f"\n{'─' * 60}")
        print(f"測試集：{set_name}  ({len(noisy_files)} 個檔案)")
        print(f"{'─' * 60}")

        results[set_name] = {}

        for model_name in available:
            api_url = MODELS[model_name]
            snr_improvements = []
            pesq_scores      = []
            stoi_scores      = []
            snr_befores      = []

            print(f"\n  [{model_name}]")

            for fname in noisy_files:
                noisy_path = os.path.join(noisy_dir, fname)

                # 找對應的乾淨檔案（music 測試集沒有 ground truth）
                clean_path = None
                if set_name != "Music BG":
                    clean_path = find_clean(fname, CLEAN_DIR)
                    if not clean_path:
                        print(f"    找不到對應乾淨檔案：{fname}，跳過")
                        continue

                print(f"    處理：{fname}", end="", flush=True)

                result = denoise_file(noisy_path, api_url)
                if result is None:
                    print("  → 失敗")
                    continue

                enhanced, sr_out = result

                if clean_path:
                    clean, sr_clean = load_wav(clean_path, target_sr=16000)
                    noisy, sr_noisy = load_wav(noisy_path, target_sr=16000)
                    enhanced_16k    = librosa.resample(enhanced, orig_sr=sr_out, target_sr=16000)

                    snr_b = compute_snr(clean, noisy)
                    snr_a = compute_snr(clean, enhanced_16k)
                    p     = compute_pesq(clean, enhanced_16k, sr=16000)
                    s     = compute_stoi(clean, enhanced_16k, sr=16000)

                    snr_befores.append(snr_b)
                    snr_improvements.append(snr_a - snr_b)
                    if p is not None: pesq_scores.append(p)
                    if s is not None: stoi_scores.append(s)

                    print(f"  → SNR {snr_a - snr_b:+.1f} dB  PESQ {p:.2f}  STOI {s:.3f}" if p else f"  → SNR {snr_a - snr_b:+.1f} dB")
                else:
                    print("  → 完成（無 ground truth）")

            if snr_improvements:
                results[set_name][model_name] = {
                    'snr_before':     np.mean(snr_befores),
                    'snr_improvement': np.mean(snr_improvements),
                    'pesq':           np.mean(pesq_scores) if pesq_scores else None,
                    'stoi':           np.mean(stoi_scores) if stoi_scores else None,
                    'count':          len(snr_improvements),
                }

    # ── 輸出總表 ────────────────────────────────────────────
    print(f"\n\n{'=' * 60}")
    print("評估結果總表")
    print(f"{'=' * 60}\n")

    for set_name, model_results in results.items():
        if not model_results:
            continue
        print(f"【{set_name}】")
        print(f"  {'模型':<16} {'SNR前(dB)':>10} {'SNR改善':>10} {'PESQ':>8} {'STOI':>8} {'樣本數':>6}")
        print(f"  {'─'*16} {'─'*10} {'─'*10} {'─'*8} {'─'*8} {'─'*6}")
        for model_name, r in model_results.items():
            pesq_str = f"{r['pesq']:.2f}" if r['pesq'] else "  N/A"
            stoi_str = f"{r['stoi']:.3f}" if r['stoi'] else "  N/A"
            print(f"  {model_name:<16} {r['snr_before']:>10.1f} {r['snr_improvement']:>+10.1f} {pesq_str:>8} {stoi_str:>8} {r['count']:>6}")
        print()

    # 儲存 CSV
    csv_path = "evaluation_results.csv"
    with open(csv_path, 'w') as f:
        f.write("test_set,model,snr_before_dB,snr_improvement_dB,pesq,stoi,n_samples\n")
        for set_name, model_results in results.items():
            for model_name, r in model_results.items():
                pesq_val = f"{r['pesq']:.4f}" if r['pesq'] else ""
                stoi_val = f"{r['stoi']:.4f}" if r['stoi'] else ""
                f.write(f"{set_name},{model_name},{r['snr_before']:.2f},"
                        f"{r['snr_improvement']:.2f},{pesq_val},{stoi_val},{r['count']}\n")

    print(f"CSV 已儲存：{csv_path}")


if __name__ == "__main__":
    main()
