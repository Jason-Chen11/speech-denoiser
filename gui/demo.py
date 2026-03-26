from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QLabel, 
    QFileDialog, QVBoxLayout, QWidget, QProgressBar,
    QMessageBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
import requests
import sys
import os
import subprocess
import shutil

def check_ffmpeg():
    return shutil.which('ffmpeg') is not None

CONVERTER_AVAILABLE = check_ffmpeg()

class DenoiseWorker(QThread):
    finished = pyqtSignal(str)
    error = pyqtSignal(str)
    progress = pyqtSignal(str)
    
    def __init__(self, input_file):
        super().__init__()
        self.input_file = input_file
    
    def run(self):
        try:
            process_file = self.input_file
            temp_wav = None
            
            if not self.input_file.lower().endswith('.wav'):
                if not CONVERTER_AVAILABLE:
                    self.error.emit("缺少 ffmpeg！\n請執行：brew install ffmpeg")
                    return
                
                self.progress.emit("⏳ 正在轉換音訊格式...")
                
                temp_wav = self.input_file.rsplit('.', 1)[0] + '_temp.wav'
                
                try:
                    result = subprocess.run([
                        'ffmpeg', '-i', self.input_file,
                        '-acodec', 'pcm_s16le',
                        '-ar', '16000',
                        '-ac', '1',
                        '-y',
                        temp_wav
                    ], capture_output=True, text=True, timeout=30)
                    
                    if result.returncode != 0:
                        raise Exception("ffmpeg 轉換失敗")
                    
                    process_file = temp_wav
                    self.progress.emit("✓ 格式轉換完成，開始降噪...")
                except subprocess.TimeoutExpired:
                    self.error.emit("轉換超時，檔案可能太大")
                    return
                except Exception as e:
                    self.error.emit(f"格式轉換失敗：{str(e)}\n\n請確認已安裝 ffmpeg：\nbrew install ffmpeg")
                    return
            
            with open(process_file, 'rb') as f:
                response = requests.post(
                    "http://localhost:8001/denoise",
                    files={"file": f},
                    timeout=60
                )
            
            if response.status_code == 200:
                output = self.input_file.rsplit('.', 1)[0] + '_降噪.wav'
                with open(output, 'wb') as f:
                    f.write(response.content)
                
                if temp_wav and os.path.exists(temp_wav):
                    os.remove(temp_wav)
                
                self.finished.emit(output)
            else:
                self.error.emit(f"API 錯誤：{response.status_code}")
                
        except requests.exceptions.ConnectionError:
            self.error.emit("無法連接到 Docker 容器！\n請確認容器正在運行：docker ps")
        except Exception as e:
            self.error.emit(str(e))
        finally:
            if temp_wav and os.path.exists(temp_wav):
                try:
                    os.remove(temp_wav)
                except:
                    pass

class DemoWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🎵 語音降噪系統 - 專題 Demo")
        self.setGeometry(200, 200, 650, 420)
        
        self.setStyleSheet("""
            QMainWindow {
                background: #1e1e1e;
            }
            QLabel {
                color: #ffffff;
            }
        """)
        
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setSpacing(15)
        layout.setContentsMargins(30, 30, 30, 30)
        
        title = QLabel("🎵 語音降噪系統")
        title.setStyleSheet("""
            font-size: 28px; 
            font-weight: bold; 
            padding: 20px;
            color: #4CAF50;
        """)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        subtitle = QLabel("Docker 容器化 + PyQt6 圖形介面 + 自動格式轉換")
        subtitle.setStyleSheet("""
            font-size: 14px; 
            color: #888; 
            padding-bottom: 10px;
        """)
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(subtitle)
        
        self.file_label = QLabel("📂 尚未選擇檔案")
        self.file_label.setStyleSheet("""
            padding: 20px; 
            background: #2d2d2d; 
            border-radius: 10px; 
            border: 2px dashed #555;
            font-size: 14px;
            color: #aaa;
        """)
        self.file_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.file_label)
        
        btn_select = QPushButton("📁 選擇音訊檔案 (支援 WAV/MP3/M4A/FLAC/OGG/AAC)")
        btn_select.setStyleSheet("""
            QPushButton {
                padding: 15px; 
                font-size: 15px; 
                background: #2196F3; 
                color: white; 
                border-radius: 8px; 
                border: none;
                font-weight: bold;
            }
            QPushButton:hover {
                background: #1976D2;
            }
            QPushButton:pressed {
                background: #0D47A1;
            }
        """)
        btn_select.clicked.connect(self.select_file)
        layout.addWidget(btn_select)
        
        self.btn_denoise = QPushButton("🎵 開始降噪（RNNoise 模型）")
        self.btn_denoise.setStyleSheet("""
            QPushButton {
                padding: 15px; 
                font-size: 15px; 
                background: #4CAF50; 
                color: white; 
                border-radius: 8px; 
                border: none;
                font-weight: bold;
            }
            QPushButton:hover {
                background: #45a049;
            }
            QPushButton:pressed {
                background: #388E3C;
            }
            QPushButton:disabled {
                background: #555;
                color: #888;
            }
        """)
        self.btn_denoise.clicked.connect(self.start_denoise)
        self.btn_denoise.setEnabled(False)
        layout.addWidget(self.btn_denoise)
        
        self.progress = QProgressBar()
        self.progress.setStyleSheet("""
            QProgressBar {
                border: 2px solid #555;
                border-radius: 5px;
                text-align: center;
                height: 25px;
                background: #2d2d2d;
                color: white;
            }
            QProgressBar::chunk {
                background: #4CAF50;
                border-radius: 3px;
            }
        """)
        self.progress.setVisible(False)
        layout.addWidget(self.progress)
        
        self.status = QLabel("💡 請選擇音訊檔案開始")
        self.status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status.setStyleSheet("""
            padding: 15px; 
            color: #888;
            font-size: 13px;
        """)
        layout.addWidget(self.status)
        
        layout.addStretch()
        
        self.input_file = None
        self.worker = None
        
        self.check_docker()
        self.check_converter()
    
    def check_docker(self):
        try:
            response = requests.get("http://localhost:8001/health", timeout=2)
            if response.status_code == 200:
                self.status.setText("✅ Docker 容器連接正常")
                self.status.setStyleSheet("padding: 15px; color: #4CAF50; font-size: 13px;")
            else:
                self.status.setText("⚠️ Docker 容器回應異常")
                self.status.setStyleSheet("padding: 15px; color: #FF9800; font-size: 13px;")
        except:
            self.status.setText("❌ Docker 容器未運行！\n請執行：docker run -d -p 8001:8001 rnnoise-service:v1")
            self.status.setStyleSheet("padding: 15px; color: #f44336; font-size: 13px;")
    
    def check_converter(self):
        if not CONVERTER_AVAILABLE:
            msg = "⚠️ 未安裝 ffmpeg，無法轉換音訊格式\n只能使用 .wav 檔案\n\n安裝方法：brew install ffmpeg"
            print(msg)
    
    def select_file(self):
        file, _ = QFileDialog.getOpenFileName(
            self, 
            "選擇音訊檔案", 
            "", 
            "音訊檔案 (*.wav *.mp3 *.m4a *.flac *.ogg *.aac);;所有檔案 (*.*)"
        )
        if file:
            self.input_file = file
            filename = file.split('/')[-1]
            
            if not file.lower().endswith('.wav'):
                if CONVERTER_AVAILABLE:
                    self.file_label.setText(f"📄 已選擇：{filename}\n⚙️ 將自動轉換為 WAV 格式")
                else:
                    self.file_label.setText(f"📄 已選擇：{filename}\n❌ 需要安裝 ffmpeg 才能轉換")
                    self.file_label.setStyleSheet("""
                        padding: 20px; 
                        background: #5d1f1f; 
                        border-radius: 10px; 
                        border: 2px solid #f44336;
                        font-size: 14px;
                        color: #ffcdd2;
                    """)
                    self.btn_denoise.setEnabled(False)
                    self.status.setText("❌ 請安裝 ffmpeg：brew install ffmpeg")
                    self.status.setStyleSheet("padding: 15px; color: #f44336; font-size: 13px;")
                    return
            else:
                self.file_label.setText(f"📄 已選擇：{filename}")
            
            self.file_label.setStyleSheet("""
                padding: 20px; 
                background: #1b5e20; 
                border-radius: 10px; 
                border: 2px solid #4CAF50;
                font-size: 14px;
                color: #a5d6a7;
            """)
            self.btn_denoise.setEnabled(True)
            self.status.setText("✓ 可以開始降噪")
            self.status.setStyleSheet("padding: 15px; color: #4CAF50; font-size: 13px;")
    
    def start_denoise(self):
        if not self.input_file:
            return
        
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)
        self.status.setText("⏳ 正在處理中，請稍候...")
        self.status.setStyleSheet("padding: 15px; color: #2196F3; font-size: 13px;")
        self.btn_denoise.setEnabled(False)
        
        self.worker = DenoiseWorker(self.input_file)
        self.worker.finished.connect(self.on_success)
        self.worker.error.connect(self.on_error)
        self.worker.progress.connect(self.on_progress)
        self.worker.start()
    
    def on_progress(self, message):
        self.status.setText(message)
        self.status.setStyleSheet("padding: 15px; color: #2196F3; font-size: 13px;")
    
    def on_success(self, output_file):
        self.progress.setVisible(False)
        self.btn_denoise.setEnabled(True)
        self.status.setText("✅ 處理完成！")
        self.status.setStyleSheet("padding: 15px; color: #4CAF50; font-size: 13px;")
        
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Icon.Information)
        msg.setWindowTitle("🎉 處理完成")
        msg.setText(f"降噪成功！\n\n輸出檔案：\n{output_file}")
        msg.setStandardButtons(
            QMessageBox.StandardButton.Open | 
            QMessageBox.StandardButton.Ok
        )
        
        result = msg.exec()
        if result == QMessageBox.StandardButton.Open:
            os.system(f'open "{output_file}"')
    
    def on_error(self, error_msg):
        self.progress.setVisible(False)
        self.btn_denoise.setEnabled(True)
        self.status.setText("❌ 處理失敗")
        self.status.setStyleSheet("padding: 15px; color: #f44336; font-size: 13px;")
        
        QMessageBox.critical(self, "錯誤", f"處理失敗：\n\n{error_msg}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    window = DemoWindow()
    window.show()
    sys.exit(app.exec())