import matplotlib
matplotlib.rcParams['font.family'] = ['Arial Unicode MS', 'sans-serif']

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QProgressBar, QListWidget,
    QTextEdit, QButtonGroup, QRadioButton, QStackedWidget, QFrame,
    QSizePolicy, QMessageBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt6.QtGui import QColor
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import qtawesome as qta
import numpy as np
import requests
import sys
import os
import subprocess
import shutil
import time

C = {
    'bg_base':     '#0a0c0f',
    'bg_sidebar':  '#0d1117',
    'bg_mid':      '#161b22',
    'bg_surface':  '#1c2128',
    'bg_input':    '#21262d',
    'border':      '#30363d',
    'border_hi':   '#444c56',
    'text_pri':    '#e6edf3',
    'text_sec':    '#8b949e',
    'text_dis':    '#484f58',
    'accent':      '#388bfd',
    'accent_dim':  '#1f6feb',
    'success':     '#3fb950',
    'success_bg':  '#3fb95015',
    'danger':      '#f85149',
    'danger_bg':   '#f8514915',
}

MODELS = {
    "RNNoise":       "http://localhost:8001",
    "DeepFilterNet": "http://localhost:8002",
    "Demucs":        "http://localhost:8003",
}

SUPPORTED_FORMATS = ('.wav', '.mp3', '.m4a', '.flac', '.ogg', '.aac')


def check_ffmpeg():
    return shutil.which('ffmpeg') is not None

CONVERTER_AVAILABLE = check_ffmpeg()


def compute_spectrogram(filepath):
    try:
        import librosa
        y, sr = librosa.load(filepath, sr=None, mono=True)
        D = librosa.amplitude_to_db(
            np.abs(librosa.stft(y, n_fft=1024, hop_length=256)), ref=np.max)
        return D, sr
    except Exception:
        return None, None


class AudioPlayer:
    def play(self, filepath):
        self.stop()
        try:
            import sounddevice as sd
            import soundfile as sf
            if not filepath.lower().endswith('.wav'):
                tmp = filepath.rsplit('.', 1)[0] + '_preview.wav'
                subprocess.run(['ffmpeg', '-i', filepath,
                                '-acodec', 'pcm_s16le', '-ar', '48000', '-ac', '1',
                                '-y', tmp], capture_output=True)
                data, sr = sf.read(tmp)
                os.remove(tmp)
            else:
                data, sr = sf.read(filepath)
            sd.play(data, sr)
        except Exception as e:
            print(f"播放失敗：{e}")

    def stop(self):
        try:
            import sounddevice as sd
            sd.stop()
        except Exception:
            pass


class DenoiseWorker(QThread):
    finished = pyqtSignal(str)
    error    = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, input_file, api_url):
        super().__init__()
        self.input_file = input_file
        self.api_url    = api_url

    def run(self):
        temp_wav = None
        try:
            process_file = self.input_file
            if not self.input_file.lower().endswith('.wav'):
                if not CONVERTER_AVAILABLE:
                    self.error.emit("缺少 ffmpeg"); return
                self.progress.emit("converting...")
                temp_wav = self.input_file.rsplit('.', 1)[0] + '_temp.wav'
                r = subprocess.run([
                    'ffmpeg', '-i', self.input_file,
                    '-acodec', 'pcm_s16le', '-ar', '48000', '-ac', '1', '-y', temp_wav
                ], capture_output=True, text=True, timeout=30)
                if r.returncode != 0:
                    self.error.emit("ffmpeg failed"); return
                process_file = temp_wav
                self.progress.emit("denoising...")

            with open(process_file, 'rb') as f:
                resp = requests.post(f"{self.api_url}/denoise",
                                     files={"file": f}, timeout=120)
            if resp.status_code == 200:
                output = self.input_file.rsplit('.', 1)[0] + '_降噪.wav'
                with open(output, 'wb') as f:
                    f.write(resp.content)
                self.finished.emit(output)
            else:
                self.error.emit(f"API error {resp.status_code}")
        except requests.exceptions.ConnectionError:
            self.error.emit("cannot connect to container")
        except Exception as e:
            self.error.emit(str(e))
        finally:
            if temp_wav and os.path.exists(temp_wav):
                try: os.remove(temp_wav)
                except: pass


class BatchWorker(QThread):
    file_done = pyqtSignal(int, str, bool, str)
    all_done  = pyqtSignal(int, int)

    def __init__(self, files, api_url, output_dir):
        super().__init__()
        self.files      = files
        self.api_url    = api_url
        self.output_dir = output_dir
        self._stop      = False

    def stop(self): self._stop = True

    def run(self):
        success = fail = 0
        for i, filepath in enumerate(self.files):
            if self._stop: break
            temp_wav = None
            try:
                process_file = filepath
                if not filepath.lower().endswith('.wav'):
                    temp_wav = filepath.rsplit('.', 1)[0] + '_batch_temp.wav'
                    r = subprocess.run([
                        'ffmpeg', '-i', filepath,
                        '-acodec', 'pcm_s16le', '-ar', '48000', '-ac', '1', '-y', temp_wav
                    ], capture_output=True, timeout=60)
                    if r.returncode != 0: raise Exception("ffmpeg failed")
                    process_file = temp_wav

                with open(process_file, 'rb') as f:
                    resp = requests.post(f"{self.api_url}/denoise",
                                         files={"file": f}, timeout=180)
                if resp.status_code == 200:
                    fname = os.path.splitext(os.path.basename(filepath))[0] + '_降噪.wav'
                    with open(os.path.join(self.output_dir, fname), 'wb') as f:
                        f.write(resp.content)
                    success += 1
                    self.file_done.emit(i, os.path.basename(filepath), True, "")
                else:
                    raise Exception(f"API error {resp.status_code}")
            except Exception as e:
                fail += 1
                self.file_done.emit(i, os.path.basename(filepath), False, str(e))
            finally:
                if temp_wav and os.path.exists(temp_wav):
                    try: os.remove(temp_wav)
                    except: pass
            time.sleep(1)
        self.all_done.emit(success, fail)


def mono_label(text, size=10, color=None):
    lbl = QLabel(text.upper())
    lbl.setStyleSheet(f"""
        font-family: 'Menlo', 'Courier New', monospace;
        font-size: {size}px;
        font-weight: 600;
        color: {color or C['text_dis']};
        letter-spacing: 0.8px;
    """)
    return lbl


def divider():
    line = QFrame()
    line.setFrameShape(QFrame.Shape.HLine)
    line.setFixedHeight(1)
    line.setStyleSheet(f"background: {C['border']}; border: none;")
    return line


def primary_btn(text, icon_name=None):
    btn = QPushButton(f"  {text}" if icon_name else text)
    if icon_name:
        btn.setIcon(qta.icon(icon_name, color='#ffffff'))
        btn.setIconSize(QSize(13, 13))
    btn.setFixedHeight(32)
    btn.setStyleSheet(f"""
        QPushButton {{
            background: {C['accent']};
            color: #ffffff;
            border: none;
            border-radius: 6px;
            padding: 0 16px;
            font-size: 12px;
            font-weight: 600;
        }}
        QPushButton:hover {{ background: {C['accent_dim']}; }}
        QPushButton:pressed {{ background: #1158c7; }}
        QPushButton:disabled {{
            background: {C['bg_surface']};
            color: {C['text_dis']};
            border: 1px solid {C['border']};
        }}
    """)
    return btn


def ghost_btn(text, icon_name=None, icon_color=None):
    btn = QPushButton(f"  {text}" if icon_name else text)
    if icon_name:
        btn.setIcon(qta.icon(icon_name, color=icon_color or C['text_sec']))
        btn.setIconSize(QSize(13, 13))
    btn.setFixedHeight(32)
    btn.setStyleSheet(f"""
        QPushButton {{
            background: {C['bg_surface']};
            color: {C['text_sec']};
            border: 1px solid {C['border']};
            border-radius: 6px;
            padding: 0 14px;
            font-size: 12px;
        }}
        QPushButton:hover {{
            background: {C['bg_input']};
            border-color: {C['border_hi']};
            color: {C['text_pri']};
        }}
        QPushButton:pressed {{ background: {C['bg_mid']}; }}
    """)
    return btn


def danger_btn(text, icon_name=None):
    btn = QPushButton(f"  {text}" if icon_name else text)
    if icon_name:
        btn.setIcon(qta.icon(icon_name, color=C['danger']))
        btn.setIconSize(QSize(13, 13))
    btn.setFixedHeight(32)
    btn.setStyleSheet(f"""
        QPushButton {{
            background: {C['danger_bg']};
            color: {C['danger']};
            border: 1px solid {C['danger']}40;
            border-radius: 6px;
            padding: 0 14px;
            font-size: 12px;
            font-weight: 600;
        }}
        QPushButton:hover {{
            background: {C['danger']}30;
            border-color: {C['danger']};
        }}
        QPushButton:disabled {{
            background: {C['bg_surface']};
            color: {C['text_dis']};
            border-color: {C['border']};
        }}
    """)
    return btn


class SidebarBtn(QPushButton):
    def __init__(self, icon_name, tooltip):
        super().__init__()
        self.setFixedSize(48, 44)
        self.setToolTip(tooltip)
        self.setCheckable(True)
        self._icon = icon_name
        self._apply_style(False)
        self.toggled.connect(self._apply_style)

    def _apply_style(self, checked):
        color = C['accent'] if checked else C['text_dis']
        self.setIcon(qta.icon(self._icon, color=color))
        self.setIconSize(QSize(18, 18))
        bg = C['bg_surface'] if checked else 'transparent'
        left_border = f"border-left: 2px solid {C['accent']};" if checked else "border-left: 2px solid transparent;"
        self.setStyleSheet(f"""
            QPushButton {{
                background-color: {bg};
                border: none;
                {left_border}
                border-radius: 0px;
                outline: none;
            }}
            QPushButton:hover {{ background-color: {C['bg_surface']}; }}
            QPushButton:checked {{ background-color: {C['bg_surface']}; border-left: 2px solid {C['accent']}; }}
        """)


class Sidebar(QWidget):
    page_changed = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self.setFixedWidth(48)
        self.setStyleSheet(f"background-color: {C['bg_sidebar']}; border-right: 1px solid {C['border']};")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 8, 0, 8)
        layout.setSpacing(2)

        self.btns = [
            SidebarBtn('fa5s.file-audio', '單檔處理'),
            SidebarBtn('fa5s.layer-group', '批次處理'),
        ]

        self.group = QButtonGroup(self)
        self.group.setExclusive(True)
        for i, btn in enumerate(self.btns):
            self.group.addButton(btn, i)
            layout.addWidget(btn)

        layout.addStretch()
        self.group.idClicked.connect(self.page_changed)
        self.btns[0].setChecked(True)


class StatusBar(QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedHeight(28)
        self.setStyleSheet(f"background-color: {C['bg_sidebar']}; border-top: 1px solid {C['border']};")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 0, 16, 0)
        layout.setSpacing(0)

        self._badges = {}
        for name in MODELS:
            badge = self._make_badge(name)
            self._badges[name] = badge
            layout.addWidget(badge)
            layout.addSpacing(6)

        layout.addStretch()

    def _make_badge(self, name):
        w = QWidget()
        w.setFixedHeight(20)
        row = QHBoxLayout(w)
        row.setContentsMargins(8, 0, 8, 0)
        row.setSpacing(5)

        dot = QLabel("●")
        dot.setStyleSheet(f"font-size: 7px; color: {C['text_dis']};")
        lbl = QLabel(name)
        lbl.setStyleSheet(f"font-family: 'Menlo', 'Courier New', monospace; font-size: 10px; color: {C['text_dis']};")

        row.addWidget(dot)
        row.addWidget(lbl)

        w.setStyleSheet(f"background-color: {C['bg_surface']}; border: 1px solid {C['border']}; border-radius: 4px;")
        w._dot = dot
        w._lbl = lbl
        return w

    def set_status(self, name, online):
        badge = self._badges[name]
        if online:
            badge._dot.setStyleSheet(f"font-size: 7px; color: {C['success']};")
            badge._lbl.setStyleSheet(f"font-family: 'Menlo', 'Courier New', monospace; font-size: 10px; color: {C['success']};")
            badge.setStyleSheet(f"background-color: {C['success_bg']}; border: 1px solid {C['success']}40; border-radius: 4px;")
        else:
            badge._dot.setStyleSheet(f"font-size: 7px; color: {C['text_dis']};")
            badge._lbl.setStyleSheet(f"font-family: 'Menlo', 'Courier New', monospace; font-size: 10px; color: {C['text_dis']};")
            badge.setStyleSheet(f"background-color: {C['bg_surface']}; border: 1px solid {C['border']}; border-radius: 4px;")


class SpectrogramView(QWidget):
    back_clicked = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.player = AudioPlayer()
        self._in = self._out = None
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(14)

        top = QHBoxLayout()
        self.btn_back = ghost_btn("Back", "fa5s.arrow-left")
        self.btn_back.clicked.connect(self.back_clicked)
        top.addWidget(self.btn_back)

        self.title = QLabel("Spectrogram")
        self.title.setStyleSheet(f"font-size: 13px; font-weight: 600; color: {C['text_sec']}; padding-left: 10px;")
        top.addWidget(self.title)
        top.addStretch()
        layout.addLayout(top)

        self.canvas_area = QWidget()
        self.canvas_area.setStyleSheet(f"background-color: {C['bg_mid']}; border: 1px solid {C['border']}; border-radius: 8px;")
        self._canvas_layout = QVBoxLayout(self.canvas_area)
        self._canvas_layout.setContentsMargins(8, 8, 8, 8)
        layout.addWidget(self.canvas_area, stretch=1)

        ab = QHBoxLayout()
        ab.setSpacing(8)

        self.btn_before = ghost_btn("Original", "fa5s.play")
        self.btn_before.clicked.connect(lambda: self.player.play(self._in))
        ab.addWidget(self.btn_before)

        self.btn_after = primary_btn("Denoised", "fa5s.play")
        self.btn_after.clicked.connect(lambda: self.player.play(self._out))
        ab.addWidget(self.btn_after)

        self.btn_stop = ghost_btn("Stop", "fa5s.stop")
        self.btn_stop.clicked.connect(self.player.stop)
        ab.addWidget(self.btn_stop)

        ab.addStretch()
        layout.addLayout(ab)

    def load(self, input_file, output_file, model_name):
        self._in, self._out = input_file, output_file
        self.title.setText(f"Spectrogram  ·  {model_name}")

        for i in reversed(range(self._canvas_layout.count())):
            w = self._canvas_layout.itemAt(i).widget()
            if w: w.deleteLater()

        D_in,  sr_in  = compute_spectrogram(input_file)
        D_out, sr_out = compute_spectrogram(output_file)

        fig = Figure(facecolor=C['bg_mid'])
        canvas = FigureCanvas(fig)
        canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._canvas_layout.addWidget(canvas)

        ax1 = fig.add_subplot(2, 1, 1)
        ax2 = fig.add_subplot(2, 1, 2)

        for ax, D, sr, label, color in [
            (ax1, D_in,  sr_in,  "before", C['text_sec']),
            (ax2, D_out, sr_out, f"after  ·  {model_name}", C['accent']),
        ]:
            ax.set_facecolor(C['bg_mid'])
            if D is not None:
                import librosa
                librosa.display.specshow(D, sr=sr, hop_length=256,
                                         x_axis='time', y_axis='hz',
                                         ax=ax, cmap='magma')
            ax.set_title(label, color=color, fontsize=9, pad=6)
            ax.tick_params(colors=C['text_dis'], labelsize=8)
            ax.xaxis.label.set_color(C['text_dis'])
            ax.yaxis.label.set_color(C['text_dis'])
            for spine in ax.spines.values():
                spine.set_edgecolor(C['border'])

        fig.subplots_adjust(hspace=0.65, top=0.92, bottom=0.08)
        canvas.draw()


class SinglePage(QWidget):
    show_spectrogram = pyqtSignal(str, str, str)

    def __init__(self, model_buttons):
        super().__init__()
        self.model_buttons = model_buttons
        self.input_file  = None
        self.worker      = None
        self._output     = None
        self._model_name = None
        self.player      = AudioPlayer()
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(0)

        layout.addWidget(mono_label("input"))
        layout.addSpacing(8)

        file_row = QHBoxLayout()
        file_row.setSpacing(0)

        self.file_lbl = QLabel("No file selected")
        self.file_lbl.setStyleSheet(f"""
            background-color: {C['bg_input']};
            border: 1px solid {C['border']};
            border-right: none;
            border-radius: 6px 0 0 6px;
            color: {C['text_dis']};
            font-size: 12px;
            padding: 0 12px;
            min-height: 32px;
        """)
        file_row.addWidget(self.file_lbl, stretch=1)

        self.btn_select = QPushButton()
        self.btn_select.setIcon(qta.icon('fa5s.folder-open', color=C['text_sec']))
        self.btn_select.setIconSize(QSize(13, 13))
        self.btn_select.setFixedSize(80, 32)
        self.btn_select.setText("  Browse")
        self.btn_select.setStyleSheet(f"""
            QPushButton {{
                background-color: {C['bg_surface']};
                color: {C['text_sec']};
                border: 1px solid {C['border']};
                border-radius: 0 6px 6px 0;
                font-size: 12px;
            }}
            QPushButton:hover {{
                background-color: {C['bg_input']};
                color: {C['text_pri']};
                border-color: {C['border_hi']};
            }}
        """)
        self.btn_select.clicked.connect(self.select_file)
        file_row.addWidget(self.btn_select)

        layout.addLayout(file_row)
        layout.addSpacing(20)
        layout.addWidget(divider())
        layout.addSpacing(16)

        layout.addWidget(mono_label("model"))
        layout.addSpacing(10)

        model_row = QHBoxLayout()
        model_row.setSpacing(8)
        for name, btn in self.model_buttons.items():
            model_row.addWidget(btn)
        model_row.addStretch()
        layout.addLayout(model_row)
        layout.addSpacing(20)
        layout.addWidget(divider())
        layout.addSpacing(16)

        run_row = QHBoxLayout()
        self.btn_run = primary_btn("Denoise")
        self.btn_run.setFixedWidth(110)
        self.btn_run.setEnabled(False)
        self.btn_run.clicked.connect(self.start_denoise)
        run_row.addWidget(self.btn_run)

        self.status_lbl = QLabel("")
        self.status_lbl.setStyleSheet(f"font-family: 'Menlo', 'Courier New', monospace; font-size: 10px; color: {C['text_sec']}; padding-left: 10px;")
        run_row.addWidget(self.status_lbl)
        run_row.addStretch()
        layout.addLayout(run_row)
        layout.addSpacing(6)

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        self.progress.setFixedHeight(2)
        self.progress.setTextVisible(False)
        self.progress.setStyleSheet(f"""
            QProgressBar {{ background-color: {C['bg_surface']}; border: none; }}
            QProgressBar::chunk {{ background-color: {C['accent']}; }}
        """)
        layout.addWidget(self.progress)
        layout.addSpacing(20)

        self.result_card = QWidget()
        self.result_card.setVisible(False)
        self.result_card.setStyleSheet(f"background-color: {C['bg_surface']}; border: 1px solid {C['border']}; border-radius: 8px;")
        rc = QVBoxLayout(self.result_card)
        rc.setContentsMargins(16, 12, 16, 12)
        rc.setSpacing(10)

        rc_btns = QHBoxLayout()
        rc_btns.setSpacing(8)

        self.btn_spectrum = ghost_btn("Spectrogram", "fa5s.chart-bar")
        self.btn_spectrum.clicked.connect(self._emit_spectrogram)
        rc_btns.addWidget(self.btn_spectrum)

        self.btn_play_a = ghost_btn("Original", "fa5s.play")
        self.btn_play_a.clicked.connect(lambda: self.player.play(self.input_file))
        rc_btns.addWidget(self.btn_play_a)

        self.btn_play_b = primary_btn("Denoised", "fa5s.play")
        self.btn_play_b.clicked.connect(lambda: self.player.play(self._output))
        rc_btns.addWidget(self.btn_play_b)

        self.btn_stop = ghost_btn("Stop", "fa5s.stop")
        self.btn_stop.clicked.connect(self.player.stop)
        rc_btns.addWidget(self.btn_stop)

        rc_btns.addStretch()
        rc.addLayout(rc_btns)
        layout.addWidget(self.result_card)
        layout.addStretch()

    def get_selected_model(self):
        for name, btn in self.model_buttons.items():
            if btn.isChecked():
                return name, MODELS[name]
        return None, None

    def select_file(self):
        file, _ = QFileDialog.getOpenFileName(
            self, "Select audio file",
            os.path.expanduser("~/Downloads"),
            "Audio (*.wav *.mp3 *.m4a *.flac *.ogg *.aac);;All (*.*)"
        )
        if file:
            self.input_file = file
            self.file_lbl.setText(os.path.basename(file))
            self.file_lbl.setStyleSheet(f"""
                background-color: {C['bg_input']};
                border: 1px solid {C['accent']};
                border-right: none;
                border-radius: 6px 0 0 6px;
                color: {C['text_pri']};
                font-size: 12px;
                padding: 0 12px;
                min-height: 32px;
            """)
            self.btn_run.setEnabled(True)
            self.result_card.setVisible(False)

    def start_denoise(self):
        model_name, api_url = self.get_selected_model()
        if not model_name: return
        self._model_name = model_name

        self.progress.setVisible(True)
        self.progress.setRange(0, 0)
        self.btn_run.setEnabled(False)
        self.result_card.setVisible(False)
        self.status_lbl.setText(f"running {model_name}...")

        self.worker = DenoiseWorker(self.input_file, api_url)
        self.worker.finished.connect(self.on_success)
        self.worker.error.connect(self.on_error)
        self.worker.progress.connect(lambda m: self.status_lbl.setText(m))
        self.worker.start()

    def on_success(self, output_file):
        self._output = output_file
        self.progress.setVisible(False)
        self.btn_run.setEnabled(True)
        self.status_lbl.setText("done")
        self.status_lbl.setStyleSheet(f"font-family: 'Menlo', 'Courier New', monospace; font-size: 10px; color: {C['success']}; padding-left: 10px;")
        self.result_card.setVisible(True)

    def on_error(self, msg):
        self.progress.setVisible(False)
        self.btn_run.setEnabled(True)
        self.status_lbl.setText(f"error: {msg}")
        self.status_lbl.setStyleSheet(f"font-family: 'Menlo', 'Courier New', monospace; font-size: 10px; color: {C['danger']}; padding-left: 10px;")

    def _emit_spectrogram(self):
        if self._output:
            self.show_spectrogram.emit(self.input_file, self._output, self._model_name)


class BatchPage(QWidget):
    def __init__(self, model_buttons):
        super().__init__()
        self.model_buttons = model_buttons
        self.files      = []
        self.output_dir = None
        self.worker     = None
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(0)

        layout.addWidget(mono_label("source"))
        layout.addSpacing(8)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)

        self.btn_folder = ghost_btn("Select folder", "fa5s.folder-open")
        self.btn_folder.clicked.connect(self.select_folder)
        btn_row.addWidget(self.btn_folder)

        self.btn_output = ghost_btn("Output folder", "fa5s.folder")
        self.btn_output.clicked.connect(self.select_output)
        btn_row.addWidget(self.btn_output)

        self.btn_clear = ghost_btn("Clear", "fa5s.times")
        self.btn_clear.clicked.connect(self.clear_files)
        btn_row.addWidget(self.btn_clear)

        btn_row.addStretch()
        layout.addLayout(btn_row)
        layout.addSpacing(8)

        self.info_lbl = QLabel("No files selected")
        self.info_lbl.setStyleSheet(f"font-family: 'Menlo', 'Courier New', monospace; font-size: 10px; color: {C['text_dis']};")
        layout.addWidget(self.info_lbl)
        layout.addSpacing(10)

        self.file_list = QListWidget()
        self.file_list.setStyleSheet(f"""
            QListWidget {{
                background-color: {C['bg_input']};
                border: 1px solid {C['border']};
                border-radius: 8px;
                color: {C['text_sec']};
                font-size: 11px;
                outline: none;
                padding: 4px;
            }}
            QListWidget::item {{ padding: 4px 8px; border-radius: 4px; }}
            QListWidget::item:selected {{ background-color: {C['bg_surface']}; color: {C['text_pri']}; }}
        """)
        layout.addWidget(self.file_list, stretch=1)
        layout.addSpacing(10)

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        self.progress.setFixedHeight(2)
        self.progress.setTextVisible(False)
        self.progress.setStyleSheet(f"""
            QProgressBar {{ background-color: {C['bg_surface']}; border: none; }}
            QProgressBar::chunk {{ background-color: {C['accent']}; }}
        """)
        layout.addWidget(self.progress)
        layout.addSpacing(8)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setFixedHeight(110)
        self.log.setPlaceholderText("// output log")
        self.log.setStyleSheet(f"""
            QTextEdit {{
                background-color: {C['bg_sidebar']};
                border: 1px solid {C['border']};
                border-radius: 8px;
                color: {C['text_sec']};
                font-family: 'Menlo', 'Courier New', monospace;
                font-size: 10px;
                padding: 8px;
            }}
        """)
        layout.addWidget(self.log)
        layout.addSpacing(12)

        action_row = QHBoxLayout()
        action_row.setSpacing(8)

        self.btn_start = primary_btn("Run batch", "fa5s.play")
        self.btn_start.setEnabled(False)
        self.btn_start.clicked.connect(self.start_batch)
        action_row.addWidget(self.btn_start)

        self.btn_stop = danger_btn("Stop", "fa5s.stop")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.stop_batch)
        action_row.addWidget(self.btn_stop)

        action_row.addStretch()
        layout.addLayout(action_row)

    def get_selected_model(self):
        for name, btn in self.model_buttons.items():
            if btn.isChecked():
                return name, MODELS[name]
        return None, None

    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select audio folder",
            os.path.expanduser("~/Downloads")
        )
        if folder:
            self.files = [
                os.path.join(folder, f)
                for f in sorted(os.listdir(folder))
                if f.lower().endswith(SUPPORTED_FORMATS) and '_降噪' not in f
            ]
            self.file_list.clear()
            for f in self.files:
                self.file_list.addItem(os.path.basename(f))
            if not self.output_dir:
                self.output_dir = os.path.join(folder, 'output')
            self._update_info()

    def select_output(self):
        folder = QFileDialog.getExistingDirectory(self, "Select output folder")
        if folder:
            self.output_dir = folder
            self._update_info()

    def clear_files(self):
        self.files = []
        self.file_list.clear()
        self._update_info()

    def _update_info(self):
        n   = len(self.files)
        out = self.output_dir or "not set"
        self.info_lbl.setText(f"{n} files  ·  output: {out}")
        self.btn_start.setEnabled(n > 0)

    def start_batch(self):
        model_name, api_url = self.get_selected_model()
        if not model_name:
            QMessageBox.warning(self, "Error", "Please select a model first")
            return

        os.makedirs(self.output_dir, exist_ok=True)
        self.log.clear()
        self.progress.setVisible(True)
        self.progress.setRange(0, len(self.files))
        self.progress.setValue(0)
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.log.append(f"// model: {model_name}  files: {len(self.files)}")

        self.worker = BatchWorker(self.files, api_url, self.output_dir)
        self.worker.file_done.connect(self.on_file_done)
        self.worker.all_done.connect(self.on_all_done)
        self.worker.start()

    def stop_batch(self):
        if self.worker:
            self.worker.stop()
            self.log.append("// stopped")
            self.btn_stop.setEnabled(False)

    def on_file_done(self, idx, filename, success, error_msg):
        self.progress.setValue(idx + 1)
        total = len(self.files)
        if success:
            self.log.append(f"[{idx+1}/{total}]  ok   {filename}")
            item = self.file_list.item(idx)
            if item: item.setForeground(QColor(C['success']))
        else:
            self.log.append(f"[{idx+1}/{total}]  err  {filename}  —  {error_msg}")
            item = self.file_list.item(idx)
            if item: item.setForeground(QColor(C['danger']))

    def on_all_done(self, success, fail):
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.log.append(f"// done  ok:{success}  err:{fail}  →  {self.output_dir}")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Speech Denoiser")
        self.setMinimumSize(680, 500)
        self.resize(820, 560)
        self.setStyleSheet(f"background-color: {C['bg_base']}; color: {C['text_pri']};")

        root = QWidget()
        self.setCentralWidget(root)
        root_l = QVBoxLayout(root)
        root_l.setContentsMargins(0, 0, 0, 0)
        root_l.setSpacing(0)

        title_bar = QWidget()
        title_bar.setFixedHeight(38)
        title_bar.setStyleSheet(f"background-color: {C['bg_sidebar']}; border-bottom: 1px solid {C['border']};")
        tb_l = QHBoxLayout(title_bar)
        tb_l.setContentsMargins(14, 0, 14, 0)
        title = QLabel("SPEECH DENOISER")
        title.setStyleSheet(f"font-family: 'SF Pro Text', '-apple-system', sans-serif; font-size: 11px; font-weight: 700; color: {C['text_sec']}; letter-spacing: 2px;")
        tb_l.addWidget(title)
        tb_l.addStretch()
        root_l.addWidget(title_bar)

        body = QWidget()
        body_l = QHBoxLayout(body)
        body_l.setContentsMargins(0, 0, 0, 0)
        body_l.setSpacing(0)

        self.model_buttons = {}
        self.btn_group = QButtonGroup()
        for i, name in enumerate(MODELS):
            btn = QRadioButton(name)
            btn.setStyleSheet(f"""
                QRadioButton {{ font-size: 12px; color: {C['text_sec']}; spacing: 7px; }}
                QRadioButton::indicator {{ width: 14px; height: 14px; border: 1.5px solid {C['border_hi']}; border-radius: 7px; background-color: {C['bg_input']}; }}
                QRadioButton::indicator:checked {{ background-color: {C['accent']}; border-color: {C['accent']}; }}
                QRadioButton:checked {{ color: {C['text_pri']}; }}
                QRadioButton:disabled {{ color: {C['text_dis']}; }}
                QRadioButton::indicator:disabled {{ border-color: {C['border']}; background-color: {C['bg_base']}; }}
            """)
            self.model_buttons[name] = btn
            self.btn_group.addButton(btn, i)
        self.model_buttons["RNNoise"].setChecked(True)

        self.sidebar = Sidebar()
        self.sidebar.page_changed.connect(self._on_page)
        body_l.addWidget(self.sidebar)

        self.stack = QStackedWidget()
        self.stack.setStyleSheet(f"background-color: {C['bg_base']};")

        self.single = SinglePage(self.model_buttons)
        self.single.show_spectrogram.connect(self._show_spec)
        self.stack.addWidget(self.single)

        self.batch = BatchPage(self.model_buttons)
        self.stack.addWidget(self.batch)

        self.spec_view = SpectrogramView()
        self.spec_view.back_clicked.connect(self._hide_spec)
        self.stack.addWidget(self.spec_view)

        body_l.addWidget(self.stack, stretch=1)
        root_l.addWidget(body, stretch=1)

        self.status_bar = StatusBar()
        root_l.addWidget(self.status_bar)

        self._check_containers()

    def _on_page(self, idx):
        self.stack.setCurrentIndex(idx)

    def _show_spec(self, i, o, m):
        self.spec_view.load(i, o, m)
        self.stack.setCurrentIndex(2)

    def _hide_spec(self):
        self.stack.setCurrentIndex(0)
        self.sidebar.btns[0].setChecked(True)

    def _check_containers(self):
        available = []
        for name, url in MODELS.items():
            try:
                r = requests.get(f"{url}/health", timeout=2)
                online = r.status_code == 200
            except:
                online = False
            self.status_bar.set_status(name, online)
            self.model_buttons[name].setEnabled(online)
            if online:
                available.append(name)

        if available:
            self.btn_group.button(list(MODELS.keys()).index(available[0])).setChecked(True)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = MainWindow()
    window.show()
    sys.exit(app.exec())