import os
import sys
import threading
import traceback
import tempfile
import time
import subprocess
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import srt
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# Whisper (lokalny)
try:
    import whisper
    from whisper.transcribe import transcribe
except ImportError:
    whisper = None

# Dźwięk
import soundfile as sf
import librosa

# =========================
# KONFIGURACJA
# =========================
@dataclass
class Config:
    whisper_model: str = "base"
    vad_snap_threshold: float = 1.2
    vad_blend: float = 0.7
    fit_points: int = 9
    audio_sr: int = 16000
    language: str = None


# =========================
# POMOCNICZE
# =========================
def to_seconds(td) -> float:
    return td.total_seconds()

def from_seconds(sec: float):
    sec = max(0.0, float(sec))
    return srt.timedelta(seconds=sec)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

DEBUG_LOG = os.path.join(os.path.dirname(__file__), "debug_log.txt")

def debug(msg: str):
    ts = time.strftime("[%H:%M:%S]")
    with open(DEBUG_LOG, "a", encoding="utf-8") as f:
        f.write(f"{ts} {msg}\n")


# =========================
# DOPASOWANIE NAPISÓW
# =========================
def linear_fit_map(sub_times: List[float], speech_times: List[float], fit_points: int = 9) -> Tuple[float, float]:
    if not sub_times or not speech_times:
        return 1.0, 0.0

    fit_points = max(2, min(fit_points, min(len(sub_times), len(speech_times))))
    xs, ys = [], []
    for j in range(fit_points):
        idx_sub = int(round(j * (len(sub_times) - 1) / (fit_points - 1)))
        idx_spk = int(round(j * (len(speech_times) - 1) / (fit_points - 1)))
        xs.append(sub_times[idx_sub])
        ys.append(speech_times[idx_spk])

    X = np.vstack([np.array(xs), np.ones(len(xs))]).T
    y = np.array(ys)
    a, b = np.linalg.lstsq(X, y, rcond=None)[0]
    return float(a), float(b)


def nearest_idx(sorted_list: List[float], value: float) -> int:
    if not sorted_list:
        return -1
    arr = np.array(sorted_list)
    idx = int(np.argmin(np.abs(arr - value)))
    return idx


def adjust_with_vad(mapped_time: float, speech_starts: List[float], threshold: float, blend: float) -> float:
    idx = nearest_idx(speech_starts, mapped_time)
    if idx < 0:
        return mapped_time
    nearest = speech_starts[idx]
    diff = abs(nearest - mapped_time)
    if diff <= threshold:
        return mapped_time
    return blend * mapped_time + (1.0 - blend) * nearest


def map_subtitles_linear_with_snap(subs: List[srt.Subtitle],
                                   speech_segments: List[Tuple[float, float]],
                                   cfg: Config) -> List[srt.Subtitle]:
    sub_starts = [to_seconds(s.start) for s in subs]
    speech_starts = [seg[0] for seg in speech_segments]

    a, b = linear_fit_map(sub_starts, speech_starts, cfg.fit_points)
    debug(f"Linear fit: a={a:.5f}, b={b:.3f}")

    new_subs = []
    for s in subs:
        ns = srt.Subtitle(
            index=s.index,
            start=from_seconds(a * to_seconds(s.start) + b),
            end=from_seconds(a * to_seconds(s.end) + b),
            content=s.content,
            proprietary=s.proprietary
        )
        new_subs.append(ns)

    if speech_starts:
        for i, s in enumerate(new_subs):
            mapped_start = to_seconds(s.start)
            snapped_start = adjust_with_vad(mapped_start, speech_starts, cfg.vad_snap_threshold, cfg.vad_blend)
            shift = snapped_start - mapped_start
            if abs(shift) > 0.01:
                debug(f"Subtitle {i} shifted by {shift:.2f}s")
            duration = to_seconds(s.end) - to_seconds(s.start)
            s.start = from_seconds(snapped_start)
            s.end = from_seconds(snapped_start + max(0.2, duration))
            new_subs[i] = s

    for i in range(1, len(new_subs)):
        prev = new_subs[i - 1]
        cur = new_subs[i]
        if to_seconds(cur.start) < to_seconds(prev.end) - 0.05:
            cur_start = to_seconds(prev.end) + 0.02
            duration = max(0.2, to_seconds(cur.end) - to_seconds(cur.start))
            cur.start = from_seconds(cur_start)
            cur.end = from_seconds(cur_start + duration)
            new_subs[i] = cur

    return new_subs


# =========================
# FFmpeg + Whisper
# =========================
def get_ffmpeg_path() -> str:
    folder = os.path.dirname(__file__)
    exe_path = os.path.join(folder, "ffmpeg.exe")
    if not os.path.isfile(exe_path):
        raise FileNotFoundError("Nie znaleziono ffmpeg.exe w katalogu projektu!")
    return exe_path


def extract_audio(input_video: str, out_wav: str, sr: int = 16000):
    """Ekstrakcja audio z wideo do WAV (kompatybilna z Windows)."""
    ffmpeg_path = get_ffmpeg_path()
    debug(f"FFmpeg binary: {ffmpeg_path}")

    cmd = [
        ffmpeg_path,
        "-y",
        "-i", input_video,
        "-vn",
        "-ac", "1",
        "-ar", str(sr),
        "-map", "0:a:0",
        out_wav
    ]
    debug(f"Executing FFmpeg command: {' '.join(cmd)}")

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True
    )
    stdout, stderr = process.communicate()
    debug("FFmpeg STDOUT:\n" + (stdout or "<empty>"))
    debug("FFmpeg STDERR:\n" + (stderr or "<empty>"))

    if process.returncode != 0:
        raise RuntimeError(f"FFmpeg zwrócił błąd (kod {process.returncode}).\n{stderr.strip()}")
    if not os.path.isfile(out_wav) or os.path.getsize(out_wav) < 1000:
        raise RuntimeError("Nie udało się utworzyć pliku WAV – brak dźwięku lub nieobsługiwany kodek.")
    debug(f"Audio extraction completed successfully: {out_wav}")


def whisper_speech_segments(audio_wav: str, cfg: Config, progress_cb=None, log_cb=None) -> List[Tuple[float, float]]:
    if whisper is None:
        raise RuntimeError("Nie znaleziono biblioteki 'whisper'. Uruchom: pip install openai-whisper")

    model = whisper.load_model(cfg.whisper_model)
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    debug(f"Whisper model loaded: {cfg.whisper_model}")
    if log_cb: log_cb(f"🧠 Whisper działa na: {device}")

    result_segments = []

    import math
    y, sr = librosa.load(audio_wav, sr=cfg.audio_sr)
    total_frames = len(y)
    chunk = sr * 30  # 30 sekund
    chunks = math.ceil(total_frames / chunk)
    debug(f"Audio frames: {total_frames}, chunks: {chunks}")

    for i in range(chunks):
        start = i * chunk
        end = min((i + 1) * chunk, total_frames)
        piece = y[start:end]
        temp_path = os.path.join(tempfile.gettempdir(), f"whisper_chunk_{i}.wav")
        sf.write(temp_path, piece, sr)
        res = model.transcribe(temp_path, language=cfg.language, task="transcribe", verbose=False)
        os.remove(temp_path)
        segs = [(float(s["start"]) + i * 30, float(s["end"]) + i * 30)
                for s in res.get("segments", [])]
        result_segments.extend(segs)
        if progress_cb:
            progress_cb((i + 1) / chunks)
        if log_cb:
            log_cb(f"🧩 Przetworzono segment {i+1}/{chunks}")

    debug(f"Detected {len(result_segments)} speech segments total")
    return result_segments


# =========================
# GŁÓWNY PROCES
# =========================
def load_srt(path: str) -> List[srt.Subtitle]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return list(srt.parse(f.read()))

def save_srt(path: str, subs: List[srt.Subtitle]):
    for i, sub in enumerate(subs, start=1):
        sub.index = i
    with open(path, "w", encoding="utf-8") as f:
        f.write(srt.compose(subs))

def auto_sync(video_path: str, srt_path: str, cfg: Config, progress_cb=None, log_cb=None) -> str:
    def log(msg):
        debug(msg)
        if log_cb:
            log_cb(msg)

    def progress(p):
        if progress_cb:
            progress_cb(p)

    start_time = time.time()
    base_dir = os.path.dirname(__file__)
    temp_dir = os.path.join(base_dir, "temp")
    out_dir = os.path.join(base_dir, "output")
    ensure_dir(temp_dir)
    ensure_dir(out_dir)

    log("🔧 Przygotowanie...")
    progress(0.05)

    wav_path = os.path.join(temp_dir, "audio.wav")
    log("🎵 Ekstrakcja audio z wideo (FFmpeg)...")
    extract_audio(video_path, wav_path, sr=cfg.audio_sr)
    progress(0.2)

    log(f"🧠 Analiza mowy (Whisper model: {cfg.whisper_model})...")
    speech = whisper_speech_segments(wav_path, cfg, progress_cb=lambda x: progress(0.2 + x * 0.4), log_cb=log)
    progress(0.6)
    log(f"✅ Wykryto {len(speech)} fragmentów mowy.")

    log("📜 Wczytywanie napisów...")
    subs = load_srt(srt_path)
    log(f"✅ Załadowano {len(subs)} linii napisów.")
    progress(0.7)

    log("🧩 Dopasowywanie napisów do mowy...")
    new_subs = map_subtitles_linear_with_snap(subs, speech, cfg)
    progress(0.9)

    out_file = os.path.join(out_dir, os.path.basename(srt_path).replace(".srt", "_auto_sync.srt"))
    save_srt(out_file, new_subs)
    progress(1.0)
    elapsed = time.time() - start_time
    log(f"💾 Zapisano: {out_file}")
    log(f"🏁 Gotowe w {elapsed:.1f} s.")
    return out_file


# =========================
# GUI
# =========================
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("AutoSubtitleSync")
        self.geometry("700x500")
        self.video_path = tk.StringVar()
        self.srt_path = tk.StringVar()
        self.model_var = tk.StringVar(value="base")
        self.lang_var = tk.StringVar(value="auto (detekcja)")
        self.running = False
        self.create_widgets()

    def create_widgets(self):
        pad = {"padx": 10, "pady": 6}
        frm = ttk.Frame(self)
        frm.pack(fill="both", expand=True)

        ttk.Label(frm, text="Plik wideo:").grid(row=0, column=0, sticky="w", **pad)
        ttk.Entry(frm, textvariable=self.video_path).grid(row=0, column=1, sticky="we", **pad)
        ttk.Button(frm, text="Wybierz…", command=self.choose_video).grid(row=0, column=2, **pad)

        ttk.Label(frm, text="Plik napisów (.srt):").grid(row=1, column=0, sticky="w", **pad)
        ttk.Entry(frm, textvariable=self.srt_path).grid(row=1, column=1, sticky="we", **pad)
        ttk.Button(frm, text="Wybierz…", command=self.choose_srt).grid(row=1, column=2, **pad)

        ttk.Label(frm, text="Model Whisper:").grid(row=2, column=0, sticky="w", **pad)
        ttk.Combobox(frm, textvariable=self.model_var, values=["tiny","base","small","medium","large"], state="readonly").grid(row=2, column=1, sticky="we", **pad)

        ttk.Label(frm, text="Język mowy:").grid(row=3, column=0, sticky="w", **pad)
        ttk.Combobox(frm, textvariable=self.lang_var, values=["auto (detekcja)","pl","en","de","es","fr","it","ru"], state="readonly").grid(row=3, column=1, sticky="we", **pad)

        self.btn_run = ttk.Button(frm, text="Synchronizuj napisy automatycznie", command=self.on_run)
        self.btn_run.grid(row=4, column=0, columnspan=3, sticky="we", **pad)

        self.progress = ttk.Progressbar(frm, mode="determinate")
        self.progress.grid(row=5, column=0, columnspan=3, sticky="we", **pad)

        ttk.Label(frm, text="Log:").grid(row=6, column=0, sticky="w", **pad)
        self.txt_log = tk.Text(frm, height=14, wrap="word")
        self.txt_log.grid(row=7, column=0, columnspan=3, sticky="nsew", padx=10, pady=(0,10))
        frm.columnconfigure(1, weight=1)
        frm.rowconfigure(7, weight=1)

    def choose_video(self):
        p = filedialog.askopenfilename(filetypes=[("Wideo", "*.mp4;*.mkv;*.avi;*.mov;*.m4v")])
        if p: self.video_path.set(p)

    def choose_srt(self):
        p = filedialog.askopenfilename(filetypes=[("Napisy SRT", "*.srt")])
        if p: self.srt_path.set(p)

    def log(self, msg):
        self.txt_log.insert("end", msg + "\n")
        self.txt_log.see("end")
        self.update_idletasks()
        debug(f"[GUI] {msg}")

    def set_progress(self, p):
        self.progress["value"] = max(0,min(100,p*100))
        self.update_idletasks()

    def on_run(self):
        if self.running: return
        video, srtp = self.video_path.get().strip(), self.srt_path.get().strip()
        if not os.path.isfile(video): return messagebox.showerror("Błąd","Wybierz poprawny plik wideo.")
        if not os.path.isfile(srtp): return messagebox.showerror("Błąd","Wybierz poprawny plik .srt")

        model = self.model_var.get()
        lang = None if "auto" in self.lang_var.get() else self.lang_var.get()
        cfg = Config(whisper_model=model, language=lang)
        self.running=True
        self.btn_run.config(state="disabled")
        self.set_progress(0)
        self.log("🚀 Start synchronizacji...")

        def worker():
            try:
                out = auto_sync(video, srtp, cfg, progress_cb=self.set_progress, log_cb=self.log)
                self.log("✅ Synchronizacja zakończona.")
                messagebox.showinfo("Gotowe", f"Napisy zapisane jako:\n{out}")
            except Exception as e:
                err = traceback.format_exc()
                debug(err)
                self.log(f"❌ Błąd: {e}")
                messagebox.showerror("Błąd", str(e))
            finally:
                self.running=False
                self.btn_run.config(state="normal")

        threading.Thread(target=worker,daemon=True).start()


if __name__=="__main__":
    debug("\n=== AutoSubtitleSync uruchomiono ===")
    app=App()
    app.mainloop()
