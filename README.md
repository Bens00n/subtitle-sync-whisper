# AutoSubtitleSync

Lokalny GUI do automatycznej synchronizacji napisow SRT z wideo przy pomocy Whispera i VAD.

## Co robi
- wycina audio z pliku wideo przy uzyciu wbudowanego `ffmpeg.exe`
- wykrywa fragmenty mowy modelem Whisper (CPU lub GPU)
- dopasowuje timingi napisow liniowo z dodatkowymi przeskokami do startow mowy
- zapisuje nowe napisy w folderze `output` jako `<oryginal>_auto_sync.srt`
- loguje przebieg do `debug_log.txt`; pliki tymczasowe w `temp/audio.wav`

## Wymagania
- Python 3.10+ na Windows
- zaleznosci z `requirements.txt` (whisper/torch moga byc duze)
- GPU opcjonalne; jesli jest dostepny Torch wybierze CUDA
- `ffmpeg.exe`/`ffplay.exe`/`ffprobe.exe` w katalogu projektu (obok `AutoSubtitleSync.py`)

## Instalacja (zalecane)
1. `python -m venv .venv`
2. `.\\.venv\\Scripts\\Activate.ps1`
3. `pip install -r requirements.txt`

## Uruchomienie GUI
1. `python AutoSubtitleSync.py`
2. Wybierz plik wideo i odpowiadajacy mu `.srt`.
3. Wybierz model Whisper (`tiny`, `base`, `small`, `medium`, `large`) i jezyk mowy (lub auto).
4. Kliknij "Synchronizuj napisy automatycznie". Postep i log widoczne w oknie.
5. Wynik znajdziesz w `output/<nazwa>_auto_sync.srt`.

## Uzycie programistyczne
```
from AutoSubtitleSync import auto_sync, Config
auto_sync("film.mp4", "napisy.srt", Config(whisper_model="small", language="pl"))
```

## Struktura repo
- `AutoSubtitleSync.py` - logika VAD + Whisper + GUI Tkinter
- `ffmpeg.exe`, `ffplay.exe`, `ffprobe.exe` - binaria ffmpeg; trzymaj w katalogu projektu obok skryptu
- `output/` - gotowe napisy (tworzone automatycznie)
- `temp/` - pliki pomocnicze (audio.wav)
- `debug_log.txt` - logi ostatnich uruchomien
- `requirements.txt` - lista zaleznosci
- `.venv/` - lokalne srodowisko (duze, mozna usunac i odtworzyc)

## Uwagi
- Whisper wybiera GPU jesli dostepne; w przeciwnym razie dziala na CPU (wolniej)
- Jesli chcesz wymusic jezyk, ustaw go w polu "Jezyk mowy"; opcja "auto" korzysta z detekcji
- GPU AMD: na Windows praktycznie brak wsparcia ROCm (dziala tylko CPU). Na Linux/WSL zainstaluj ROCm i Torch z ROCm (np. `pip install --pre torch --index-url https://download.pytorch.org/whl/rocm6.0`), wtedy `torch.cuda.is_available()` bedzie True i Whisper uzyje GPU AMD. Na macOS GPU AMD nie jest wspierane.
- Jesli usuniesz binaria ffmpeg, pobierz statyczne buildy dla Windows (np. gyan.dev/BtbN) i wklej `ffmpeg.exe`, `ffplay.exe`, `ffprobe.exe` do katalogu projektu.
