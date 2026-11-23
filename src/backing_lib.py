"""Офлайн-разделение трека на стемы и сборка минусовки (без интернета).

Библиотека не делает ни одного сетевого запроса: веса модели берутся из
локальной папки ``models`` (формат demucs: ``<sig>-<sha256prefix>.th`` плюс
yaml с описанием bag'а), звук читается и пишется через libsndfile (mp3, wav,
flac, ogg), mp3 кодируется lameenc (LAME) — ffmpeg не нужен.

Основные функции:
    block_network()                      запрет любых сетевых соединений
    load_audio(path)                     -> (np.float32 [T, C], sr)
    load_model(name, models_dir)         -> demucs BagOfModels
    separate(model, wav, ...)            -> {stem: np.float32 [T, C]}
    build_backing(stems, exclude, ...)   -> (mix, used_stems)
    match_loudness(x, target_rms_db)     -> (x, info)
    write_wav / write_mp3                -> файлы на диске
    read_tags / make_id3v24              -> теги
"""

from __future__ import annotations

import socket
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

DEFAULT_MODEL = 'htdemucs_6s'
DEFAULT_EXCLUDE = ('guitar',)

SUPPORTED_INPUTS = ('.mp3', '.wav', '.flac', '.ogg', '.oga', '.aiff', '.aif')


# --------------------------------------------------------------------------
#  offline guard
# --------------------------------------------------------------------------
_LOCAL_HOSTS = {'127.0.0.1', '::1', 'localhost', '0.0.0.0', ''}


def block_network(allow_localhost: bool = True) -> None:
    """Запретить исходящие сетевые соединения и DNS в текущем процессе.

    Страховка от случайных обращений в интернет (huggingface hub, torch.hub,
    телеметрия и т.п.): любая попытка открыть сокет наружу даёт OSError.
    """
    if getattr(block_network, '_installed', False):
        return

    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex
    original_getaddrinfo = socket.getaddrinfo

    def _check(addr) -> None:
        host = addr[0] if isinstance(addr, (tuple, list)) and addr else addr
        if allow_localhost and host in _LOCAL_HOSTS:
            return
        raise OSError(101, 'offline mode: network access is blocked (%r)' % (host,))

    def connect(self, addr):
        _check(addr)
        return original_connect(self, addr)

    def connect_ex(self, addr):
        _check(addr)
        return original_connect_ex(self, addr)

    def getaddrinfo(host, port, *args, **kwargs):
        if allow_localhost and host in _LOCAL_HOSTS:
            return original_getaddrinfo(host, port, *args, **kwargs)
        raise OSError(101, 'offline mode: DNS is blocked (%r)' % (host,))

    def create_connection(address, *args, **kwargs):
        raise OSError(101, 'offline mode: network access is blocked (%r)' % (address,))

    socket.socket.connect = connect
    socket.socket.connect_ex = connect_ex
    socket.getaddrinfo = getaddrinfo
    socket.create_connection = create_connection
    block_network._installed = True


# --------------------------------------------------------------------------
#  audio i/o
# --------------------------------------------------------------------------
def _quiet_native_stderr():
    """Временно увести stderr в /dev/null — некоторые кодеки (mpg123) ругаются
    на «странные» ID3-теги прямо из C-кода, и это выглядит как ошибка."""
    import contextlib
    import os

    @contextlib.contextmanager
    def _ctx():
        try:
            old_fd = os.dup(2)
            devnull = os.open(os.devnull, os.O_WRONLY)
            os.dup2(devnull, 2)
        except OSError:
            yield
            return
        try:
            yield
        finally:
            os.dup2(old_fd, 2)
            os.close(devnull)
            os.close(old_fd)

    return _ctx()


def load_audio(path: Path) -> Tuple[np.ndarray, int]:
    """Прочитать поддерживаемый файл в float32 [T, C] и частоту дискретизации."""
    import soundfile as sf

    try:
        with _quiet_native_stderr():
            data, sr = sf.read(str(path), dtype='float32', always_2d=True)
    except Exception as exc:                          # noqa: BLE001
        raise RuntimeError('не удалось декодировать %s (%s). '
                           'Поддерживаются mp3/wav/flac/ogg; проверьте целостность файла.'
                           % (path.name, exc)) from exc
    if data.size == 0 or len(data) == 0:
        raise ValueError('файл не содержит звука: %s' % path)
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    return np.ascontiguousarray(data), int(sr)


def to_model_audio(data: np.ndarray, sr: int, model):
    """Привести звук к формату модели: torch [C, T], нужная частота и каналы."""
    import julius
    import torch

    x = torch.from_numpy(np.ascontiguousarray(data.T))  # [C, T]
    channels = int(model.audio_channels)
    if x.shape[0] < channels:
        if channels % x.shape[0] == 0:
            x = torch.cat([x] * (channels // x.shape[0]), 0)
        else:
            mono = x.mean(0, keepdim=True)
            x = torch.cat([mono] * channels, 0)
    elif x.shape[0] > channels:
        x = x[:channels]
    if int(sr) != int(model.samplerate):
        x = julius.resample_frac(x, int(sr), int(model.samplerate))
    return x.contiguous()


def resample_audio(x: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
    """Ресемплинг numpy-массива [T, C] (качественный, через julius)."""
    if int(sr_from) == int(sr_to):
        return x
    import julius
    import torch

    t = torch.from_numpy(np.ascontiguousarray(x.T))
    y = julius.resample_frac(t, int(sr_from), int(sr_to))
    return y.numpy().T.copy()


def write_wav(path: Path, x: np.ndarray, sr: int, subtype: str = 'PCM_24') -> None:
    import soundfile as sf

    sf.write(str(path), np.clip(x, -1.0, 1.0), int(sr), subtype=subtype)


# --------------------------------------------------------------------------
#  mp3 export (LAME via lameenc) + ID3v2.4
# --------------------------------------------------------------------------
def encode_mp3(x: np.ndarray, sr: int, bitrate: int = 320) -> bytes:
    import lameenc

    pcm = np.clip(x, -1.0, 1.0)
    pcm16 = (pcm * 32767.0).astype(np.int16)
    enc = lameenc.Encoder()
    enc.set_bit_rate(int(bitrate))
    enc.set_in_sample_rate(int(sr))
    enc.set_channels(int(pcm16.shape[1]))
    enc.set_quality(2)
    return enc.encode(pcm16.tobytes()) + enc.flush()


def make_id3v24(tags: Dict[str, str]) -> bytes:
    """Корректный ID3v2.4-заголовок с UTF-8 текстовыми кадрами."""
    frames = b''
    for fid, text in tags.items():
        if not text or len(fid) != 4:
            continue
        body = b'\x03' + str(text).encode('utf-8')
        frames += fid.encode('ascii') + struct.pack('>I', len(body)) + b'\x00\x00' + body
    if not frames:
        return b''
    size = len(frames)
    syncsafe = bytes([(size >> 21) & 0x7F, (size >> 14) & 0x7F, (size >> 7) & 0x7F, size & 0x7F])
    return b'ID3\x04\x00\x00' + syncsafe + frames


def strip_id3v2(data: bytes) -> bytes:
    if data[:3] != b'ID3' or len(data) < 10:
        return data
    size = ((data[6] & 0x7F) << 21) | ((data[7] & 0x7F) << 14) | ((data[8] & 0x7F) << 7) | (data[9] & 0x7F)
    return data[10 + size:]


def write_mp3(path: Path, x: np.ndarray, sr: int, bitrate: int = 320,
              tags: Optional[Dict[str, str]] = None) -> None:
    audio = encode_mp3(x, sr, bitrate)
    header = make_id3v24(tags or {})
    with open(path, 'wb') as f:
        f.write(header)
        f.write(audio)


def read_tags(path: Path) -> Dict[str, str]:
    """Достать TIT2/TPE1/TALB из ID3v2 исходника (чтобы перенести в результат)."""
    try:
        with open(path, 'rb') as f:
            raw = f.read(512 * 1024)
    except OSError:
        return {}
    if raw[:3] != b'ID3' or len(raw) < 10:
        return {}
    major = raw[3]
    size = ((raw[6] & 0x7F) << 21) | ((raw[7] & 0x7F) << 14) | ((raw[8] & 0x7F) << 7) | (raw[9] & 0x7F)
    tags: Dict[str, str] = {}
    i, end = 10, min(10 + size, len(raw))
    while i + 10 <= end:
        fid = raw[i:i + 4]
        if not fid.strip(b'\x00') or not fid[:1].isalpha():
            break
        if major == 4:
            fsz = ((raw[i + 4] & 0x7F) << 21) | ((raw[i + 5] & 0x7F) << 14) \
                | ((raw[i + 6] & 0x7F) << 7) | (raw[i + 7] & 0x7F)
        else:
            fsz = int.from_bytes(raw[i + 4:i + 8], 'big')
        if fsz <= 0 or i + 10 + fsz > len(raw):
            break
        body = raw[i + 10:i + 10 + fsz]
        name = fid.decode('ascii', 'replace')
        if name in ('TIT2', 'TPE1', 'TALB', 'TCON') and body:
            enc = body[0]
            try:
                if enc == 0:
                    text = body[1:].decode('latin-1', 'replace')
                elif enc == 1:
                    text = body[1:].decode('utf-16', 'replace')
                elif enc == 2:
                    text = body[1:].decode('utf-16-be', 'replace')
                else:
                    text = body[1:].decode('utf-8', 'replace')
            except Exception:
                text = ''
            tags[name] = text.strip('\x00').strip()
        i += 10 + fsz
    return tags


# --------------------------------------------------------------------------
#  model loading (строго локально)
# --------------------------------------------------------------------------
def load_model(name: str = DEFAULT_MODEL, models_dir: Optional[Path] = None):
    """Загрузить bag моделей demucs из локальной папки (без обращения в сеть)."""
    from demucs.pretrained import get_model

    if models_dir is None:
        raise ValueError('нужна локальная папка с весами моделей (--models-dir)')
    repo = Path(models_dir)
    if not repo.is_dir():
        raise FileNotFoundError('папка с моделями не найдена: %s' % repo)
    yaml_path = repo / ('%s.yaml' % name)
    if not yaml_path.is_file():
        available = ', '.join(sorted(p.stem for p in repo.glob('*.yaml'))) or '(нет)'
        raise FileNotFoundError(
            'нет описания модели %s: ожидался файл %s\nдоступные модели: %s'
            % (name, yaml_path, available))
    weights = list(repo.glob('*.th'))
    if not weights:
        raise FileNotFoundError('в %s нет файлов весов *.th' % repo)
    return get_model(name, repo=repo)


# --------------------------------------------------------------------------
#  separation / mixing
# --------------------------------------------------------------------------
def separate(model, wav, shifts: int = 1, overlap: float = 0.35,
             threads: int = 0, progress: bool = True) -> Dict[str, np.ndarray]:
    """Разделить torch [C, T] на стемы: {имя: np.float32 [T, C]}."""
    import torch
    from demucs.apply import apply_model

    if threads:
        torch.set_num_threads(int(threads))
    ref = wav.mean(0)
    mix = (wav - ref.mean()) / (ref.std() + 1e-8)
    with torch.no_grad():
        sources = apply_model(model, mix[None], device='cpu', shifts=int(shifts),
                              split=True, overlap=float(overlap), progress=progress)[0]
    sources = sources * ref.std() + ref.mean()
    return {name: sources[i].numpy().T.copy() for i, name in enumerate(model.sources)}


def _stft(x: np.ndarray, nfft: int, hop: int):
    win = np.hanning(nfft).astype(np.float32)
    n = 1 + int(np.ceil(max(0, len(x) - nfft) / float(hop)))
    pad = max(0, nfft + hop * (n - 1) - len(x))
    xp = np.concatenate([x, np.zeros(pad, dtype=np.float32)])
    frames = np.lib.stride_tricks.as_strided(
        xp, shape=(n, nfft), strides=(xp.strides[0] * hop, xp.strides[0]))
    return np.fft.rfft(frames * win, axis=1), win, len(x)


def _istft(spec: np.ndarray, win: np.ndarray, hop: int, length: int) -> np.ndarray:
    frames = np.fft.irfft(spec, n=len(win), axis=1).astype(np.float32) * win
    out = np.zeros(hop * (len(frames) - 1) + len(win), dtype=np.float32)
    wsum = np.zeros_like(out)
    for i, fr in enumerate(frames):
        s = i * hop
        out[s:s + len(win)] += fr
        wsum[s:s + len(win)] += win ** 2
    return out[:length] / np.maximum(wsum[:length], 1e-8)


def _smooth_rows(mask: np.ndarray, k: int = 5) -> np.ndarray:
    kernel = np.ones(k, dtype=np.float32) / float(k)
    out = np.empty_like(mask)
    for i in range(mask.shape[0]):
        out[i] = np.convolve(mask[i], kernel, mode='same')
    return out


def build_backing(stems: Dict[str, np.ndarray], exclude: Sequence[str] = DEFAULT_EXCLUDE,
                  suppress: float = 0.0, mask_floor: float = 0.0
                  ) -> Tuple[np.ndarray, List[str], Optional[str]]:
    """Сумма стемов без исключённых + опциональная зачистка остатков.

    suppress > 0 включает винеровскую маску по STFT: бины, объясняемые
    удаляемым стемом, приглушаются (не сильнее, чем до mask_floor).
    Возвращает (микс, использованные стемы, имя стема-«донора» для маски).
    """
    excl = {s.strip() for s in exclude if s and s.strip()}
    used = [k for k in stems if k not in excl]
    if not used:
        raise ValueError('нечего микшировать: исключены все стемы %s' % sorted(excl))
    n = min(len(v) for v in stems.values())
    channels = next(iter(stems.values())).shape[1]
    mix = np.zeros((n, channels), dtype=np.float32)
    for k in used:
        mix += stems[k][:n]

    ref_name = None
    if suppress > 0:
        for cand in ('guitar', 'vocals', 'other', 'piano', 'bass', 'drums'):
            if cand in stems and cand in excl:
                ref_name = cand
                break
    if ref_name is not None:
        ref = stems[ref_name][:n]
        out = np.zeros_like(mix)
        for ch in range(channels):
            M, win, length = _stft(mix[:, ch], 4096, 1024)
            G, _, _ = _stft(ref[:, ch], 4096, 1024)
            pm, pg = np.abs(M) ** 2, np.abs(G) ** 2
            mask = pm / (pm + (float(suppress) ** 2) * pg + 1e-10)
            mask = _smooth_rows(mask, 5)
            mask = np.clip(mask, float(mask_floor), 1.0)
            out[:, ch] = _istft(M * mask, win, 1024, length)
        mix = out
    return mix, used, ref_name


# --------------------------------------------------------------------------
#  loudness
# --------------------------------------------------------------------------
def rms_db(x: np.ndarray) -> float:
    return float(20.0 * np.log10(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2)) + 1e-12))


def match_loudness(x: np.ndarray, target_rms_db: float, peak_target: float = 0.94,
                   knee: float = 0.88) -> Tuple[np.ndarray, Dict[str, float]]:
    """Привести RMS к целевому уровню, затем мягко ограничить пики."""
    x = np.asarray(x, dtype=np.float32)
    rms_in = float(np.sqrt(np.mean(x.astype(np.float64) ** 2)) + 1e-12)
    peak_in = float(np.max(np.abs(x)) + 1e-12)
    gain = 10.0 ** (target_rms_db / 20.0) / rms_in
    y = x * gain
    limited = 0
    peak_now = float(np.max(np.abs(y)))
    if peak_now > 0.95:
        mag = np.abs(y)
        over = mag > knee
        limited = int(over.sum())
        sgn = np.sign(y)
        y = np.where(over, sgn * (knee + (1.0 - knee) * np.tanh((mag - knee) / (1.0 - knee))), y)
        peak_now = float(np.max(np.abs(y)))
    if peak_now > peak_target:
        y *= peak_target / peak_now
    info = {
        'gain_db': round(20.0 * np.log10(gain), 2),
        'rms_in_db': round(rms_db(x), 2),
        'rms_out_db': round(rms_db(y), 2),
        'peak_in': round(peak_in, 4),
        'peak_out': round(float(np.max(np.abs(y))), 4),
        'limited_samples': limited,
    }
    return np.ascontiguousarray(y, dtype=np.float32), info


# --------------------------------------------------------------------------
#  misc helpers
# --------------------------------------------------------------------------
def find_root(start: Path) -> Path:
    """Найти корень комплекта: ближайший вверх каталог, где есть models/."""
    start = Path(start).resolve()
    for cand in (start, start.parent, start.parent.parent, start.parent.parent.parent):
        if (cand / 'models').is_dir():
            return cand
    return start.parent


def safe_name(path: Path) -> str:
    base = path.stem.strip() or 'track'
    bad = '<>:"/\\|?*'
    base = ''.join('_' if c in bad else c for c in base).strip(' .')
    return base or 'track'


def human_duration(sec: Optional[float]) -> str:
    if sec is None:
        return '?'
    sec = int(round(sec))
    return '%d:%02d' % (sec // 60, sec % 60)


def estimate_memory_mb(seconds: float, stems_count: int, sr: int = 44100) -> float:
    """Грубая оценка пиковой памяти под стемы (float32, 2 канала)."""
    return seconds * sr * 2 * 4 * stems_count / (1024.0 * 1024.0)


@dataclass
class Options:
    """Параметры обработки одного файла."""
    model: str = DEFAULT_MODEL
    models_dir: Optional[Path] = None
    exclude: Tuple[str, ...] = DEFAULT_EXCLUDE
    suppress: float = 0.0
    mask_floor: float = 0.0
    shifts: int = 1
    overlap: float = 0.35
    threads: int = 0
    target_rms_db: Optional[float] = None     # None => повторить уровень источника
    mp3_bitrate: int = 320
    write_mp3: bool = True
    write_wav: bool = True
    guitar_track: bool = True
    stems: bool = False
    keep_sr: bool = False
    tag_suffix: str = 'backing track'
