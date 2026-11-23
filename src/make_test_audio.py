"""Сгенерировать тестовые mp3 разных форматов (для проверки офлайн-комплекта)."""
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
import backing_lib as lib

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else 'test_audio')
OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(7)


def music(seconds, sr, channels):
    t = np.arange(int(seconds * sr), dtype=np.float32) / sr
    # «минус»: бас + аккорды + перкуссия
    bass = 0.22 * np.sin(2 * np.pi * 82 * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 0.5 * t))
    chord = np.zeros_like(t)
    for f in (220.0, 277.2, 330.0):
        chord += 0.09 * np.sin(2 * np.pi * f * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 0.25 * t))
    beat = np.zeros_like(t)
    for k in range(int(seconds * 2)):
        s = int(k * sr / 2)
        n = min(int(0.04 * sr), len(beat) - s)
        if n > 0:
            beat[s:s + n] += 0.35 * rng.standard_normal(n).astype(np.float32) * np.linspace(1, 0, n)
    # «рифф»: пила с огибающей в полосе 300-800 Гц (то, что должно уйти)
    riff = 0.16 * (2 * ((t * 196) % 1) - 1) * (0.5 + 0.5 * np.sin(2 * np.pi * 2.0 * t))
    mix = bass + chord + beat + riff
    mix = mix / (np.max(np.abs(mix)) + 1e-9) * 0.85
    if channels == 2:
        return np.stack([mix, mix * 0.92 + 0.02 * rng.standard_normal(len(mix)).astype(np.float32)], 1)
    return mix[:, None]


cases = [
    ('Тест моно 22к', 22050, 1, 'normal', 8.0),
    ('test spaces and (parens)', 44100, 2, 'normal', 8.0),
    ('тест 48к стерео', 48000, 2, 'normal', 8.0),
    ('test_mono_8k', 8000, 1, 'loud', 6.0),
]

for name, sr, ch, kind, seconds in cases:
    x = music(seconds, sr, ch)
    if kind == 'loud':
        x = np.clip(x * 1.4, -1, 1)
    path = OUT / (name + '.mp3')
    tags = {'TIT2': name, 'TPE1': 'Test Artist', 'TALB': 'Offline test'}
    lib.write_mp3(path, x, sr, 192, tags)
    print('%-32s %5d Гц %dch %5.1f с  %6.1f КБ  %s' % (
        path.name, sr, ch, seconds, path.stat().st_size / 1024, path))
