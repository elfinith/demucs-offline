"""Проверка офлайн-комплекта: окружение, модели, звук, запрет сети.

Запуск:
    python offline_check.py            # полная проверка (загружает модель, ~20-40 с)
    python offline_check.py --no-model # только окружение и файлы, без прогона сети

Скрипт ничего не скачивает. Если какой-то пункт не проходит — печатает, что
именно исправить, и возвращает код 1.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
import traceback
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    import backing_lib as _lib
    ROOT = _lib.find_root(SRC_DIR)
except Exception as _exc:                             # noqa: BLE001
    print('не удалось загрузить модули (numpy/torch/soundfile?): %s' % _exc)
    print('создайте окружение:  powershell -ExecutionPolicy Bypass -File setup_offline.ps1')
    sys.exit(1)

REQUIRED_MODULES = [
    ('numpy', 'numpy'),
    ('torch', 'torch'),
    ('demucs', 'demucs'),
    ('soundfile', 'soundfile'),
    ('lameenc', 'lameenc'),
    ('julius', 'julius'),
    ('yaml', 'pyyaml'),
    ('tqdm', 'tqdm'),
    ('einops', 'einops'),
]

RESULTS = []


def check(name, fn):
    t0 = time.time()
    try:
        detail = fn()
        RESULTS.append((True, name, detail or '', time.time() - t0))
        print('  [ OK ] %-38s %s' % (name, detail or ''))
        return True
    except Exception as exc:                          # noqa: BLE001
        detail = '%s: %s' % (type(exc).__name__, exc)
        RESULTS.append((False, name, detail, time.time() - t0))
        print('  [FAIL] %-38s %s' % (name, detail))
        return False


def sha256_prefix(path: Path, n: int = 8) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()[:n]


def main() -> int:
    ap = argparse.ArgumentParser(description='Проверка готовности офлайн-комплекта.')
    ap.add_argument('--models-dir', default=str(ROOT / 'models'))
    ap.add_argument('--model', default='htdemucs_6s')
    ap.add_argument('--no-model', action='store_true', help='не запускать пробное разделение')
    ap.add_argument('--allow-network', action='store_true', help='не блокировать сеть при тесте')
    args = ap.parse_args()

    print('корень комплекта: %s' % ROOT)
    print('python: %s' % sys.version.split()[0])
    print()

    # --- 1. окружение
    print('1) python и библиотеки')

    def python_version():
        if sys.version_info[:2] != (3, 12):
            raise RuntimeError('нужен Python 3.12.x, найден %s' % sys.version.split()[0])
        return sys.version.split()[0]

    def arch():
        import platform
        if platform.machine().lower() not in ('amd64', 'x86_64'):
            raise RuntimeError('нужна 64-битная Windows (x64), найдено %s' % platform.machine())
        return platform.machine()

    check('версия python', python_version)
    check('разрядность', arch)

    for mod_name, pkg in REQUIRED_MODULES:
        def _imp(mod_name=mod_name):
            mod = __import__(mod_name)
            ver = getattr(mod, '__version__', '')
            if mod_name == 'torch':
                import torch
                if not torch.__version__.endswith('+cpu') and 'cpu' not in torch.__version__:
                    pass
                return '%s (%s, cpu-only ожидается)' % (ver, 'threads=%d' % torch.get_num_threads())
            return str(ver)
        check('модуль %s' % pkg, _imp)

    # --- 2. файлы комплекта
    print()
    print('2) файлы комплекта')
    models_dir = Path(args.models_dir)
    wheels_dir = ROOT / 'wheels'

    def models_present():
        if not models_dir.is_dir():
            raise FileNotFoundError('нет папки %s' % models_dir)
        yamls = sorted(p.name for p in models_dir.glob('*.yaml'))
        ths = sorted(models_dir.glob('*.th'))
        if args.model + '.yaml' not in yamls:
            raise FileNotFoundError('нет %s.yaml (есть: %s)' % (args.model, ', '.join(yamls) or '—'))
        if not ths:
            raise FileNotFoundError('нет весов *.th')
        return '%d весов, bag-модели: %s' % (len(ths), ', '.join(y.split('.')[0] for y in yamls))

    def checksums():
        bad = []
        for th in models_dir.glob('*.th'):
            stem = th.stem
            if '-' not in stem:
                continue
            expected = stem.rsplit('-', 1)[1]
            actual = sha256_prefix(th, len(expected))
            if actual != expected:
                bad.append('%s (ожидался %s, получен %s)' % (th.name, expected, actual))
        if bad:
            raise RuntimeError('повреждённые веса: ' + '; '.join(bad))
        return 'sha256 весов совпадает с именами файлов'

    def wheels_present():
        if not wheels_dir.is_dir():
            return 'папки wheels нет (нужна только для переустановки)'
        whl = list(wheels_dir.glob('*.whl'))
        if not whl:
            raise FileNotFoundError('в wheels нет *.whl')
        return '%d wheel-файлов' % len(whl)

    check('папка моделей', models_present)
    check('контрольные суммы весов', checksums)
    check('wheel-файлы для переустановки', wheels_present)

    # --- 3. аудио-подсистема (mp3 без ffmpeg)
    print()
    print('3) аудио без ffmpeg (libsndfile + LAME)')

    def mp3_roundtrip():
        import numpy as np
        import soundfile as sf
        import tempfile
        import os
        import backing_lib as lib

        sr = 44100
        t = np.arange(sr, dtype=np.float32) / sr
        tone = 0.3 * np.sin(2 * np.pi * 440 * t)
        x = np.stack([tone, tone * 0.5], axis=1).astype(np.float32)
        data = lib.encode_mp3(x, sr, 192)
        if len(data) < 1000:
            raise RuntimeError('mp3 слишком короткий: %d байт' % len(data))
        tmp = Path(tempfile.gettempdir()) / 'dsh_offline_check.mp3'
        tmp.write_bytes(lib.make_id3v24({'TIT2': 'offline check', 'TPE1': 'test'}) + data)
        y, sr2 = lib.load_audio(tmp)
        os.unlink(tmp)
        if sr2 != sr:
            raise RuntimeError('частота не совпала: %d' % sr2)
        if len(y) < sr * 0.9:
            raise RuntimeError('декодировано слишком мало: %d сэмплов' % len(y))
        return 'mp3 %d кБ, декодировано %.2f с' % (len(data) // 1024, len(y) / sr)

    check('mp3 encode/decode', mp3_roundtrip)

    # --- 4. запрет сети
    print()
    print('4) офлайн-режим')

    def network_blocked():
        import socket
        import backing_lib as lib
        if not args.allow_network:
            lib.block_network()
        try:
            socket.create_connection(('1.1.1.1', 443), timeout=2)
        except OSError:
            try:
                socket.getaddrinfo('pypi.org', 443)
            except OSError:
                return 'соединения и DNS заблокированы'
            raise RuntimeError('DNS не заблокирован')
        raise RuntimeError('сеть НЕ заблокирована — проверка не пройдена')

    check('блокировка сети', network_blocked)

    # --- 5. пробное разделение
    if not args.no_model:
        print()
        print('5) пробное разделение (синтетический сигнал, 6 с)')

        def separation_works():
            import numpy as np
            import torch
            import backing_lib as lib

            sr = 44100
            seconds = 6.0
            t = np.arange(int(sr * seconds), dtype=np.float32) / sr
            bass = 0.25 * np.sin(2 * np.pi * 82 * t)
            mid = 0.18 * np.sin(2 * np.pi * 440 * t)
            beat = np.zeros_like(t)
            for k in range(int(seconds * 2)):
                s = int(k * sr / 2)
                beat[s:s + int(0.03 * sr)] = 0.5 * np.random.default_rng(1).standard_normal(
                    min(int(0.03 * sr), len(beat) - s)).astype(np.float32)
            mix = bass + mid + beat
            stereo = np.stack([mix, mix * 0.9], axis=1).astype(np.float32)

            t0 = time.time()
            model = lib.load_model(args.model, models_dir)
            wav = lib.to_model_audio(stereo, sr, model)
            stems = lib.separate(model, wav, shifts=1, overlap=0.25, progress=False)
            dt = time.time() - t0
            if not stems:
                raise RuntimeError('нет стемов')
            recon = sum(v[:len(stereo)] for v in stems.values())
            a = stereo.mean(1)
            b = recon[:len(a)].mean(1)
            corr = float(np.corrcoef(a, b)[0, 1])
            if corr < 0.9:
                raise RuntimeError('сумма стемов не сходится с исходником (corr=%.3f)' % corr)
            backing, used, _ = lib.build_backing(stems, ('guitar',), 0.8, 0.45)
            if not np.isfinite(backing).all() or float(np.max(np.abs(backing))) < 1e-6:
                raise RuntimeError('минусовка пустая')
            return '%d стемов (%s), corr %.4f, разделение %.1f с' % (
                len(stems), ','.join(stems), corr, dt)

        check('модель %s' % args.model, separation_works)

        def mp3_output():
            import numpy as np
            import backing_lib as lib
            import tempfile
            import os
            import soundfile as sf

            sr = 44100
            x = (0.2 * np.sin(2 * np.pi * 220 * np.arange(sr) / sr)).astype(np.float32)
            x = np.stack([x, x], 1)
            tmp = Path(tempfile.gettempdir()) / 'dsh_offline_check_out.mp3'
            lib.write_mp3(tmp, x, sr, 320, {'TIT2': 'test', 'TPE1': 'test'})
            info = sf.info(str(tmp))
            os.unlink(tmp)
            return 'mp3 на выходе: %.2f с, %d Гц' % (info.frames / info.samplerate, info.samplerate)

        check('запись mp3 с тегами', mp3_output)

    # --- итог
    print()
    failed = [r for r in RESULTS if not r[0]]
    total_time = sum(r[3] for r in RESULTS)
    if failed:
        print('ИТОГ: %d из %d проверок не пройдено (%.1f с)' % (len(failed), len(RESULTS), total_time))
        for _, name, detail, _ in failed:
            print('  - %s: %s' % (name, detail))
        print('\nЧастые причины: не создан venv (запустите setup_offline.ps1), '
              'отсутствуют веса в папке models, python не 3.12 x64.')
        return 1
    print('ИТОГ: всё готово к офлайн-работе (%d проверок, %.1f с)' % (len(RESULTS), total_time))
    print('Запуск: python make_backing.py "путь\\к\\файлу.mp3"')
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print('\nпрервано пользователем', file=sys.stderr)
        sys.exit(130)
    except Exception:                                 # noqa: BLE001
        traceback.print_exc()
        sys.exit(1)
