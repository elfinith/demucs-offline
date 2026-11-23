"""CLI: минусовка (backing track) из произвольного аудиофайла — офлайн.

Примеры:
    python make_backing.py "D:\\music\\song.mp3"
    python make_backing.py "D:\\music\\*.mp3" -o D:\\out --suppress 0.8 --mask-floor 0.45
    python make_backing.py track.mp3 --stems --guitar --keep-sr --verify
    python make_backing.py song.mp3 --max-seconds 30      # быстрый пробный прогон

Работает без интернета: веса берутся из локальной папки models, сеть блокируется
(см. --allow-network), ffmpeg не требуется.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np  # noqa: E402

import backing_lib as lib  # noqa: E402

ROOT = lib.find_root(SRC_DIR)            # корень комплекта (там лежат models/ и wheels/)


# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='make_backing.py',
        description='Минусовка без выбранного инструмента (по умолчанию — без гитары) '
                    'из произвольного аудиофайла. Работает офлайн (CPU).',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('inputs', nargs='+', metavar='ФАЙЛ',
                   help='аудиофайл(ы), папка или маска (например "mp3/*.mp3")')
    p.add_argument('-o', '--out', default='backing_out', metavar='ПАПКА',
                   help='куда складывать результат')
    p.add_argument('-r', '--recursive', action='store_true', help='обходить папки рекурсивно')
    p.add_argument('--force', action='store_true', help='перезаписывать существующие файлы')
    p.add_argument('--dry-run', action='store_true', help='только показать, что будет сделано')

    g = p.add_argument_group('разделение')
    g.add_argument('-n', '--model', default=lib.DEFAULT_MODEL,
                   help='имя модели demucs (htdemucs_6s — 6 стемов с гитарой)')
    g.add_argument('--models-dir', default=str(ROOT / 'models'), metavar='ПАПКА',
                   help='папка с локальными весами *.th и *.yaml')
    g.add_argument('--shifts', type=int, default=2,
                   help='усреднение по сдвигам (1 = быстро, 2-4 = чище и медленнее)')
    g.add_argument('--overlap', type=float, default=0.5, help='перекрытие сегментов 0..1')
    g.add_argument('--threads', type=int, default=0,
                   help='потоков torch (0 = авто, обычно число физ. ядер)')
    g.add_argument('--exclude', default=','.join(lib.DEFAULT_EXCLUDE),
                   help='какие стемы убрать из минусовки (через запятую)')

    g = p.add_argument_group('зачистка остатков')
    g.add_argument('--suppress', type=float, default=0.8,
                   help='0 = выключено; 0.5-1.2 = мягкая винеровская маска по удаляемому стему')
    g.add_argument('--mask-floor', type=float, default=0.45,
                   help='предел подавления одного бина (0.0 = до нуля, 0.45 ≈ -7 дБ)')

    g = p.add_argument_group('громкость и формат')
    g.add_argument('--target-rms-db', type=float, default=None,
                   help='RMS результата в dBFS; по умолчанию — как у исходника')
    g.add_argument('--mp3-bitrate', type=int, default=320, help='битрейт mp3 (CBR)')
    g.add_argument('--no-mp3', action='store_true', help='не писать mp3')
    g.add_argument('--no-wav', action='store_true', help='не писать wav (24 бит)')
    g.add_argument('--keep-sr', action='store_true',
                   help='вернуть исходную частоту дискретизации (по умолчанию 44.1 кГц)')

    g = p.add_argument_group('что ещё сохранить')
    g.add_argument('--stems', action='store_true', help='сохранить все стемы в wav')
    g.add_argument('--no-guitar', action='store_true',
                   help='не сохранять отдельную дорожку удалённого инструмента')
    g.add_argument('--verify', action='store_true',
                   help='контрольный прогон разделения по готовой минусовке (дольше)')

    g = p.add_argument_group('прочее')
    g.add_argument('--max-seconds', type=float, default=None,
                   help='обработать только первые N секунд (для проверки)')
    g.add_argument('--report', default=None, metavar='ФАЙЛ',
                   help='JSON-отчёт (по умолчанию <out>/report.json)')
    g.add_argument('--quiet', action='store_true', help='минимум вывода')
    g.add_argument('--allow-network', action='store_true',
                   help='разрешить сеть (по умолчанию она заблокирована)')
    return p


# --------------------------------------------------------------------------
def discover(inputs, recursive: bool):
    files = []
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            pattern = '**/*' if recursive else '*'
            files.extend(sorted(x for x in p.glob(pattern) if x.is_file()))
        elif p.is_file():
            files.append(p)
        else:
            matches = sorted(Path(m) for m in glob.glob(item, recursive=recursive) if Path(m).is_file())
            files.extend(matches)
    seen, out = set(), []
    for f in files:
        if f.suffix.lower() not in lib.SUPPORTED_INPUTS:
            continue
        key = str(f.resolve()).lower()
        if key not in seen:
            seen.add(key)
            out.append(f)
    return out


def log(msg, quiet=False):
    if not quiet:
        print(msg, flush=True)


# --------------------------------------------------------------------------
def band_share(stem: np.ndarray, source: np.ndarray, sr: int) -> dict:
    """Доля энергии исходника, объясняемая удалённым стемом (%, с ограничением 100)."""
    n = min(len(stem), len(source))
    s0 = stem[:n].mean(1)
    m0 = source[:n].mean(1)
    ss = np.abs(np.fft.rfft(s0 * np.hanning(n))) ** 2
    sm = np.abs(np.fft.rfft(m0 * np.hanning(n))) ** 2
    f = np.fft.rfftfreq(n, 1.0 / sr)
    bands = {'sub_80': (0, 80), 'low_80_250': (80, 250), 'mid_250_1k': (250, 1000),
             'hi_1k_4k': (1000, 4000), 'top_4k': (4000, sr / 2)}
    out = {}
    for name, (lo, hi) in bands.items():
        m = (f >= lo) & (f < hi)
        share = 100.0 * float(ss[m].sum()) / (float(sm[m].sum()) + 1e-20)
        out[name] = round(min(share, 100.0), 1)
    total = 100.0 * float(ss.sum()) / (float(sm.sum()) + 1e-20)
    out['total'] = round(min(total, 100.0), 1)
    return out


def verify_attenuation(model, backing: np.ndarray, guitar_ref: np.ndarray,
                       overlap: float) -> dict:
    """Повторное разделение минусовки: сколько «гитары» в ней ещё находится."""
    import torch

    wav = torch.from_numpy(np.ascontiguousarray(backing.T))
    channels = int(model.audio_channels)
    if wav.shape[0] != channels:
        wav = wav.repeat(channels, 1) if wav.shape[0] == 1 else wav[:channels]
    found = lib.separate(model, wav, shifts=1, overlap=overlap, progress=False)
    if 'guitar' not in found:
        return {'skipped': 'в модели нет стема guitar'}
    a = lib.rms_db(guitar_ref)
    b = lib.rms_db(found['guitar'])
    return {
        'reference_stem_rms_db': round(a, 2),
        'found_in_backing_rms_db': round(b, 2),
        'attenuation_db': round(a - b, 2),
    }


# --------------------------------------------------------------------------
def process_one(path: Path, out_dir: Path, opts, model, args) -> dict:
    t_start = time.time()
    base = lib.safe_name(path)
    result = {'input': str(path), 'output_dir': str(out_dir), 'status': 'ok', 'files': []}

    log('  [1/4] чтение %s' % path.name, args.quiet)
    t0 = time.time()
    data, sr = lib.load_audio(path)
    if args.max_seconds:
        data = data[:int(args.max_seconds * sr)]
    duration = len(data) / float(sr)
    src_rms = lib.rms_db(data)
    src_peak = float(np.max(np.abs(data)))
    src_tags = lib.read_tags(path)
    result.update({
        'duration_s': round(duration, 2),
        'source': {'sr': sr, 'channels': int(data.shape[1]), 'rms_db': round(src_rms, 2),
                   'peak': round(src_peak, 3), 'tags': src_tags},
        'load_s': round(time.time() - t0, 2),
    })
    log('        %.1f с, %d Гц, %d кан., RMS %.1f dBFS' % (duration, sr, data.shape[1], src_rms),
        args.quiet)

    est_mb = lib.estimate_memory_mb(duration, 6 if '6' in opts.model else 4, int(model.samplerate))
    log('  [2/4] разделение (%s, shifts=%d, overlap=%.2f, ~%.0f МБ RAM)'
        % (opts.model, opts.shifts, opts.overlap, est_mb), args.quiet)
    t0 = time.time()
    wav = lib.to_model_audio(data, sr, model)
    show_progress = (not args.quiet) and sys.stderr.isatty()
    stems = lib.separate(model, wav, shifts=opts.shifts, overlap=opts.overlap,
                         threads=opts.threads, progress=show_progress)
    sep_s = time.time() - t0
    result['separation_s'] = round(sep_s, 1)
    result['stems'] = {k: round(lib.rms_db(v), 2) for k, v in stems.items()}
    log('        %s' % ', '.join('%s %.1f dBFS' % (k, lib.rms_db(v)) for k, v in stems.items()),
        args.quiet)

    log('  [3/4] сборка минусовки (исключаем: %s)' % ', '.join(opts.exclude), args.quiet)
    t0 = time.time()
    backing, used, ref_used = lib.build_backing(stems, opts.exclude, opts.suppress, opts.mask_floor)
    result['mix'] = {'used_stems': used, 'excluded': list(opts.exclude),
                     'suppress': opts.suppress, 'mask_floor': opts.mask_floor,
                     'mask_reference': ref_used, 'mix_s': round(time.time() - t0, 2)}

    # доля исходника, приходящаяся на удалённый инструмент (в домене модели)
    if 'guitar' in stems:
        result['guitar_share_pct'] = band_share(stems['guitar'], wav.numpy().T, int(model.samplerate))
        g_db = lib.rms_db(stems['guitar'])
        mix_db = lib.rms_db(wav.numpy().T)
        result['guitar_present'] = bool(g_db > mix_db - 35.0)
        if not result['guitar_present']:
            log('        ВНИМАНИЕ: гитары в треке почти нет (%.1f dBFS против %.1f dBFS микса);'
                ' минусовка будет почти равна оригиналу' % (g_db, mix_db), args.quiet)

    if args.verify and 'guitar' in stems:
        log('  [проверка] повторное разделение минусовки…', args.quiet)
        t0 = time.time()
        try:
            result['verify'] = verify_attenuation(model, backing, stems['guitar'][:len(backing)],
                                                  opts.overlap)
            result['verify']['time_s'] = round(time.time() - t0, 1)
            log('        остаточная гитара: %(found_in_backing_rms_db).1f dBFS '
                '(подавление %(attenuation_db).1f дБ)' % result['verify'], args.quiet)
        except Exception as exc:                      # noqa: BLE001
            result['verify'] = {'error': str(exc)}

    target = opts.target_rms_db if opts.target_rms_db is not None else src_rms
    backing, level_info = lib.match_loudness(backing, target)
    result['level'] = level_info

    if opts.keep_sr and int(sr) != int(model.samplerate):
        backing = lib.resample_audio(backing, int(model.samplerate), int(sr))
        out_sr = int(sr)
    else:
        out_sr = int(model.samplerate)
    result['output_sr'] = out_sr

    # что именно удалили — в тех же координатах и с тем же уровнем
    removed = {}
    if opts.guitar_track:
        for name in opts.exclude:
            if name not in stems:
                continue
            sig = stems[name]
            if opts.keep_sr and int(sr) != int(model.samplerate):
                sig = lib.resample_audio(sig, int(model.samplerate), int(sr))
            removed[name] = lib.match_loudness(sig, target - 3.0)[0]

    log('  [4/4] запись', args.quiet)
    t0 = time.time()
    tag_base = src_tags.get('TIT2') or base
    artist = src_tags.get('TPE1', '')
    album = src_tags.get('TALB', '')

    if opts.write_mp3:
        p = out_dir / ('%s_backing.mp3' % base)
        lib.write_mp3(p, backing, out_sr, opts.mp3_bitrate,
                      {'TIT2': '%s - %s' % (tag_base, opts.tag_suffix),
                       'TPE1': artist, 'TALB': album, 'TCON': src_tags.get('TCON', '')})
        result['files'].append(str(p))
        log('        %s (%.1f МБ)' % (p.name, p.stat().st_size / 1048576.0), args.quiet)

    if opts.write_wav:
        p = out_dir / ('%s_backing.wav' % base)
        lib.write_wav(p, backing, out_sr)
        result['files'].append(str(p))
        log('        %s (%.1f МБ)' % (p.name, p.stat().st_size / 1048576.0), args.quiet)

    for name, sig in removed.items():
        if opts.write_mp3:
            p = out_dir / ('%s_%s.mp3' % (base, name))
            lib.write_mp3(p, sig, out_sr, opts.mp3_bitrate,
                          {'TIT2': '%s - %s only' % (tag_base, name), 'TPE1': artist, 'TALB': album})
            result['files'].append(str(p))
        if opts.write_wav:
            p = out_dir / ('%s_%s.wav' % (base, name))
            lib.write_wav(p, sig, out_sr)
            result['files'].append(str(p))
        log('        %s_%s.{mp3,wav}' % (base, name), args.quiet)

    if opts.stems:
        stems_dir = out_dir / ('%s_stems' % base)
        stems_dir.mkdir(parents=True, exist_ok=True)
        for name, sig in stems.items():
            p = stems_dir / ('%s.wav' % name)
            lib.write_wav(p, sig, int(model.samplerate))
            result['files'].append(str(p))
        log('        %s\\*.wav (%d стемов)' % (stems_dir.name, len(stems)), args.quiet)

    result['write_s'] = round(time.time() - t0, 2)
    result['total_s'] = round(time.time() - t_start, 1)
    return result


# --------------------------------------------------------------------------
def _check_dependencies() -> str:
    """Понятное сообщение, если окружение не создано."""
    missing = []
    for mod in ('numpy', 'torch', 'demucs', 'soundfile', 'lameenc', 'julius'):
        try:
            __import__(mod)
        except Exception:                              # noqa: BLE001
            missing.append(mod)
    if missing:
        return ('не хватает python-пакетов: %s\n'
                'Создайте окружение:  setup_offline.cmd   (или setup_offline.ps1)\n'
                'Либо запустите через .venv: .venv\\Scripts\\python.exe src\\make_backing.py ...'
                % ', '.join(missing))
    return ''


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if not args.allow_network:
        lib.block_network()
    if args.no_mp3 and args.no_wav:
        print('нечего сохранять: указаны и --no-mp3, и --no-wav', file=sys.stderr)
        return 2

    models_dir = Path(args.models_dir).expanduser()
    files = discover(args.inputs, args.recursive)
    if not files:
        print('не найдено подходящих файлов (%s)' % ', '.join(lib.SUPPORTED_INPUTS), file=sys.stderr)
        print('проверьте путь и расширение: %s' % ', '.join(args.inputs), file=sys.stderr)
        return 2
    if args.dry_run:
        for f in files:
            log('  %s -> %s' % (f, Path(args.out) / (lib.safe_name(f) + '_backing.mp3')), args.quiet)
        return 0
    deps_error = _check_dependencies()
    if deps_error:
        print(deps_error, file=sys.stderr)
        return 3

    out_dir = Path(args.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    log('файлов к обработке: %d' % len(files), args.quiet)

    opts = lib.Options(
        model=args.model, models_dir=models_dir,
        exclude=tuple(s for s in args.exclude.split(',') if s.strip()),
        suppress=args.suppress, mask_floor=args.mask_floor,
        shifts=args.shifts, overlap=args.overlap, threads=args.threads,
        target_rms_db=args.target_rms_db, mp3_bitrate=args.mp3_bitrate,
        write_mp3=not args.no_mp3, write_wav=not args.no_wav,
        guitar_track=not args.no_guitar, stems=args.stems, keep_sr=args.keep_sr,
    )

    try:
        model = lib.load_model(opts.model, models_dir)
    except Exception as exc:                          # noqa: BLE001
        print('не удалось загрузить модель: %s' % exc, file=sys.stderr)
        print('проверьте папку моделей: %s' % models_dir, file=sys.stderr)
        return 3

    log('модель: %s | источники: %s' % (opts.model, ', '.join(model.sources)), args.quiet)

    report_path = Path(args.report) if args.report else (out_dir / 'report.json')
    report = {'tool': 'make_backing.py', 'model': opts.model, 'items': []}
    if report_path.is_file():
        try:
            old = json.loads(report_path.read_text(encoding='utf-8'))
            if isinstance(old, dict) and isinstance(old.get('items'), list):
                report['items'] = old['items']
        except Exception:                              # noqa: BLE001
            pass

    failed = 0
    t_all = time.time()
    for idx, path in enumerate(files, 1):
        base = lib.safe_name(path)
        expected = out_dir / ('%s_backing.mp3' % base)
        if expected.exists() and not args.force:
            log('[%d/%d] %s — уже готово, пропускаю (--force чтобы перезаписать)'
                % (idx, len(files), path.name), args.quiet)
            continue
        log('[%d/%d] %s' % (idx, len(files), path.name), args.quiet)
        try:
            item = process_one(path, out_dir, opts, model, args)
        except Exception as exc:                      # noqa: BLE001
            failed += 1
            item = {'input': str(path), 'status': 'error', 'error': '%s: %s' % (type(exc).__name__, exc)}
            print('  ОШИБКА: %s' % item['error'], file=sys.stderr)
        report['items'].append(item)
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')

    elapsed = time.time() - t_all
    ok = [it for it in report['items'] if it.get('status') == 'ok']
    log('', args.quiet)
    log('готово: %d из %d за %s' % (len(ok), len(files), lib.human_duration(elapsed)), args.quiet)
    if ok:
        log('результат: %s' % out_dir.resolve(), args.quiet)
        for it in ok:
            log('  %-40s %s' % (Path(it['input']).name,
                               ', '.join(Path(f).name for f in it.get('files', [])[:2])), args.quiet)
    log('отчёт: %s' % report_path.resolve(), args.quiet)
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
