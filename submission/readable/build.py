"""Build the readable submission without modifying the source package.

Reads the consolidated package, copies executable inputs without changing their
bytes, and writes a separate package, editable review sources and CSV evidence.
Run with --help for locations. Verification records remain beside the package.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    return fields


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def scalar_fields(data, prefix=''):
    out = {}
    for k, v in data.items():
        name = prefix + k
        if isinstance(v, dict):
            out.update(scalar_fields(v, name + '_'))
        elif isinstance(v, (str, int, float, bool)):
            out[name] = v
    return out


def build(source, package):
    package.mkdir(parents=True, exist_ok=True)
    editable = HERE / 'editable_sources' / 'review'
    editable.mkdir(parents=True, exist_ok=True)
    inventory, copied = [], []
    snapshot_paths = set(read_json(source / 'numpy_impl/SNAPSHOT.json')['files_sha256'])
    for p in sorted(source.rglob('*')):
        if not p.is_file():
            continue
        n = p.relative_to(source).as_posix()
        label, destination, reason = 'Read', None, 'Replaced by reader documentation'
        if n.startswith(('numpy_impl/', 'jax_impl/')):
            if p.suffix in ('.py', '.npz', '.json', '.jsonl', '.csv', '.log', '.tex', '.in') or (p.name.startswith('requirements') and p.suffix == '.txt'):
                # Existing scripts read these formats by relative path. Keep their
                # recorded content, including hashes and logs used by number scripts.
                label = 'Runs' if p.suffix in ('.py', '.npz', '.json', '.tex', '.in') or p.name.startswith('requirements') else 'Raw data'
                destination = package / 'code' / n
                reason = 'Preserved input or executable; original relative paths retained'
            elif n.startswith('numpy_impl/') and n.removeprefix('numpy_impl/') in snapshot_paths:
                label, destination, reason = 'Runs', package / 'code' / n, 'Retained because packaged snapshot manifest names this file'
            elif p.suffix.lower() in ('.png', '.jpg', '.pdf'):
                label, reason = 'Source for an output', 'Plot galleries omitted; report figures retained with editable sources'
            elif p.name == '.gitkeep':
                label, reason = 'Empty', 'Empty-folder marker omitted'
        elif n == 'smoke_test.py':
            label, destination, reason = 'Runs', package / 'code' / n, 'Unchanged combined installation check'
        elif n.startswith('review/'):
            label, reason = 'Source for an output', 'Editable report kept next to submission'
            if p.suffix not in ('.pdf', '.log', '.aux', '.out'):
                destination = HERE / 'editable_sources' / n
            if p.suffix == '.py':
                # Required full-code deliverable: preserve the original sibling
                # layout for the scripts that regenerate the review's numbers.
                q = package / 'code' / n
                q.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, q)
                copied.append({'source': n, 'destination': q.relative_to(package).as_posix(), 'sha256': digest(p)})
        elif n == 'Contribution_statement.pdf':
            destination, reason = package / n, 'Required contribution record'
        elif p.name.startswith('requirements'):
            label, destination, reason = 'Runs', package / p.name, 'Installation dependencies'
        elif n == 'MANIFEST.json':
            label, reason = 'Machine-only', 'Build verification retained outside submission'
        elif n.startswith('declarations/'):
            label, reason = 'Empty', 'Authors supply their completed declarations separately'
        if destination:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, destination)
            if destination.is_relative_to(package):
                copied.append({'source': n, 'destination': destination.relative_to(package).as_posix(), 'sha256': digest(p)})
        inventory.append({'original_file': n, 'label': label, 'action': reason, 'new_location': str(destination.relative_to(HERE)) if destination else ''})

    # These are empty output destinations expected by the unchanged figure code.
    (package / 'code/review/figures').mkdir(parents=True, exist_ok=True)
    for name in ('numpy_impl', 'jax_impl'):
        (package / 'code' / name / 'results/figures').mkdir(parents=True, exist_ok=True)
    evidence = package / 'evidence'
    evidence.mkdir(exist_ok=True)
    np_runs = source / 'numpy_impl/results/runs'
    runs, modules, histories = [], [], []
    for d in sorted(np_runs.iterdir()):
        if not d.is_dir() or not (d / 'config.json').exists():
            continue
        cfg = read_json(d / 'config.json')
        metric = read_json(d / 'metrics.json')
        origin = d.relative_to(source).as_posix()
        runs.append({'original_run': origin, **scalar_fields(cfg, 'setting_'), **scalar_fields(metric, 'result_')})
        for i, module in enumerate(metric.get('modules', [])):
            modules.append({'original_run': origin, 'module_index': i, **scalar_fields(module)})
        h = d / 'hist.npz'
        if h.exists():
            with np.load(h, allow_pickle=False) as z:
                arrays = {k: z[k] for k in z.files}
                sizes = [len(v) for v in arrays.values() if v.ndim == 1]
                count = max(sizes, default=0)
                for i in range(count):
                    histories.append({'implementation': 'NumPy', 'original_run': origin, 'record_index': i,
                                      **{k: v[i].item() for k, v in arrays.items() if v.ndim == 1 and len(v) == count}})
    write_csv(evidence / 'numpy_runs.csv', runs)
    write_csv(evidence / 'numpy_modules.csv', modules)
    for h in sorted((source / 'jax_impl/results').rglob('history.jsonl')):
        for i, line in enumerate(h.read_text().splitlines()):
            if line.strip():
                histories.append({'implementation': 'JAX', 'original_run': h.parent.relative_to(source).as_posix(), 'record_index': i, **json.loads(line)})
    write_csv(evidence / 'training_history.csv', histories)
    mapping = {
        'jax_diagnostics.csv': 'jax_impl/results/diagnostic_study/report/runs.csv',
        'common_evaluation.csv': 'jax_impl/results/partner_results_20261026/common_evaluation.csv',
        'sampling_batches.csv': 'jax_impl/results/constraint_control/batches.csv',
    }
    for name, src in mapping.items():
        with (source / src).open(encoding='utf-8-sig', newline='') as f:
            values = list(csv.DictReader(f))
        for row in values:
            row['original_run'] = src
            row.pop('checkpoint_sha256', None)
        write_csv(evidence / name, values)
    control = read_json(source / 'jax_impl/results/constraint_control/summary.json')
    write_csv(evidence / 'sampling_summary.csv', [{'original_run': 'jax_impl/results/constraint_control', **m} for m in control['measurements']])
    # Flatten every aggregate value without rounding, including arrays. This is
    # the supporting table for scaling, angles, sweeps and ablations.
    aggregate = []
    def leaves(value, key):
        if isinstance(value, dict):
            for k, v in value.items():
                yield from leaves(v, key + '/' + k)
        elif isinstance(value, list):
            for i, v in enumerate(value):
                yield from leaves(v, key + '/' + str(i))
        else:
            yield key, value
    for f in sorted((source / 'numpy_impl/results/tables').glob('*.json')):
        for key, value in leaves(read_json(f), ''):
            aggregate.append({'original_run': f.relative_to(source).as_posix(), 'measurement': key.lstrip('/'), 'value': json.dumps(value, ensure_ascii=False), 'value_type': type(value).__name__})
    write_csv(evidence / 'numpy_aggregate_results.csv', aggregate)
    # Round-trip all scalar CSV cells. Numeric strings retain full precision;
    # no report number is rounded or recomputed by the restructuring step.
    for name, expected in [('numpy_runs.csv', runs), ('numpy_modules.csv', modules), ('training_history.csv', histories), ('numpy_aggregate_results.csv', aggregate)]:
        with (evidence / name).open(newline='', encoding='utf-8') as f:
            actual = list(csv.DictReader(f))
        assert len(actual) == len(expected), name
        for a, e in zip(actual, expected):
            for k, v in e.items():
                assert a[k] == ('' if v is None else str(v)), (name, k)
    for row in copied:
        assert digest(package / row['destination']) == row['sha256'], row
    write_csv(HERE / 'file_inventory.csv', inventory)
    (HERE / 'preserved_files.json').write_text(json.dumps(copied, indent=2) + '\n')
    (HERE / 'source_fingerprint.json').write_text(json.dumps({p.relative_to(source).as_posix(): digest(p) for p in source.rglob('*') if p.is_file()}, indent=2) + '\n')
    print(f'Preserved {len(copied)} files; exported {len(runs)} NumPy runs, {len(modules)} modules and {len(histories)} history rows.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=REPO / 'dist/NeuroProject_combined_corrected')
    parser.add_argument('--destination', type=Path, default=HERE / 'NeuroProject_2683459_2717495')
    args = parser.parse_args()
    build(args.source.resolve(), args.destination.resolve())
