"""Verify the reorganised package against preserved source files and CSV values.

Reads the original package, the copied inputs and the exported evidence.
Writes no scientific results. Verification notes stay outside the submission.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import re

HERE = Path(__file__).resolve().parent


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def rows(p):
    with p.open(newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def verify(source, package):
    original = json.loads((HERE / 'source_fingerprint.json').read_text())
    for name, expected in original.items():
        assert sha(source / name) == expected, f'Original source changed: {name}'
    preserved = json.loads((HERE / 'preserved_files.json').read_text())
    for item in preserved:
        assert sha(package / item['destination']) == item['sha256'], f'Copied input changed: {item["destination"]}'
    snap = json.loads((package / 'code/numpy_impl/SNAPSHOT.json').read_text())
    for name, expected in snap['files_sha256'].items():
        assert sha(package / 'code/numpy_impl' / name) == expected, f'Snapshot member missing or changed: {name}'
    evidence = package / 'evidence'
    for table, original_table in [('common_evaluation.csv', 'jax_impl/results/partner_results_20261026/common_evaluation.csv'), ('jax_diagnostics.csv', 'jax_impl/results/diagnostic_study/report/runs.csv'), ('sampling_batches.csv', 'jax_impl/results/constraint_control/batches.csv')]:
        old, new = rows(source / original_table), rows(evidence / table)
        assert len(old) == len(new), table
        for a, b in zip(old, new):
            for k, v in a.items():
                if k != 'checkpoint_sha256':
                    assert b[k] == v, (table, k)
    numpy = rows(evidence / 'numpy_runs.csv')
    assert len(numpy) == 60
    for r in numpy:
        original_metric = json.loads((source / r['original_run'] / 'metrics.json').read_text())
        for k in ('n_modules', 'n_hex_modules', 'frac_grid_analytic'):
            if k in original_metric:
                assert float(r['result_' + k]) == original_metric[k]
    mains = {Path(r['original_run']).name: r for r in numpy}
    assert [int(mains[f'main_sigma1.0_seed{i}']['result_n_hex_modules']) for i in (0, 1)] == [1, 1]
    assert [int(mains[f'main_sigma0.2_seed{i}']['result_n_hex_modules']) for i in (0, 1, 2)] == [0, 0, 0]
    loaded = {}
    for row in rows(evidence / 'numpy_aggregate_results.csv'):
        if row['original_run'] not in loaded:
            loaded[row['original_run']] = json.loads((source / row['original_run']).read_text())
        value = loaded[row['original_run']]
        for key in row['measurement'].split('/'):
            value = value[int(key)] if isinstance(value, list) else value[key]
        # NaN is retained from the saved analysis and is unequal to itself;
        # compare its serialisation rather than treating it as a changed value.
        assert json.dumps(json.loads(row['value']), sort_keys=True) == json.dumps(value, sort_keys=True)
    before = (source / 'review/main.tex').read_text()
    after = (HERE / 'editable_sources/review/main.tex').read_text()
    for pattern in (r'\\section\*?\{([^}]+)\}', r'\\label\{([^}]+)\}', r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}'):
        assert re.findall(pattern, before) == re.findall(pattern, after), pattern
    # Table bodies and captions, figure captions, bibliography and numeric macro
    # sources retain the original numbers. Main prose edits change no numerals.
    assert re.findall(r'\d+(?:\.\d+)?', before) == re.findall(r'\d+(?:\.\d+)?', after), 'Report numerals changed'
    for name in ('numbers_numpy.tex', 'numbers_extra.tex'):
        assert (source / 'review' / name).read_bytes() == (HERE / 'editable_sources/review' / name).read_bytes(), name
    assert '2683459' in after and '2717495' in after
    print(f'PASS: {len(original)} original files unchanged; {len(preserved)} package inputs byte-identical; snapshot complete.')
    print('PASS: evidence values, headline outcomes, report numerals, section order, labels and figures unchanged.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=HERE.parents[1] / 'dist/NeuroProject_combined_corrected')
    parser.add_argument('--package', type=Path, default=HERE / 'NeuroProject_2683459_2717495')
    args = parser.parse_args()
    verify(args.source, args.package)
