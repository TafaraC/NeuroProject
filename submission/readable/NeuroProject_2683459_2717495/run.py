"""Run checks or reproduce an experiment from this submission folder.

Reads the unchanged implementations under code/. New experiments write to
reproduced_results/; saved evidence is not overwritten by these commands.
Use `python run.py COMMAND --help` for a task's options.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
CODE = ROOT / 'code'


def call(args, folder, env):
    try:
        subprocess.run([sys.executable, *args], cwd=folder, env=env, check=True)
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f'Command failed with exit code {exc.returncode}. Read the error above; check installation using README.txt.') from exc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for name, desc in [('tests', 'Run 20 NumPy and 15 JAX tests.'), ('smoke', 'Run both test suites, short training jobs and the shared-equation check.'), ('equations', 'Compare losses and derivatives on the shared fixture.'), ('compare', 'Re-evaluate five saved NumPy checkpoints with the JAX evaluator.'), ('control', 'Reproduce the constructed-code sampling control.'), ('numbers', 'Regenerate numerical macros from saved results; no training.')]:
        sub.add_parser(name, help=desc, description=desc)
    p = sub.add_parser('numpy', help='Run one named NumPy training job into new results.')
    p.add_argument('--job', default='main_sigma1.0_seed0', help='Job key in code/numpy_impl/experiments/jobs.py')
    p.add_argument('--quick', action='store_true', help='Use the original 1,500-step workflow check, not a full reproduction.')
    p = sub.add_parser('jax', help='Run a JAX reference configuration into new results.')
    p.add_argument('--condition', choices=['baseline', 'fine_resolution'], default='baseline')
    p.add_argument('--steps', type=int, help='Optional shorter workflow check; omit for the full 150,000 steps.')
    p = sub.add_parser('diagnostics', help='Plan, run or report the nine JAX diagnostic experiments.')
    p.add_argument('action', choices=['plan', 'run', 'report'])
    p.add_argument('--workers', type=int, choices=[1, 2], default=1)
    args = parser.parse_args()
    work = ROOT / '.work'
    work.mkdir(exist_ok=True)
    output = ROOT / 'reproduced_results'
    output.mkdir(exist_ok=True)
    env = dict(os.environ, TEMP=str(work), TMP=str(work), MPLCONFIGDIR=str(work / 'matplotlib'), PYTEST_DISABLE_PLUGIN_AUTOLOAD='1')
    npdir, jxdir = CODE / 'numpy_impl', CODE / 'jax_impl'
    command = args.command
    if command == 'tests':
        call(['-m', 'unittest', 'discover', '-s', 'tests', '-v'], npdir, env)
        call(['-m', 'pytest', 'tests', '-q', '-p', 'no:cacheprovider'], jxdir, env)
    elif command == 'smoke':
        call(['smoke_test.py'], CODE, env)
    elif command == 'equations':
        call(['-m', 'comparison.partner_equations', '--output', str(output / 'equations')], jxdir, env)
    elif command == 'compare':
        call(['-m', 'comparison.partner_results', '--snapshot', str(npdir), '--output', str(output / 'common_evaluation')], jxdir, env)
    elif command == 'control':
        call(['-m', 'comparison.constraint_control', '--output', str(output / 'sampling_control')], jxdir, env)
    elif command == 'numbers':
        # The package keeps the original figure/number scripts in their sibling layout.
        call(['make_review_numbers.py', '--regenerate'], CODE / 'review', env)
    elif command == 'numpy':
        env['ACTIONABLE_RESULTS'] = str(output / 'numpy')
        call(['experiments/exp1_grid_emergence.py', '--job', args.job] + (['--quick'] if args.quick else []), npdir, env)
    elif command == 'jax':
        call(['-m', 'experiments.run', '--config', f'experiments/configs/{args.condition}.json', '--output', str(output / ('jax_' + args.condition)), '--evaluate'] + (['--steps', str(args.steps)] if args.steps else []), jxdir, env)
    else:
        call(['-m', 'experiments.diagnostics', args.action, '--output', str(output / 'diagnostics'), '--workers', str(args.workers)], jxdir, env)


if __name__ == '__main__':
    main()
