"""Zip the readable submission, excluding caches and newly reproduced outputs.

Reads the package folder and writes a sibling ZIP. Editable sources, internal
verification records and build tools are deliberately outside this archive.
"""
from pathlib import Path
import argparse
import zipfile

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=Path, default=HERE / 'NeuroProject_2683459_2717495')
    parser.add_argument('--output', type=Path, default=HERE / 'NeuroProject_2683459_2717495.zip')
    args = parser.parse_args()
    excluded = {'.work', '.venv', '__pycache__', '.pytest_cache', 'reproduced_results'}
    with zipfile.ZipFile(args.output, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in sorted(args.package.rglob('*')):
            rel = p.relative_to(args.package)
            if p.is_file() and not excluded.intersection(rel.parts):
                z.write(p, args.package.name + '/' + rel.as_posix())
    with zipfile.ZipFile(args.output) as z:
        assert z.testzip() is None
        pdf = args.package / 'NeuroProject_Report.pdf'
        assert z.read(args.package.name + '/' + pdf.name) == pdf.read_bytes()
        assert {n.split('/')[0] for n in z.namelist()} == {args.package.name}
    print(f'Verified archive: {args.output.name} ({args.output.stat().st_size:,} bytes)')


if __name__ == '__main__':
    main()
