"""Prepare a code-only Git path list for E123 v2; never stage or commit files."""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
CONTRACT = ROOT / 'results/vcg-conditioned-e123-absolute-progress-8x8-187k-v2/contract.json'
EXCLUDED = {'projects', 'results', 'runs', '.git', '.venv', '__pycache__'}


def collect_sources(root: Path, seeds: list[str]) -> list[str]:
    """Follow local absolute/relative imports, package initializers and file literals."""
    pending = [root / name for name in seeds]
    found: set[Path] = set()

    def module(name: str) -> None:
        if not name:
            return
        path = root.joinpath(*name.split('.'))
        if path.with_suffix('.py').is_file():
            pending.append(path.with_suffix('.py'))
        elif (path / '__init__.py').is_file():
            pending.append(path / '__init__.py')

    while pending:
        path = pending.pop()
        rel = path.relative_to(root)
        if set(rel.parts) & EXCLUDED:
            raise RuntimeError(f'Non-Shipyard dependency needs manual review: {rel}')
        if path in found:
            continue
        if not path.is_file():
            raise FileNotFoundError(path)
        found.add(path)
        for parent in path.parents:
            if parent == root:
                break
            init = parent / '__init__.py'
            if init.is_file() and init not in found:
                pending.append(init)
        if path.suffix != '.py':
            continue
        tree = ast.parse(path.read_text(), filename=str(rel))
        package = list(rel.with_suffix('').parts[:-1])
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module(alias.name)
            elif isinstance(node, ast.ImportFrom):
                prefix = package[:len(package)-node.level+1] if node.level else []
                base = '.'.join(prefix + ([node.module] if node.module else []))
                module(base)
                for alias in node.names:
                    if alias.name != '*':
                        module('.'.join(filter(None, (base, alias.name))))
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                value = node.value
                if '\n' not in value and len(value) < 240:
                    if value.endswith(('.py', '.sh', '.json')) and not Path(value).is_absolute():
                        candidate = root / value
                        if candidate.is_file() and not set(candidate.relative_to(root).parts) & EXCLUDED:
                            pending.append(candidate)
                    if value.startswith(('experiments.', 'methods.', 'PSLAP.', 'example.')):
                        module(value)
    # Document the selected experiment modules, without sweeping sibling experiments.
    for directory in {p.parent for p in found if 'experiments' in p.parts}:
        for name in ('README.md', 'README.txt'):
            path = directory / name
            if path.is_file():
                found.add(path)
    return sorted(str(path.relative_to(root)) for path in found)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=Path('/tmp/shipyard-e123-push.paths'))
    args = parser.parse_args()
    contract = json.loads(CONTRACT.read_text())
    seeds = list(contract['source_sha256'])
    seeds += ['prepare_e123_push.py', 'tests/test_prepare_e123_push.py',
              'requirements.txt', 'pyproject.toml']
    paths = collect_sources(ROOT, seeds)
    # Existing tracked dependencies are already available from Git. Do not scoop
    # up unrelated local edits merely because those modules are imported.
    modified = set(subprocess.check_output(
        ['git', 'diff', '--name-only', 'HEAD'], cwd=ROOT, text=True).splitlines())
    unrelated = sorted((set(paths) & modified) - set(contract['source_sha256']))
    paths = [path for path in paths if path not in unrelated]
    ignored = subprocess.run(['git', 'check-ignore', '--stdin'], cwd=ROOT,
        input='\n'.join(paths)+'\n', text=True, capture_output=True)
    if ignored.returncode not in (0, 1):
        raise RuntimeError(ignored.stderr)
    if ignored.stdout.strip():
        raise RuntimeError(f'Manifest contains ignored paths: {ignored.stdout}')
    args.manifest.write_text('\n'.join(paths)+'\n')
    changed = subprocess.check_output(['git', 'status', '--short', '--untracked-files=all',
                                      '--', *paths], cwd=ROOT, text=True)
    print(f'Contract: {contract["contract_sha256"]}\nCode/documentation paths: {len(paths)}')
    if unrelated:
        print('Preserving unrelated tracked edits (not included): ' + ', '.join(unrelated))
    print(f'Manifest: {args.manifest}\nNot staged or committed. Selected changes:\n{changed}')


if __name__ == '__main__':
    main()
