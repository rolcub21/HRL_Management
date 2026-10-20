import pytest

from prepare_e123_push import collect_sources


def put(root, name, content=''):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def test_absolute_relative_and_deferred_imports(tmp_path):
    put(tmp_path, 'pkg/__init__.py')
    put(tmp_path, 'pkg/run.py', 'from . import helper\nfrom pkg.child import value\n'
        'def later():\n    import other\n')
    put(tmp_path, 'pkg/helper.py')
    put(tmp_path, 'pkg/child.py', 'value = 1')
    put(tmp_path, 'other.py')
    put(tmp_path, 'unrelated.py')
    assert collect_sources(tmp_path, ['pkg/run.py']) == [
        'other.py', 'pkg/__init__.py', 'pkg/child.py', 'pkg/helper.py', 'pkg/run.py']


def test_file_literals_include_source_but_not_result_artifacts(tmp_path):
    put(tmp_path, 'run.py', 'SOURCE = "extra.py"\nOUTPUT = "results/contract.json"\n')
    put(tmp_path, 'extra.py')
    put(tmp_path, 'results/contract.json', '{}')
    assert collect_sources(tmp_path, ['run.py']) == ['extra.py', 'run.py']


def test_non_shipyard_import_rejected(tmp_path):
    put(tmp_path, 'run.py', 'import projects.antmaze\n')
    put(tmp_path, 'projects/antmaze.py')
    with pytest.raises(RuntimeError, match='Non-Shipyard'):
        collect_sources(tmp_path, ['run.py'])
