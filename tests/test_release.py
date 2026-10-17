"""CPU-safe release contracts; these tests do not claim CUDA inference parity."""
import ast
import csv
import importlib.util
import json
import re
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'tools')]
from dualtrace.cli import read_rows
from restore_assets import safe_relative
from export_source import public_files, audit


class ReleaseTests(unittest.TestCase):
    def test_tar_traversal_rejected(self):
        with self.assertRaises(ValueError):
            safe_relative('opt/v178/assets/../../secret', 'opt/v178/assets/')

    def test_tar_scope(self):
        self.assertIsNone(safe_relative('etc/passwd', 'opt/v178/assets/'))
        self.assertEqual(safe_relative('./opt/v178/assets/heads/a.pth', 'opt/v178/assets/'), Path('heads/a.pth'))

    def test_csv_keeps_ids_and_empty_text(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'text.csv'
            p.write_text('id,text\n001,\n002,example\n')
            rows = read_rows(p, 'text')
            self.assertEqual(rows[0], {'id': '001', 'text': ''})

    def test_duplicate_ids_fail(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'text.csv'
            p.write_text('id,text\n001,a\n001,b\n')
            with self.assertRaises(ValueError):
                read_rows(p, 'text')

    def test_python_syntax(self):
        for p in public_files():
            if p.suffix == '.py':
                with self.subTest(path=p.relative_to(ROOT)):
                    ast.parse(p.read_text(), filename=str(p))

    def test_public_no_private_files(self):
        for p in public_files():
            rel = p.relative_to(ROOT)
            self.assertNotIn('.local', rel.parts)
            self.assertNotIn('outputs', rel.parts)
            self.assertNotIn(p.suffix, {'.pt', '.pth', '.safetensors', '.joblib', '.tar'})

    def test_public_secret_scan(self):
        self.assertEqual(audit(public_files()), [])

    def test_public_documentation(self):
        self.assertIn('面向 AI', (ROOT / 'README.md').read_text())
        self.assertTrue((ROOT / 'README.en.md').is_file())
        self.assertEqual([p.name for p in (ROOT / 'docs').glob('*.md')], ['method.md'])
        for path in [ROOT / 'README.md', ROOT / 'README.en.md', ROOT / 'docs/method.md']:
            content = path.read_text()
            self.assertIsNone(re.search(r'\bT?V\d+\b', content), path)
            self.assertNotIn('\\operatorname', content, path)
            self.assertEqual(content.count('```') % 2, 0, path)
            for target in re.findall(r'\]\(([^)]+)\)', content):
                if not target.startswith(('https://', 'http://', '#')):
                    self.assertTrue((path.parent / target.split('#')[0]).exists(), (path, target))

    def test_image_plan_sources_exist_and_flags_match(self):
        p = ROOT / 'training/image/train.py'
        spec = importlib.util.spec_from_file_location('image_plan', p)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for stage in module.plan(ROOT / 'research/image', ROOT / 'runs/test_plan'):
            command = stage['command']
            source = Path(command[1])
            self.assertTrue(source.is_file(), source)
            tree = ast.parse(source.read_text())
            options = {v.value for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'add_argument' for v in node.args if isinstance(v, ast.Constant) and isinstance(v.value, str)}
            for arg in command[2:]:
                if arg.startswith('--'):
                    self.assertIn(arg, options, (stage['name'], arg))
            if '--config' in command:
                self.assertTrue(Path(command[command.index('--config') + 1]).is_file())

    def test_resume_rejects_changed_recipe(self):
        spec = importlib.util.spec_from_file_location('image_resume', ROOT / 'training/image/train.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            stage = {'name': 'example', 'command': ['python', 'train.py'], 'outputs': ['model']}
            marker = work / 'done.json'
            (work / 'model').touch()
            marker.write_text(json.dumps(dict(stage, source_fingerprint='original')))
            self.assertTrue(module.can_resume(marker, stage, 'original', work))
            with self.assertRaises(RuntimeError):
                module.can_resume(marker, stage, 'changed', work)
            (work / 'model').unlink()
            with self.assertRaises(RuntimeError):
                module.can_resume(marker, stage, 'original', work)


if __name__ == '__main__':
    unittest.main()
