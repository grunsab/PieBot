import datetime as dt
import hashlib
import importlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        self.fetch = importlib.import_module('training.nnue.fetch_lc0_bins')
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_dated_tar_inventory_includes_boundaries_and_records_size(self):
        names = ['training-run2-test91-20260706-2359.tar', 'training-run2-test91-20260707-0000.tar',
                 'training-run2-test91-20260907-1020.tar', 'training-run2--20260908-0000.tar',
                 'training-run2--20260801-0000.tar', '../escape.tar']
        def listing(url):
            return '\n'.join(f'<a href="{name}">{name}</a> 07-Sep-2026 12:00 {10240 if "20260801" in name else 45678}' for name in names)
        manifest = self.fetch.discover_snapshot(['test91'], self.root, '2026-07-07',
                   '2026-09-07', listing_func=listing,
                   now=dt.datetime(2026,9,7,12,tzinfo=dt.timezone.utc))
        self.assertEqual(3,len(manifest['files']))
        self.assertEqual(101596,manifest['total_bytes'])
        self.assertEqual('2026-09-07T12:00:00+00:00',manifest['until'])
        self.assertEqual('tar_member_mtime',manifest['date_basis'])
        self.assertEqual([],manifest['empty_archives'])
        self.assertIn(10240,[entry['size'] for entry in manifest['files']])

    def test_snapshot_resume_verifies_sha_and_reuses_frozen_inventory(self):
        dest=self.root/'test91'/'a.tar'
        dest.parent.mkdir()
        dest.write_bytes(b'archive')
        entry={'dest':str(dest),'url':'https://example.com/a.tar','size':7,
               'sha256':hashlib.sha256(b'archive').hexdigest(),'status':'downloaded'}
        manifest={'files':[entry],'failures':[],'complete':True}
        path=self.root/'manifest.json'
        self.fetch.write_json_atomic(path,manifest)
        original=path.read_bytes()
        with mock.patch.object(self.fetch,'download_curl') as dl:
            self.assertEqual(0,self.fetch.download_snapshot(path,min_free_bytes=0))
            dl.assert_not_called()
        self.assertEqual(original,path.read_bytes())
        dest.write_bytes(b'corrupt')
        with mock.patch.object(self.fetch,'download_curl',side_effect=OSError('network lost')):
            self.assertEqual(1,self.fetch.download_snapshot(path,min_free_bytes=0))
        loaded=json.loads(path.read_text())
        self.assertTrue(loaded['failures'])
        self.assertIn('error',loaded['files'][0]['status'])
        self.assertFalse(loaded['complete'])

    def test_curl_failure_preserves_existing_and_removes_partial(self):
        dest=self.root/'a.tar'
        dest.write_bytes(b'old')
        def fail(cmd,**kwargs):
            Path(cmd[cmd.index('--output')+1]).write_bytes(b'partial')
            raise subprocess.CalledProcessError(22,cmd)
        with mock.patch.object(self.fetch.subprocess,'run',side_effect=fail):
            with self.assertRaises(subprocess.CalledProcessError):
                self.fetch.download_curl(self.fetch.DownloadJob('https://example.com/a.tar',dest))
        self.assertEqual(b'old',dest.read_bytes())
        self.assertFalse(Path(str(dest)+'.part').exists())

    def test_curl_checksum_mismatch_does_not_publish(self):
        dest=self.root/'a.tar'
        def success(cmd,**kwargs):
            Path(cmd[cmd.index('--output')+1]).write_bytes(b'content')
            return subprocess.CompletedProcess(cmd,0)
        with mock.patch.object(self.fetch.subprocess,'run',side_effect=success):
            with self.assertRaisesRegex(ValueError,'checksum'):
                self.fetch.download_curl(self.fetch.DownloadJob('https://example.com/a.tar',dest),
                                         expected_sha256='0'*64)
        self.assertFalse(dest.exists())


if __name__=='__main__':
    unittest.main()
