"""Tests TaskQC object and Bpod extractors
NB: FPGA TaskQC extractor is tested in test_ephys_extraction_choiceWorld
"""
import unittest
import shutil
from pathlib import Path

import numpy as np

from ibllib.qc.oneutils import download_taskqc_raw_data
from ibllib.qc.task_metrics import TaskQC
from ibllib.qc.task_extractors import TaskQCExtractor
from ibllib.qc.oneutils import download_taskqc_raw_data
from oneibl.one import ONE

one = ONE(
    base_url="https://test.alyx.internationalbrainlab.org",
    username="test_user",
    password="TapetesBloc18",
)


class TestTaskQCObject(unittest.TestCase):
    def setUp(self):
        self.one = one
        # self.eid = 'a3df91c8-52a6-4afa-957b-3479a7d0897c'  # KS022/2019-12-10/001
        self.eid = 'b1c968ad-4874-468d-b2e4-5ffa9b9964e9'
        # Make sure the data exists locally
        download_taskqc_raw_data(self.eid, one=one)
        self.session_path = self.one.path_from_eid(self.eid)
        self.qc = TaskQC(self.eid, one=self.one)
        self.qc.load_data(bpod_only=True)  # Test session has no raw FPGA data

    def test_compute(self):
        # Compute metrics
        self.assertTrue(self.qc.metrics is None)
        self.assertTrue(self.qc.passed is None)
        self.qc.compute()
        self.assertTrue(self.qc.metrics is not None)
        self.assertTrue(self.qc.passed is not None)
        self.assertIn('_task_passed_trial_checks', self.qc.passed, 'failed to add meta checks')

    def test_run(self):
        # Reset Alyx fields before test
        reset = self.qc.update('NOT_SET', override=True)
        assert reset == 'NOT_SET', 'failed to reset QC field for test'
        extended = self.one.alyx.json_field_write('sessions', field_name='extended_qc',
                                                  uuid=self.eid, data={})
        assert not extended, 'failed to reset extended QC field for test'

        # Test update as False
        outcome, _ = self.qc.run(update=False)
        self.assertEqual('FAIL', outcome)
        extended = self.one.alyx.rest('sessions', 'read', id=self.eid)['extended_qc']
        self.assertDictEqual({}, extended, 'unexpected update to extended qc')
        outcome = self.one.alyx.rest('sessions', 'read', id=self.eid)['qc']
        self.assertEqual('NOT_SET', outcome, 'unexpected update to qc')

        # Test update as True
        outcome, results = self.qc.run(update=True)
        self.assertEqual('FAIL', outcome)
        extended = self.one.alyx.rest('sessions', 'read', id=self.eid)['extended_qc']
        expected = list(results.keys()) + ['task']
        self.assertCountEqual(expected, extended.keys(), 'unexpected update to extended qc')
        qc_field = self.one.alyx.rest('sessions', 'read', id=self.eid)['qc']
        self.assertEqual(outcome, qc_field, 'unexpected update to qc')

    def test_compute_session_status(self):
        with self.assertRaises(AttributeError):
            self.qc.compute_session_status()
        self.qc.compute()
        outcome, results, outcomes = self.qc.compute_session_status()
        self.assertEqual('FAIL', outcome)

        # Check each outcome matches...
        # NOT_SET
        not_set = [k for k, v in results.items() if np.isnan(v)]
        self.assertTrue(all(outcomes[k] == 'NOT_SET' for k in not_set))
        # PASS
        passed = [k for k, v in results.items() if v >= self.qc.criteria['PASS']]
        self.assertTrue(all(outcomes[k] == 'PASS' for k in passed))
        # WARNING
        wrn = [k for k, v in results.items()
               if self.qc.criteria['WARNING'] <= v <= self.qc.criteria['PASS']]
        self.assertTrue(all(outcomes[k] == 'WARNING' for k in wrn))
        # FAIL
        fail = [k for k, v in results.items() if v <= self.qc.criteria['FAIL']]
        self.assertTrue(all(outcomes[k] == 'FAIL' for k in fail))


class TestBpodQCExtractors(unittest.TestCase):

    def setUp(self):
        self.one = one
        self.eid = 'b1c968ad-4874-468d-b2e4-5ffa9b9964e9'
        self.eid_incomplete = '4e0b3320-47b7-416e-b842-c34dc9004cf8'  # Missing required datasets
        # Make sure the data exists locally
        download_taskqc_raw_data(self.eid, one=one, fpga=True)
        self.session_path = self.one.path_from_eid(self.eid)

    def test_lazy_extract(self):
        ex = TaskQCExtractor(self.session_path, lazy=True, one=self.one)
        self.assertIsNone(ex.data)

    def test_extraction(self):
        ex = TaskQCExtractor(self.session_path, lazy=True, one=self.one, bpod_only=True)
        self.assertIsNone(ex.raw_data)

        # Test loading raw data
        ex.load_raw_data()
        self.assertIsNotNone(ex.raw_data, 'Failed to load raw data')
        self.assertIsNotNone(ex.settings, 'Failed to load task settings')
        self.assertIsNotNone(ex.frame_ttls, 'Failed to load BNC1')
        self.assertIsNotNone(ex.audio_ttls, 'Failed to load BNC2')

        # Test extraction
        ex.extract_data()
        expected = [
            'stimOnTrigger_times', 'stimOffTrigger_times', 'stimOn_times', 'stimOff_times',
            'stimFreeze_times', 'stimFreezeTrigger_times', 'errorCueTrigger_times', 'itiIn_times',
            'choice', 'feedbackType', 'goCueTrigger_times', 'wheel_timestamps', 'wheel_position',
            'wheel_moves_intervals', 'wheel_moves_peak_amplitude', 'firstMovement_times', 'phase',
            'goCue_times', 'rewardVolume', 'response_times', 'feedback_times', 'probabilityLeft',
            'position', 'contrast', 'quiescence', 'contrastRight', 'contrastLeft', 'outcome',
            'errorCue_times', 'valveOpen_times', 'correct', 'intervals'
        ]
        self.assertCountEqual(expected, ex.data.keys())
        self.assertEqual('ephys', ex.type)
        self.assertEqual('X1', ex.wheel_encoding)

    def test_partial_extraction(self):
        ex = TaskQCExtractor(self.session_path, lazy=True, one=self.one, bpod_only=True)
        ex.extract_data(partial=True)
        expected = [
            'stimOnTrigger_times', 'stimOffTrigger_times', 'stimOn_times', 'stimOff_times',
            'stimFreeze_times', 'stimFreezeTrigger_times', 'errorCueTrigger_times', 'itiIn_times',
            'position', 'contrast', 'quiescence', 'phase', 'probabilityLeft', 'contrastRight',
            'contrastLeft'
        ]
        self.assertCountEqual(expected, ex.data.keys())

    def test_download_data(self):
        """Test behavior when download_data flag is True
        """
        # path = self.one.path_from_eid(self.eid_incomplete)  # FIXME Returns None
        det = one.get_details(self.eid_incomplete)
        path = (Path(one._par.CACHE_DIR) / det['lab'] / 'Subjects' /
                det['subject'] / det['start_time'][:10] / str(det['number']))
        ex = TaskQCExtractor(path, lazy=True, one=self.one, download_data=True)
        self.assertTrue(ex.lazy, 'Failed to set lazy flag')

        shutil.rmtree(self.session_path)  # Remove downloaded data
        assert self.session_path.exists() is False, 'Failed to remove session folder'
        TaskQCExtractor(self.session_path, lazy=True, one=self.one, download_data=True)
        files = list(self.session_path.rglob('*.*'))
        expected = 6  # NB This session is missing raw ephys data and missing some datasets
        self.assertEqual(len(files), expected)


if __name__ == "__main__":
    unittest.main(exit=False)
