"""Focused arithmetic checks plus independent retained-artifact verification."""
import csv
import json
import math
import statistics
import unittest
from pathlib import Path

from analyze import HERE, affine, digest, p90, sweep


class AnalysisChecks(unittest.TestCase):
    def row(self, start, end, lane=0):
        return dict(start_ts_s=start, end_ts_s=end, target_replica_id=1, target_replica_local_id=lane)

    def test_window_end_precedes_equal_time_start(self):
        self.assertEqual(sweep([self.row(0, 1), self.row(1, 2)])['peak_requests'], 1)

    def test_overlapping_lanes_count_separately(self):
        s = sweep([self.row(0, 2), self.row(1, 3, lane=1)])
        self.assertEqual((s['peak_requests'], s['peak_active_lanes']), (2, 2))

    def test_same_lane_parallel_reads(self):
        s = sweep([self.row(0, 2), self.row(1, 3)])
        self.assertEqual((s['peak_requests'], s['peak_active_lanes']), (2, 1))

    def test_nonpositive_window_rejected(self):
        with self.assertRaises(ValueError):
            sweep([self.row(1, 1)])

    def test_producer_p90_convention(self):
        self.assertEqual(p90(list(range(30))), 27)
        self.assertEqual(p90([1]), 1)

    def test_physical_affine_has_positive_intercept(self):
        m = affine([1, 2, 3], [3, 5, 7])
        self.assertAlmostEqual(m['intercept_ms'], 1)
        self.assertAlmostEqual(m['slope_ms_per_MB'], 2)

    def test_negative_intercept_rejected_without_clamp(self):
        with self.assertRaises(ValueError):
            affine([1, 2, 3], [1, 3, 5])

    def test_negative_slope_rejected(self):
        with self.assertRaises(ValueError):
            affine([1, 2, 3], [5, 4, 3])

    def test_all_artifact_rows_present_and_failed_rules_preserved(self):
        d = json.loads((HERE / 'results.json').read_text())
        rr = d['candidates']['all_case_results']
        self.assertEqual(len({(r['case'], r['pass_index']) for r in rr}), 90)
        self.assertEqual(sum(r['affine_median_rule_pass'] for r in rr), 90)
        self.assertEqual(sum(r['point_median_rule_pass'] for r in rr), 90)
        self.assertEqual(sum(r['affine_trial_rule_pass'] for r in rr), 82)
        self.assertEqual(sum(r['point_trial_rule_pass'] for r in rr), 83)
        self.assertEqual(sum(r['bound32_tail_pass'] for r in rr), 79)
        self.assertEqual(sum(r['bound64_tail_pass'] for r in rr), 86)
        self.assertAlmostEqual(max(r['critical_max_ms'] for r in rr), 27.49703824520111)
        self.assertEqual(len(list(csv.DictReader((HERE / 'reader_summary.csv').open()))), 405)
        trials = list(csv.DictReader((HERE / 'scratch/reader_trials.csv').open()))
        self.assertEqual(len(trials), 12150)
        self.assertIn('complete_ms', trials[0])
        self.assertNotIn('complete_us', trials[0])
        self.assertAlmostEqual(max(float(t['complete_ms']) for t in trials), 27.49703824520111)

    def test_fit_independent_closed_form_and_every_holdout_pass(self):
        rr = json.loads((HERE / 'results.json').read_text())['candidates']
        for excluded in (None, 0, 1, 2):
            rows = [r for r in rr['all_case_results'] if r['pass_index'] != excluded]
            xs = [r['bytes_per_reader'] / 1e6 for r in rows]
            ys = [r['critical_median_ms'] for r in rows]
            xm, ym = statistics.mean(xs), statistics.mean(ys)
            b = sum((x - xm) * (y - ym) for x, y in zip(xs, ys)) / sum((x - xm) ** 2 for x in xs)
            a = ym - b * xm
            model = rr['model'] if excluded is None else rr['leave_one_pass_out_models'][str(excluded)]
            self.assertTrue(a > 0 and b > 0)
            self.assertAlmostEqual(a, model['intercept_ms'], places=12)
            self.assertAlmostEqual(b, model['slope_ms_per_MB'], places=12)

    def test_window_proof_and_empirical_bound(self):
        d = json.loads((HERE / 'results.json').read_text())
        for w in d['replay_windows']:
            self.assertEqual(w['full_transfer_occupancy']['peak_requests'], 1)
            self.assertEqual(w['max_simultaneously_reading_pairs'], 8)
            self.assertEqual(w['zero_duration_decode_forwards'], 0)
            self.assertLess(w['observed_p2_tail_max_ms'], w['min_subsequent_step_gap_ms'])
            self.assertLess(w['all_n1_control_tail_max_ms'], w['min_subsequent_step_gap_ms'])
            self.assertLess(w['proposed_price_ms'], w['min_subsequent_step_gap_ms'])
        self.assertEqual(len(list(csv.DictReader((HERE / 'transfer_windows.csv').open()))), 512)
        proof = list(csv.DictReader((HERE / 'possible_post_windows.csv').open()))
        self.assertEqual(len(proof), 2034)
        self.assertTrue(all(r['deterministic_invariance'] == 'True' and r['empirical_p2_tail_pass'] == 'True' for r in proof))

    def test_retained_input_hashes_unchanged(self):
        d = json.loads((HERE / 'results.json').read_text())
        for filename, entry in d['evidence_hashes'].items():
            with self.subTest(path=filename):
                self.assertEqual(digest(filename), entry['sha256'])
                self.assertEqual(Path(filename).stat().st_size, entry['bytes'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
