"""Run with python3 -m unittest discover -s bench -p 'test_*.py'."""
import unittest
import report


def cell(engine, source='kodim01', repeat=0, budget='steps-800', psnr=20, ms=1000):
    return dict(_run='run', suite='kodak/' + source, engine=engine, loss='l2',
                source_id=source, target_label=source, protocol='equal-step',
                n=512, opt_width=768, opt_height=512, budget=budget, repeat=repeat,
                seed=7, run_id=f'{engine}-{source}-{repeat}', psnr_db=psnr,
                optimize_ms=ms, steps_completed=800)


class ReportTests(unittest.TestCase):
    def test_parity_uses_scored_final_quality_and_only_step_budgets(self):
        w, d = cell('windfoil', psnr=22), cell('diffvg', psnr=21, ms=10000)
        traces = {('run', w['run_id']): [(1, 5, 20), (2, 10, 21), (3, 15, 22)]}
        row = report.comparisons([w, d], traces)[0]
        self.assertEqual(row['parity_ms'], 10)
        self.assertEqual(row['parity_speedup'], 1000)
        self.assertEqual(row['delta_db'], 1)
        d['psnr_db'] = 23
        self.assertIsNone(report.comparisons([w, d], traces)[0]['parity_speedup'])
        w['budget'] = d['budget'] = 'seconds-60'
        d['psnr_db'] = 21
        self.assertIsNone(report.comparisons([w, d], traces)[0]['parity_speedup'])

    def test_corpus_mean_median_coverage_and_portrait_grouping(self):
        cells, traces = [], {}
        for i, ratio in enumerate((2, 4, None)):
            source = f'kodim{i+1:02}'
            w = cell('windfoil', source, psnr=22, ms=1000)
            d = cell('diffvg', source, psnr=20 if ratio else 23, ms=10000)
            if i == 1:
                w['opt_width'], w['opt_height'] = 512, 768
                d['opt_width'], d['opt_height'] = 512, 768
            cells += [w, d]
            traces[('run', w['run_id'])] = [(1, 10000 / ratio if ratio else 1000, 22)]
        rows = report.summary(cells, report.comparisons(cells, traces))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['diffvg_parity_speedup'], 3)
        self.assertEqual(rows[0]['diffvg_parity_reached'], 2)
        self.assertEqual(rows[0]['diffvg_parity_total'], 3)
        self.assertEqual(rows[0]['windfoil_targets'], 3)
        self.assertEqual(rows[0]['diffvg_psnr_db'], 21)
        self.assertEqual(rows[0]['windfoil_optimize_s'], 1)

    def test_duplicate_cells_are_rejected(self):
        w = cell('windfoil')
        with self.assertRaisesRegex(ValueError, 'duplicate engine'):
            report.comparisons([w, w], {})


if __name__ == '__main__':
    unittest.main()
