import unittest

import numpy as np

from analysis.bootstrap_cnn_vit_results import compute_es_auc_from_global
from analysis.metadata_vs_image_delta_metrics import paired_delong_delta_ci_p_value
from analysis.metadata_vs_image_delta_metrics_student_t import t_ci_p
from analysis.render_final_table_csv import row_experiment


class StatisticalContractTests(unittest.TestCase):
    def test_es_auc_uses_proportion_scale_without_extra_factor(self):
        overall_auc = 0.835
        subgroup_aucs = [0.410, 0.995, 0.842, 0.751]
        expected = overall_auc / (
            1.0 + sum(abs(value - overall_auc) for value in subgroup_aucs)
        )
        self.assertAlmostEqual(
            compute_es_auc_from_global(overall_auc, subgroup_aucs), expected
        )

    def test_delong_identical_predictions_have_zero_delta(self):
        y_true = np.asarray([0, 0, 0, 1, 1, 1], dtype=int)
        scores = np.asarray([0.1, 0.2, 0.4, 0.6, 0.8, 0.9], dtype=float)
        delta, low, high, p_value = paired_delong_delta_ci_p_value(
            y_true, scores, scores
        )
        self.assertEqual(delta, 0.0)
        self.assertLessEqual(low, 0.0)
        self.assertGreaterEqual(high, 0.0)
        self.assertEqual(p_value, 1.0)

    def test_student_t_interval_and_p_value_are_coherent(self):
        samples = np.asarray([-0.04, -0.02, 0.0, 0.02, 0.04])
        low, high, p_value = t_ci_p(0.0, samples)
        self.assertLessEqual(low, 0.0)
        self.assertGreaterEqual(high, 0.0)
        self.assertGreaterEqual(p_value, 0.05)

    def test_fairness_table_labels_map_to_prediction_experiments(self):
        self.assertEqual(
            row_experiment("DR/Race", "ViT + Adv"), "trm_dr_ADV_race"
        )
        self.assertEqual(
            row_experiment("AMD/Ethnic", "ViT+FSCL"),
            "trm_amdbin_FSCL_ethnic_finetune",
        )


if __name__ == "__main__":
    unittest.main()
