import importlib
import unittest


MODULES = [
    "data_loaders.cnn",
    "data_loaders.external_eye",
    "data_loaders.labels",
    "models.model_EfficientNet",
    "models.model_WideResNet",
    "models.model_ViT",
    "models.model_ViT_adv",
    "models.model_ViT_fscl",
    "models.model_ViT_fsclft2",
    "models.model_ViT_fis",
    "networks.efficientnet",
    "networks.mctformer",
    "networks.resnet38d",
    "networks.vision_transformer",
    "tools.effutils",
    "tools.imutils",
    "tools.losses",
    "tools.pyutils",
    "tools.trmutils",
    "tools.utils",
    "analysis.bootstrap_cnn_vit_results",
    "analysis.fairness_metrics",
    "analysis.fairness_strategy_paired_tests_corrected",
    "analysis.metadata_only_baselines",
    "analysis.metadata_vs_image_delta_metrics",
    "analysis.metadata_vs_image_delta_metrics_student_t",
    "analysis.render_final_table_csv",
]


class ImportTests(unittest.TestCase):
    def test_all_modules_import(self):
        for module in MODULES:
            with self.subTest(module=module):
                importlib.import_module(module)


if __name__ == "__main__":
    unittest.main()
