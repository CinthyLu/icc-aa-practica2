import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


class Part3Test(unittest.TestCase):
    def setUp(self):
        notebook = json.loads((ROOT / "practica2_svm.ipynb").read_text(encoding="utf-8"))
        self.cells = {c["id"]: "".join(c["source"]) for c in notebook["cells"]}

    def test_prediction_function_present(self):
        self.assertIn("parte3-funcion", self.cells.keys(), "Missing versioned prediction function")

    def test_examples_and_shap_present(self):
        self.assertIn("parte3-casos", self.cells.keys(), "Missing normal/holiday examples")
        self.assertIn("parte3-shap", self.cells.keys(), "Missing SHAP explanations")

    def test_version_loading_validation_and_shap(self):
        for cell_id in ["parte3-config", "parte3-funcion", "parte3-casos", "parte3-shap"]:
            self.assertIn(cell_id, self.cells.keys(), "Missing executable Part 3")
        if "PARTE3_TEST_DIR" not in os.environ:
            with tempfile.TemporaryDirectory(prefix="parte3-test-") as temp:
                result = subprocess.run(
                    [sys.executable, str(Path(__file__).resolve())],
                    env={**os.environ, "PARTE3_TEST_DIR": temp, "PYTHONIOENCODING": "utf-8",
                         "MPLBACKEND": "Agg", "MLFLOW_DISABLE_TELEMETRY": "true"},
                    capture_output=True, text=True, encoding="utf-8", timeout=240)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            return

        import mlflow
        import mlflow.sklearn
        import numpy as np
        import pandas as pd
        from mlflow.models import infer_signature
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVR

        previous = Path.cwd()
        try:
            os.chdir(os.environ["PARTE3_TEST_DIR"])
            Path("data").mkdir()
            for path in (ROOT / "data").glob("*.csv"):
                shutil.copy2(path, Path("data") / path.name)
            before = {p.name: hashlib.sha256(p.read_bytes()).digest() for p in Path("data").glob("*.csv")}
            scope = {"display": lambda *args: None}
            for cell_id in ["d810f523", "d5b4b76b", "cc273549"]:
                exec(self.cells[cell_id], scope)
            uri = "sqlite:///" + (Path.cwd() / "mlflow.db").as_posix()
            mlflow.set_tracking_uri(uri)
            mlflow.set_registry_uri(uri)
            x = scope["X_tr"].iloc[::30].head(60)
            y = scope["y_tr"].loc[x.index]
            models = []
            for offset in [0, 5]:
                model = make_pipeline(StandardScaler(), SVR(C=10)).fit(x, y + offset)
                models.append(model)
                with mlflow.start_run():
                    info = mlflow.sklearn.log_model(
                        model, name="modelo", signature=infer_signature(x, model.predict(x)))
                    mlflow.register_model(info.model_uri, "svr_demanda")
            Path("resultados").mkdir()
            Path("resultados/parte2_modelo.json").write_text(json.dumps(
                {"nombre": "svr_demanda", "version": 2, "columnas": list(x.columns)}), encoding="utf-8")
            exec(self.cells["parte3-config"], scope)
            exec(self.cells["parte3-funcion"], scope)
            predict = scope["predecir"]
            sample = scope["X_te"].iloc[[0]].copy()
            with patch.object(SVR, "fit", side_effect=AssertionError("Inference retrained SVR")), \
                    patch.object(StandardScaler, "fit", side_effect=AssertionError("Inference refitted scaler")):
                predictions = [predict(sample, version) for version in [1, 2]]
                for i, prediction in enumerate(predictions):
                    self.assertAlmostEqual(prediction, float(models[i].predict(sample)[0]))
                self.assertGreater(abs(predictions[0] - predictions[1]), 1)
                self.assertAlmostEqual(predict(sample[sample.columns[::-1]], 1), predictions[0])
                for version in [0, -1, True, "1", "../1", 1.5]:
                    with self.subTest(version=version), self.assertRaises(ValueError):
                        predict(sample, version)
                bad_samples = [sample.iloc[:0], pd.concat([sample, sample]), sample.to_dict(),
                               sample.drop(columns=sample.columns[0]), sample.assign(extra=1)]
                for value in [np.nan, np.inf, "invalid"]:
                    bad = sample.copy()
                    bad[bad.columns[0]] = value
                    bad_samples.append(bad)
                bad = sample.copy()
                bad.columns = [bad.columns[1], *bad.columns[1:]]
                bad_samples.append(bad)
                for dummy_value in [0, 1, 0.5]:
                    bad = sample.copy()
                    bad.loc[:, bad.columns.str.startswith("prod_")] = dummy_value
                    bad_samples.append(bad)
                for bad in bad_samples:
                    with self.subTest(sample_type=type(bad).__name__), self.assertRaises(ValueError):
                        predict(bad, 1)
                with self.assertRaises(mlflow.exceptions.MlflowException):
                    predict(sample, 999)
                for cell_id in ["parte3-casos", "parte3-shap", "parte3-conclusiones"]:
                    exec(self.cells[cell_id], scope)
            summary = scope["resumen_casos"]
            self.assertEqual(list(summary["caso"]), ["normal", "feriado"])
            self.assertEqual(list(summary["es_feriado"]), [0, 1])
            self.assertEqual(scope["valores_shap"].shape, (2, 22))
            np.testing.assert_allclose(
                scope["base_shap"] + scope["valores_shap"].sum(axis=1),
                summary["prediccion"], rtol=1e-5, atol=1e-5)
            self.assertEqual(len(pd.read_csv("resultados/parte3_predicciones.csv")), 2)
            self.assertEqual(len(pd.read_csv("resultados/parte3_shap.csv")), 44)
            for name in ["06_shap_normal.png", "07_shap_feriado.png"]:
                self.assertTrue((Path("figuras") / name).is_file())
            for p in Path("data").glob("*.csv"):
                self.assertEqual(hashlib.sha256(p.read_bytes()).digest(), before[p.name])
        finally:
            mlflow.end_run()
            mlflow.set_tracking_uri(None)
            mlflow.set_registry_uri(None)
            os.chdir(previous)


if __name__ == "__main__":
    unittest.main()
