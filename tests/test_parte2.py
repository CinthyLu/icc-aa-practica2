from contextlib import nullcontext
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class Part2Test(unittest.TestCase):
    def test_markdown_after_styled_headings_is_rendered(self):
        from nbconvert.filters import markdown2html

        class Text(HTMLParser):
            def __init__(self):
                super().__init__()
                self.text = []
                self.list_items = 0

            def handle_data(self, data):
                self.text.append(data)

            def handle_starttag(self, tag, attrs):
                if tag == "li":
                    self.list_items += 1

        notebook = json.loads((ROOT / "practica2_svm.ipynb").read_text(encoding="utf-8"))
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] != "markdown":
                continue
            with self.subTest(cell=index, cell_id=cell.get("id")):
                source = "".join(cell["source"])
                rendered = Text()
                rendered.feed(markdown2html(source))
                visible = "".join(rendered.text)
                self.assertNotIn("`", visible, "Inline code was left as raw Markdown")
                self.assertNotIn("*", visible, "Emphasis was left as raw Markdown")
                if cell.get("id") == "e8e9ce55":
                    expected = sum(bool(re.match(r"^(?:\d+\.|-)\s", line)) for line in source.splitlines())
                    self.assertEqual(rendered.list_items, expected, "Conclusions and references must be lists")

    def test_search_tracking_and_registered_pipeline(self):
        notebook = json.loads((ROOT / "practica2_svm.ipynb").read_text(encoding="utf-8"))
        cells, inside = [], False
        for cell in notebook["cells"]:
            source = "".join(cell["source"])
            if cell["cell_type"] == "markdown" and (
                    source.startswith("## Parte 2:") or 'id="parte2"' in source):
                inside = True
            elif cell["cell_type"] == "markdown" and (
                    source.startswith("## Conclusiones") or 'id="conclusiones-parte1"' in source
                    or 'id="parte3"' in source):
                inside = False
            elif inside and cell["cell_type"] == "code":
                cells.append(source)
        self.assertGreaterEqual(len(cells), 4, "Missing executable Part 2")

        if "PARTE2_TEST_DIR" not in os.environ:
            # SQLite connections must close before Windows can remove the test directory.
            with tempfile.TemporaryDirectory(prefix="parte2-test-") as temp:
                result = subprocess.run(
                    [sys.executable, str(Path(__file__).resolve())],
                    env={**os.environ, "PARTE2_TEST_DIR": temp, "PYTHONIOENCODING": "utf-8"},
                    capture_output=True, text=True, encoding="utf-8", timeout=180)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            return

        os.environ["MPLBACKEND"] = "Agg"
        import mlflow
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from sklearn.metrics import mean_squared_error
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVR

        features = pd.DataFrame({"timestamp": np.arange(60), "venta_ayer": np.sin(np.arange(60))})
        target = 10 + 0.1 * features["timestamp"] + features["venta_ayer"]
        scope = dict(np=np, pd=pd, plt=plt, Path=Path, make_pipeline=make_pipeline,
                     StandardScaler=StandardScaler, SVR=SVR,
                     rmse=lambda y, p: np.sqrt(mean_squared_error(y, p)),
                     display=lambda *args: None,
                     X_tr=features.iloc[:40], y_tr=target.iloc[:40],
                     X_va=features.iloc[40:50], y_va=target.iloc[40:50],
                     X_te=features.iloc[50:], y_te=target.iloc[50:])
        previous = Path.cwd()
        with nullcontext(os.environ["PARTE2_TEST_DIR"]) as temp:
            try:
                os.chdir(temp)
                scope["FIG"] = Path("figuras")
                scope["FIG"].mkdir()
                exec(cells[0], scope)
                train, validation = next(scope["division"].split())
                np.testing.assert_array_equal(train, np.arange(40))
                np.testing.assert_array_equal(validation, np.arange(40, 50))
                space = [{"svr__kernel": ["linear", "rbf"], "svr__C": [1, 10],
                          "svr__gamma": ["scale"], "svr__epsilon": [0.1]}]
                scope["busquedas"]["GridSearch"].set_params(param_grid=space)
                scope["busquedas"]["RandomSearch"].set_params(param_distributions=space, n_iter=3)
                for source in cells[1:]:
                    exec(source, scope)
                self.assertEqual(len(scope["resultados"]), 7)
                self.assertTrue(np.isfinite(scope["resultados"][["rmse_train", "rmse_val"]]).all().all())
                self.assertTrue((scope["resumen_busquedas"]["tiempo_s"] > 0).all())
                client = scope["cliente"]
                runs = client.search_runs([scope["experimento_id"]], max_results=100)
                children = [run for run in runs if "mlflow.parentRunId" in run.data.tags]
                self.assertEqual(len(children), 7)
                for run in children:
                    self.assertTrue({"kernel", "C", "gamma", "epsilon"} <= run.data.params.keys())
                    self.assertTrue({"rmse_train", "rmse_val"} <= run.data.metrics.keys())
                    self.assertIn("rmse.png", [item.path for item in client.list_artifacts(run.info.run_id)])
                uri = scope["modelo_uri"]
                self.assertTrue(uri.endswith("/1"))
                loaded = mlflow.sklearn.load_model(uri)
                expected = scope["mejor_modelo"].predict(scope["X_te"])
                np.testing.assert_allclose(loaded.predict(scope["X_te"]), expected)
                self.assertEqual(list(loaded.feature_names_in_), list(features.columns))
                self.assertTrue(Path("figuras/05_grid_vs_random.png").is_file())
                self.assertTrue(Path("resultados/parte2_experimentos.csv").is_file())
                self.assertTrue(Path("resultados/parte2_modelo.json").is_file())
                self.assertNotIn("predecir", scope)
            finally:
                mlflow.end_run()
                mlflow.set_tracking_uri(None)
                mlflow.set_registry_uri(None)
                plt.close("all")
                os.chdir(previous)


if __name__ == "__main__":
    unittest.main()
