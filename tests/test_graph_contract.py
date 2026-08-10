from __future__ import annotations

import ast
import unittest
from pathlib import Path


GRAPH_SOURCE = Path("src/longitudinal_dbt/preprocessing/graph_construction.py")


def _function_source(name: str) -> str:
    source = GRAPH_SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    return ast.get_source_segment(source, function) or ""


class GraphContractTests(unittest.TestCase):
    def test_graph_module_compiles_without_importing_gpu_stack(self) -> None:
        compile(GRAPH_SOURCE.read_text(encoding="utf-8"), str(GRAPH_SOURCE), "exec")

    def test_graph_label_contract_is_explicit_in_source(self) -> None:
        function = _function_source("_class_label_from_row")
        self.assertIn("normal=0", function)
        self.assertIn("malignant=1", function)
        self.assertIn("benign=2", function)
        self.assertIn("GraphLabel must be one of 0, 1, or 2", function)


if __name__ == "__main__":
    unittest.main()
