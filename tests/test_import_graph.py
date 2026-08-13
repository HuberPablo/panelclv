"""The subpackage import graph is acyclic.

`panelclv` is split by altitude: each subpackage names the ones below it and none
of the ones above. Two cycles had formed against that rule, and neither was
visible from anything the suite ran — one of them survived only because the
upward import sat inside a function body, where it neither breaks at load time
nor shows up in an import list.

This asserts the graph itself. Every `panelclv.<sub>` import is counted,
**including deferred ones inside function bodies**: a lazy import is still a
dependency, and hiding one is exactly how the last cycle stayed alive.
"""

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "panelclv"


def _subpackage_graph() -> dict[str, set[str]]:
    """{subpackage: subpackages it imports}, over every module under `src/panelclv/*/`."""
    graph: dict[str, set[str]] = {}
    for py in sorted(SRC.rglob("*.py")):
        sub = py.relative_to(SRC).parts[0]
        if not (SRC / sub).is_dir():
            continue                       # `panelclv/__init__.py` itself
        imports = graph.setdefault(sub, set())
        for node in ast.walk(ast.parse(py.read_text())):
            # `from panelclv.x.y import z` and `import panelclv.x.y` both name the
            # subpackage at position 1 of the dotted path.
            if isinstance(node, ast.ImportFrom):
                names = [node.module] if node.module else []
            elif isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            else:
                continue
            for name in names:
                parts = name.split(".")
                if parts[0] == "panelclv" and len(parts) > 1 and parts[1] != sub:
                    imports.add(parts[1])
    return graph


def _cycles(graph: dict[str, set[str]]) -> set[frozenset[str]]:
    """Every set of subpackages that can reach each other — a cycle of any length.

    Pairwise `a imports b, b imports a` is the shape both known cycles had, but a
    three-hop `a -> b -> c -> a` is the same defect and would slip past a pair
    check, so this walks the graph: two subpackages are in one cycle when each
    can reach the other.
    """
    def reachable(start: str) -> set[str]:
        seen: set[str] = set()
        stack = list(graph.get(start, ()))
        while stack:
            node = stack.pop()
            if node in seen or node not in graph:
                continue
            seen.add(node)
            stack.extend(graph[node])
        return seen

    reach = {sub: reachable(sub) for sub in graph}
    return {
        frozenset({a, b})
        for a in graph for b in reach[a]
        if a != b and a in reach[b]
    }


def test_subpackage_imports_are_acyclic():
    graph = _subpackage_graph()
    assert graph, f"no subpackages found under {SRC}"
    assert _cycles(graph) == set()
