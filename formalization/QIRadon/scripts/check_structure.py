#!/usr/bin/env python3
"""Check the active Lean import closure and isolation of archived material."""
from pathlib import Path
import json
import re

ROOT = Path(__file__).resolve().parents[1]


def imports(path):
    return [name for line in re.findall(r"^import\s+([^\n]+)", path.read_text(), re.M)
            for name in line.split("--", 1)[0].split()]


def main():
    config = re.sub(r"#[^\n]*", "", (ROOT / "lakefile.toml").read_text())
    assert re.search(r'defaultTargets\s*=\s*\["QIRadon"\]', config), "Unexpected default target"
    libraries = re.findall(r"\[\[lean_lib\]\]([^\[]*)", config)
    assert len(libraries) == 1 and re.sub(r"\s", "", libraries[0]) == 'name="QIRadon"', \
        "Review changed library roots"
    modules = {".".join(p.relative_to(ROOT).with_suffix("").parts): p
               for p in (ROOT / "QIRadon").rglob("*.lean")}
    modules["QIRadon"] = ROOT / "QIRadon.lean"
    graph = {}
    for name, path in modules.items():
        dependencies = imports(path)
        assert all("archive" not in d.lower().split(".") for d in dependencies), path
        graph[name] = [d for d in dependencies if d == "QIRadon" or d.startswith("QIRadon.")]
        for dependency in graph[name]:
            assert dependency in modules, f"Missing source: {name} imports {dependency}"
    seen, visiting = set(), set()

    def visit(name):
        assert name not in visiting, f"Import cycle at {name}"
        if name in seen:
            return
        visiting.add(name)
        for dependency in graph[name]:
            visit(dependency)
        visiting.remove(name)
        seen.add(name)

    visit("QIRadon")
    assert seen == set(modules), f"Unused active modules: {sorted(set(modules) - seen)}"
    assert imports(ROOT / "Audit.lean") == ["QIRadon"], "Unexpected audit import"
    archive = (ROOT / "archive").resolve()
    assert all(not modules[name].resolve().is_relative_to(archive) for name in seen)
    for path in modules.values():
        source = re.sub(r"/-.*?-/|--[^\n]*", "", path.read_text(), flags=re.S)
        assert not re.search(r"\b(sorry|admit|sorryAx|native_decide|unsafe)\b|^\s*axiom\s",
                             source, re.M), f"Review proof shortcut in {path}"
    result = {"active_modules": len(modules)-1, "entry_imports": len(graph["QIRadon"]),
              "unreachable_modules": [], "missing_source_imports": [],
              "archive_imports": [], "archive_in_build": False}
    (ROOT / "verification" / "structure.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
