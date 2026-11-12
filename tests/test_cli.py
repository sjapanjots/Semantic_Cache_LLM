import json

import pytest

from semcache import SemanticCache
from semcache.cli import main


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "cache.db"
    with SemanticCache(path=path, pricing={"m": (1.0, 1.0)}) as cache:
        cache.set("What is the capital of France?", "Paris", model="m", input_tokens=10**6)
        cache.set("Who wrote Hamlet?", "Shakespeare", namespace="lit")
        cache.get("what is the capital of france", model="m")
        cache.get("something nobody asked")
    return str(path)


def test_stats_text(db, capsys):
    assert main(["stats", "--db", db]) == 0
    out = capsys.readouterr().out
    assert "entries" in out and "50.0%" in out and "$1.0000" in out and "by model" in out


def test_stats_json(db, capsys):
    main(["stats", "--db", db, "--json"])
    data = json.loads(capsys.readouterr().out)
    assert data["entries"] == 2 and data["hits"] == 1 and data["cost_saved"] == 1.0


def test_list(db, capsys):
    main(["list", "--db", db])
    out = capsys.readouterr().out.splitlines()
    assert "capital of France" in out[0] and "1 hits" in out[0]


def test_clear_namespace_and_all(db, capsys):
    main(["clear", "--db", db, "--namespace", "lit"])
    assert "removed 1" in capsys.readouterr().out
    main(["clear", "--db", db, "--stats"])
    assert "removed 1" in capsys.readouterr().out
    main(["stats", "--db", db, "--json"])
    data = json.loads(capsys.readouterr().out)
    assert data["entries"] == 0 and data["lookups"] == 0


def test_prune(db, capsys):
    main(["prune", "--db", db, "--older-than", "3600"])
    assert "removed 0" in capsys.readouterr().out
    main(["prune", "--db", db, "--older-than", "0"])
    assert "removed 2" in capsys.readouterr().out


def test_prices(capsys):
    main(["prices"])
    assert "claude-opus-5-5" in capsys.readouterr().out


def test_missing_file(tmp_path):
    with pytest.raises(SystemExit, match="no cache file"):
        main(["stats", "--db", str(tmp_path / "nope.db")])


def test_clear_namespace_includes_wrapper_entries(tmp_path, capsys):
    path = tmp_path / "w.db"
    with SemanticCache(path=path) as cache:
        cache.set("a", "1", namespace="openai/abc123")
        cache.set("b", "2")
    main(["clear", "--db", str(path), "--namespace", "openai"])
    assert "removed 1" in capsys.readouterr().out
