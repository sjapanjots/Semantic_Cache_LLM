"""``semcache`` command-line tool for inspecting a persistent cache file."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from . import __version__
from .pricing import PriceBook
from .sqlite_store import SQLiteStore
from .stats import CacheStats


def _open(path: str) -> SQLiteStore:
    if not Path(path).exists():
        raise SystemExit(f"semcache: no cache file at {path}")
    return SQLiteStore(path)


def _cmd_stats(args: argparse.Namespace) -> int:
    store = _open(args.db)
    stats = CacheStats.from_dict(store.load_stats() or {})
    if args.json:
        data = stats.to_dict()
        data["entries"] = len(store)
        print(json.dumps(data, indent=2))
    else:
        print(f"cache file     {args.db}")
        print(f"entries        {len(store):>12,}")
        print(stats.summary())
        if stats.by_model:
            print("\nby model")
            rows = sorted(stats.by_model.items(), key=lambda kv: -kv[1].cost_saved)
            for name, m in rows:
                rate = f"({m.hit_rate:.1%})"
                print(f"  {name:<28} hits {m.hits:>8,} {rate:>8}  saved ${m.cost_saved:,.4f}")
    store.close()
    return 0


def _cmd_list(args: argparse.Namespace) -> int:
    store = _open(args.db)
    entries = sorted(store.entries(), key=lambda e: (-e.hits, -e.created_at))[: args.limit]
    for e in entries:
        prompt = e.prompt.replace("\n", " ")
        if len(prompt) > 70:
            prompt = prompt[:67] + "..."
        print(f"{e.hits:>6} hits  {e.model or '-':<20} {prompt}")
    if not entries:
        print("(empty)")
    store.close()
    return 0


def _cmd_clear(args: argparse.Namespace) -> int:
    store = _open(args.db)
    if args.namespace:
        prefix = f"{args.namespace}::"
        scopes = {e.scope for e in store.entries() if e.scope.startswith(prefix)}
        removed = sum(store.clear(s) for s in scopes)
    else:
        removed = store.clear()
    if args.stats:
        store.save_stats(CacheStats().to_dict())
    print(f"removed {removed} entries")
    store.close()
    return 0


def _cmd_prune(args: argparse.Namespace) -> int:
    store = _open(args.db)
    cutoff = time.time() - args.older_than
    stale = [e.id for e in store.entries() if e.created_at < cutoff]
    for entry_id in stale:
        store.delete(entry_id)
    print(f"removed {len(stale)} entries older than {args.older_than:g}s")
    store.close()
    return 0


def _cmd_prices(args: argparse.Namespace) -> int:
    book = PriceBook()
    print(f"{'model':<24} {'input $/1M':>12} {'output $/1M':>12}")
    for model in book.models():
        price = book.get(model)
        assert price is not None
        print(f"{model:<24} {price.input_per_million:>12.2f} {price.output_per_million:>12.2f}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="semcache", description="Inspect and manage a semcache SQLite cache file."
    )
    parser.add_argument("--version", action="version", version=f"semcache {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    def with_db(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
        p.add_argument("--db", default="semcache.db", help="cache file (default: semcache.db)")
        return p

    p = with_db(sub.add_parser("stats", help="hit rate, tokens and money saved"))
    p.add_argument("--json", action="store_true", help="print machine-readable JSON")
    p.set_defaults(func=_cmd_stats)

    p = with_db(sub.add_parser("list", help="show the most-hit entries"))
    p.add_argument("--limit", type=int, default=20)
    p.set_defaults(func=_cmd_list)

    p = with_db(sub.add_parser("clear", help="delete entries"))
    p.add_argument("--namespace", help="only clear this namespace")
    p.add_argument("--stats", action="store_true", help="also reset the stats")
    p.set_defaults(func=_cmd_clear)

    p = with_db(sub.add_parser("prune", help="delete entries older than N seconds"))
    p.add_argument("--older-than", type=float, required=True, metavar="SECONDS")
    p.set_defaults(func=_cmd_prune)

    p = sub.add_parser("prices", help="show the bundled price table")
    p.set_defaults(func=_cmd_prices)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
