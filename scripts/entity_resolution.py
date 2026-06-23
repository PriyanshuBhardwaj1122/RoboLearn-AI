"""
scripts/entity_resolution.py
Post-processing script that merges duplicate nodes in Neo4j.

Finds nodes with similar names using string similarity and merges them
into a single canonical node, redirecting all edges.

Usage:
    python scripts/entity_resolution.py --stats     # show duplication stats
    python scripts/entity_resolution.py --dry-run   # preview merges
    python scripts/entity_resolution.py --run       # actually merge
"""
import argparse
import sys
import os
from collections import defaultdict
from difflib import SequenceMatcher

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config.settings import settings
from neo4j import GraphDatabase

GRN = "\033[92m"; YLW = "\033[93m"; RED = "\033[91m"
GRY = "\033[90m"; BLD = "\033[1m";  RST = "\033[0m"


def similarity(a: str, b: str) -> float:
    a, b = a.lower().strip(), b.lower().strip()
    if a == b:
        return 1.0
    if a in b or b in a:
        longer = max(len(a), len(b))
        shorter = min(len(a), len(b))
        # Only high similarity if lengths are close
        if shorter / longer > 0.6:
            return 0.92
    return SequenceMatcher(None, a, b).ratio()


def find_duplicate_groups(session, threshold: float = 0.88) -> list:
    rows = session.run(
        "MATCH (n) WHERE n.name IS NOT NULL "
        "RETURN n.name AS name ORDER BY n.name"
    )
    all_names = [r["name"] for r in rows
                 if r["name"] and len(r["name"].strip()) > 2]

    print(f"  Total named nodes: {len(all_names)}")

    # Bucket by first 3 chars to reduce O(n²) comparisons
    buckets = defaultdict(list)
    for name in all_names:
        buckets[name.lower()[:3]].append(name)

    groups  = []
    visited = set()

    for key, names in buckets.items():
        if len(names) < 2:
            continue
        for i, a in enumerate(names):
            if a in visited:
                continue
            group = [a]
            for j, b in enumerate(names):
                if i == j or b in visited:
                    continue
                if similarity(a, b) >= threshold:
                    group.append(b)
            if len(group) > 1:
                groups.append(group)
                visited.update(group)

    return groups


def canonical_name(group: list) -> str:
    """Pick the most descriptive name — longest with spaces preferred."""
    with_spaces = [n for n in group if " " in n]
    if with_spaces:
        return max(with_spaces, key=len)
    return max(group, key=len)


def merge_group(session, group: list, dry_run: bool = True) -> int:
    canon = canonical_name(group)
    dups  = [n for n in group if n != canon]
    merges = 0

    for dup in dups:
        if dry_run:
            print(f"    {YLW}WOULD MERGE:{RST} '{dup}'  →  '{canon}'")
            merges += 1
            continue

        # Redirect outgoing edges from duplicate to canonical
        out_rows = list(session.run("""
            MATCH (d {name: $dup})-[r]->(t)
            WHERE t.name IS NOT NULL
            RETURN type(r) AS rel_type, t.name AS target
        """, dup=dup))

        for row in out_rows:
            if row["target"] and row["target"] != canon:
                try:
                    session.run(f"""
                        MATCH (c {{name: $canon}}), (t {{name: $target}})
                        MERGE (c)-[:{row['rel_type']}]->(t)
                    """, canon=canon, target=row["target"])
                except Exception:
                    pass

        # Redirect incoming edges to canonical
        in_rows = list(session.run("""
            MATCH (s)-[r]->(d {name: $dup})
            WHERE s.name IS NOT NULL
            RETURN s.name AS source, type(r) AS rel_type
        """, dup=dup))

        for row in in_rows:
            if row["source"] and row["source"] != canon:
                try:
                    session.run(f"""
                        MATCH (s {{name: $source}}), (c {{name: $canon}})
                        MERGE (s)-[:{row['rel_type']}]->(c)
                    """, source=row["source"], canon=canon)
                except Exception:
                    pass

        # Delete the duplicate node
        session.run("MATCH (d {name: $dup}) DETACH DELETE d", dup=dup)
        print(f"    {GRN}MERGED:{RST} '{dup}'  →  '{canon}'")
        merges += 1

    return merges


def main():
    parser = argparse.ArgumentParser(
        description="Entity resolution — merge duplicate Neo4j nodes"
    )
    parser.add_argument("--stats",     action="store_true")
    parser.add_argument("--dry-run",   action="store_true")
    parser.add_argument("--run",       action="store_true")
    parser.add_argument("--threshold", type=float, default=0.88,
                        help="Similarity threshold (default 0.88)")
    args = parser.parse_args()

    print(f"\n{BLD}Entity Resolution{RST}")
    driver = GraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password)
    )

    with driver.session() as session:
        nodes_before = session.run("MATCH (n) RETURN count(n) AS c").single()["c"]
        rels_before  = session.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
        print(f"  Graph: {nodes_before} nodes, {rels_before} relations\n")

        print(f"Finding duplicates (threshold={args.threshold})...")
        groups = find_duplicate_groups(session, threshold=args.threshold)

        total_can_merge = sum(len(g) - 1 for g in groups)
        print(f"  Groups found    : {len(groups)}")
        print(f"  Nodes mergeable : {total_can_merge}\n")

        if not groups:
            print(f"{GRN}No duplicates found at threshold {args.threshold}{RST}")
            driver.close()
            return

        if args.stats:
            print(f"{BLD}Sample duplicate groups:{RST}")
            for g in groups[:20]:
                canon = canonical_name(g)
                dups  = [n for n in g if n != canon]
                print(f"  {GRN}'{canon}'{RST}")
                for d in dups:
                    print(f"    ← {GRY}'{d}'{RST}")
            driver.close()
            return

        # Dry run or actual run
        total = 0
        for group in groups:
            canon = canonical_name(group)
            dups  = [n for n in group if n != canon]
            print(f"\n  {GRN}'{canon}'{RST}  ←  {dups}")
            total += merge_group(session, group, dry_run=not args.run)

        print(f"\n{'─'*55}")
        if not args.run:
            print(f"{YLW}DRY RUN — {total} merges would be performed{RST}")
            print(f"Run with --run to apply")
        else:
            nodes_after = session.run("MATCH (n) RETURN count(n) AS c").single()["c"]
            rels_after  = session.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
            print(f"{GRN}Done — {total} nodes merged{RST}")
            print(f"  Before: {nodes_before} nodes, {rels_before} relations")
            print(f"  After:  {nodes_after} nodes, {rels_after} relations")

    driver.close()


if __name__ == "__main__":
    main()