from __future__ import annotations

from datetime import datetime
from pathlib import Path

import networkx as nx
import pytest

from experiments.bot_detection.cache import BotDetectionDB


@pytest.fixture
def db(tmp_path: Path) -> BotDetectionDB:
    db = BotDetectionDB(tmp_path / "test.duckdb")
    yield db
    db.close()


def _insert_pr(
    db: BotDetectionDB,
    repo: str,
    number: int,
    author: str,
    state: str = "MERGED",
    created: str = "2024-01-01",
) -> None:
    db.con.execute(
        """INSERT INTO prs (repo, number, author, title, created_at, state, source)
        VALUES (?, ?, ?, ?, ?, ?, 'test')""",
        [repo, number, author, "PR", datetime.fromisoformat(created), state],
    )


def _build_bipartite_graph(
    pairs: list[tuple[str, str, int]],
) -> nx.Graph:
    """Build a bipartite graph from (author, repo, count) triples."""
    g = nx.Graph()
    for author, repo, count in pairs:
        g.add_node(author, bipartite=0)
        g.add_node(repo, bipartite=1)
        g.add_edge(author, repo, weight=count)
    return g


class TestGetAllAuthorRepoPairs:
    def test_basic(self, db: BotDetectionDB) -> None:
        _insert_pr(db, "org/a", 1, "alice")
        _insert_pr(db, "org/a", 2, "alice")
        _insert_pr(db, "org/b", 3, "alice")
        _insert_pr(db, "org/a", 4, "bob")

        pairs = db.get_all_author_repo_pairs()
        pair_dict = {(a, r): c for a, r, c in pairs}
        assert pair_dict[("alice", "org/a")] == 2
        assert pair_dict[("alice", "org/b")] == 1
        assert pair_dict[("bob", "org/a")] == 1
        assert len(pairs) == 3

    def test_empty(self, db: BotDetectionDB) -> None:
        pairs = db.get_all_author_repo_pairs()
        assert pairs == []


class TestHubScore:
    def test_star_topology_high_hub_score(self) -> None:
        """Author connected to many repos should have high hub score."""
        from experiments.bot_detection.stages.stage5_author_features import (
            _compute_network_features,
        )

        pairs = [
            ("star-author", f"repo-{i}", 1) for i in range(10)
        ]
        # Add a leaf author with just one repo
        pairs.append(("leaf-author", "repo-0", 1))

        df = _compute_network_features(pairs)
        star_hub = df.loc[df["login"] == "star-author", "hub_score"].iloc[0]
        leaf_hub = df.loc[df["login"] == "leaf-author", "hub_score"].iloc[0]
        assert star_hub > leaf_hub

    def test_equal_authors_equal_hubs(self) -> None:
        """Symmetric authors should have equal hub scores."""
        from experiments.bot_detection.stages.stage5_author_features import (
            _compute_network_features,
        )

        pairs = [
            ("alice", "repo-a", 1),
            ("alice", "repo-b", 1),
            ("bob", "repo-a", 1),
            ("bob", "repo-b", 1),
        ]
        df = _compute_network_features(pairs)
        alice_hub = df.loc[df["login"] == "alice", "hub_score"].iloc[0]
        bob_hub = df.loc[df["login"] == "bob", "hub_score"].iloc[0]
        assert abs(alice_hub - bob_hub) < 1e-6


class TestBipartiteClustering:
    def test_clique_high_clustering(self) -> None:
        """Authors in a clique (all share all repos) should have high clustering."""
        # 3 authors each connected to 2 repos -> complete bipartite subgraph
        pairs = [
            ("a1", "r1", 1), ("a1", "r2", 1),
            ("a2", "r1", 1), ("a2", "r2", 1),
            ("a3", "r1", 1), ("a3", "r2", 1),
        ]
        g = _build_bipartite_graph(pairs)
        top_nodes = {n for n, d in g.nodes(data=True) if d["bipartite"] == 0}
        clustering = nx.bipartite.clustering(g, nodes=top_nodes)

        for author in ["a1", "a2", "a3"]:
            assert clustering[author] > 0.5

    def test_isolated_author_zero_clustering(self) -> None:
        """Author connected to a single unique repo -> zero clustering."""
        pairs = [
            ("isolated", "unique-repo", 1),
            ("other1", "shared-repo", 1),
            ("other2", "shared-repo", 1),
        ]
        g = _build_bipartite_graph(pairs)
        top_nodes = {n for n, d in g.nodes(data=True) if d["bipartite"] == 0}
        clustering = nx.bipartite.clustering(g, nodes=top_nodes)

        assert clustering["isolated"] == 0.0
