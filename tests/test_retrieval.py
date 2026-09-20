"""Unit tests for the parts of retrieval that need no database: the fusion arithmetic."""

import pytest

from fineprint.retrieval import rrf_fuse


def fused_order(rankings: list[list[int]], k: int = 60) -> list[int]:
    """Just the chunk ids, in the order fusion put them."""
    return [chunk_id for chunk_id, _ in rrf_fuse(rankings, k)]


def test_a_chunk_in_both_rankings_beats_chunks_in_only_one():
    """The whole point of fusing: agreement between the two retrievers counts for more."""
    lexical = [11, 22, 33]
    vector = [44, 22, 55]

    assert fused_order([lexical, vector])[0] == 22


def test_a_chunk_ranked_second_by_both_beats_a_chunk_ranked_first_by_one():
    """Two seconds (1/62 + 1/62) outweigh one first (1/61), which is why k is large."""
    assert fused_order([[11, 22], [33, 22]])[0] == 22


def test_the_score_is_the_sum_of_one_over_k_plus_rank():
    scores = dict(rrf_fuse([[10, 20], [20]], k=60))

    assert scores[10] == pytest.approx(1 / 61)
    assert scores[20] == pytest.approx(1 / 62 + 1 / 61)


def test_k_changes_the_scores():
    """A smaller k spreads the scores out, because rank 1 is then worth much more than rank 2."""
    assert dict(rrf_fuse([[10, 20]], k=1))[10] == pytest.approx(1 / 2)


def test_ties_are_broken_by_chunk_id():
    """Both chunks score 1/61 + 1/62, so the smaller id has to come first every time."""
    assert fused_order([[5, 9], [9, 5]]) == [5, 9]
    assert fused_order([[9, 5], [5, 9]]) == [5, 9]


def test_a_single_ranking_comes_back_in_its_own_order():
    assert fused_order([[9, 4, 7]]) == [9, 4, 7]


def test_an_empty_ranking_changes_nothing():
    """A retriever that found nothing must not disturb the one that found something."""
    assert rrf_fuse([[3, 1], []]) == rrf_fuse([[3, 1]])


def test_no_rankings_and_only_empty_rankings_fuse_to_nothing():
    assert rrf_fuse([]) == []
    assert rrf_fuse([[], []]) == []
