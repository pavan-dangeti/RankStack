"""Tests for `CodeIndex`, the compact user map serving loads instead of a fifteen-million-key dict.

The claim to prove is narrow: for every id, the index returns exactly the code the dict did, and for
every other string it returns None. A wrong code would serve one user another user's history.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from hypothesis import given
from hypothesis import strategies as st

from rankstack.data.encode import (
    CodeIndex,
    load_encoding,
    load_serving_encoding,
    write_encoding,
    write_serving_encoding,
)

if TYPE_CHECKING:
    from pathlib import Path

    from tests.conftest import Trained


def test_every_user_gets_the_code_the_dict_gave(trained: Trained) -> None:
    """All synthetic users, compared one by one, and iteration in the dict's own order."""
    mapping = trained.encoding.user_to_code
    index = CodeIndex.from_mapping(mapping)

    assert len(index) == len(mapping)
    for user, code in mapping.items():
        assert index.get(user) == code
        assert index[user] == code
        assert user in index
    assert list(index) == list(mapping)
    assert index.agrees_with(mapping)


def test_id_of_finds_the_id_holding_a_code(trained: Trained) -> None:
    """The reverse lookup the demo uses to turn a committed code back into its user."""
    mapping = trained.encoding.user_to_code
    index = CodeIndex.from_mapping(mapping)
    for user, code in mapping.items():
        assert index.id_of(code) == user
    assert index.id_of(len(mapping) + 10) is None


def test_unknown_ids_return_none(trained: Trained) -> None:
    """Near misses of a real id are not real ids: prefixes, extensions, padding, other types."""
    index = CodeIndex.from_mapping(trained.encoding.user_to_code)
    real = next(iter(trained.encoding.user_to_code))
    width = max(len(user.encode()) for user in trained.encoding.user_to_code)
    for unknown in ("", "no-such-user", real[:-1], f"{real}X", f"{real}\x00", real.lower(),
                    real + "X" * (width + 5), "\ud800"):  # fmt: skip
        if unknown in trained.encoding.user_to_code:
            continue
        assert index.get(unknown) is None, unknown
        assert unknown not in index
        with pytest.raises(KeyError):
            index[unknown]
    not_a_string: object = 7
    assert not_a_string not in index


def test_an_over_long_query_cannot_match_a_stored_prefix() -> None:
    """Bytes beyond the array width would be cut off; the lookup must refuse rather than match."""
    index = CodeIndex.from_mapping({"abc": 0, "abd": 1})
    assert index.get("abcd") is None
    assert index.get("abc") == 0


@given(
    st.dictionaries(
        st.text(min_size=1, max_size=12).filter(lambda key: not key.endswith("\x00")),
        st.integers(min_value=0, max_value=2**31 - 1),
        min_size=1,
        max_size=50,
    ),
    st.lists(st.text(max_size=14), max_size=20),
)
def test_any_mapping_round_trips_and_strangers_are_unknown(
    mapping: dict[str, int], strangers: list[str]
) -> None:
    """Arbitrary Unicode ids and codes: same answers as the dict for members and non-members."""
    index = CodeIndex.from_mapping(mapping)
    for key, code in mapping.items():
        assert index.get(key) == code
    for stranger in strangers:
        assert index.get(stranger) == mapping.get(stranger)
    assert index.agrees_with(mapping)


def test_save_and_load_give_back_the_same_index(trained: Trained, tmp_path: Path) -> None:
    """The arrays on disk are the arrays in memory."""
    mapping = trained.encoding.user_to_code
    loaded = CodeIndex.load(CodeIndex.from_mapping(mapping).save(tmp_path / "users.npz"))
    assert loaded.agrees_with(mapping)
    assert list(loaded) == list(mapping)


def test_agrees_with_catches_a_changed_code(trained: Trained) -> None:
    """The build-time check is only worth running if it can fail."""
    mapping = dict(trained.encoding.user_to_code)
    index = CodeIndex.from_mapping(mapping)
    user = next(iter(mapping))
    mapping[user] += 1
    assert not index.agrees_with(mapping)
    assert not index.agrees_with({**mapping, "extra-user": 0})


def test_an_id_ending_in_nul_is_refused_at_build() -> None:
    """Fixed-width bytes drop trailing NULs, so such an id could collide with another."""
    with pytest.raises(ValueError, match="NUL"):
        CodeIndex.from_mapping({"abc\x00": 0})


def test_serving_encoding_matches_the_full_encoding(trained: Trained, tmp_path: Path) -> None:
    """What serving loads holds the same users, items and boundaries as what training wrote."""
    write_encoding(trained.encoding, tmp_path / "corpus")
    full = load_encoding(tmp_path, "corpus")
    write_serving_encoding(full, tmp_path / "corpus")
    served = load_serving_encoding(tmp_path, "corpus")

    assert isinstance(served.user_to_code, CodeIndex)
    assert served.user_to_code.agrees_with(full.user_to_code)
    assert served.item_to_code == full.item_to_code
    assert served.boundaries == full.boundaries
    assert (served.n_users, served.n_items) == (full.n_users, full.n_items)
