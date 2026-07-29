from reasongraph._canonical import AliasCanonicalizer


def test_whitespace_normalization():
    c = AliasCanonicalizer()
    assert c("  Apple   Inc.  ") == "Apple"
    assert c("Federal   Reserve") == "Federal Reserve"


def test_strips_corporate_suffixes():
    c = AliasCanonicalizer()
    assert c("Apple Inc.") == "Apple"
    assert c("Acme Corp.") == "Acme"
    assert c("Foo Ltd") == "Foo"
    assert c("Bar LLC") == "Bar"
    assert c("SAP SE") == "SAP"
    assert c("Volkswagen AG") == "Volkswagen"
    assert c("BlackRock Holdings") == "BlackRock"


def test_does_not_overstrip_names_that_merely_end_in_suffix_letters():
    c = AliasCanonicalizer()
    # No separator before "co"/"as"/"ag" -> not a suffix, left intact.
    assert c("Cisco") == "Cisco"
    assert c("Atlas") == "Atlas"
    assert c("Incoming") == "Incoming"


def test_suffix_stripping_can_be_disabled():
    c = AliasCanonicalizer(strip_suffixes=False)
    assert c("Apple Inc.") == "Apple Inc."


def test_alias_map_is_case_and_whitespace_insensitive():
    c = AliasCanonicalizer({"the fed": "Federal Reserve"})
    assert c("The Fed") == "Federal Reserve"
    assert c("the  FED") == "Federal Reserve"
    assert c("Federal Reserve") == "Federal Reserve"  # already canonical, unchanged


def test_alias_applies_after_suffix_strip():
    # "Apple Inc." is not an alias key, but its suffix-stripped form "Apple" is.
    c = AliasCanonicalizer({"apple": "AAPL"})
    assert c("Apple Inc.") == "AAPL"
    assert c("Apple") == "AAPL"


def test_alias_beats_suffix_strip_when_full_form_is_keyed():
    c = AliasCanonicalizer({"apple inc.": "AAPL-full"})
    assert c("Apple Inc.") == "AAPL-full"


def test_strips_stacked_suffixes_to_a_fixed_point():
    # More than one trailing corporate token collapses fully in a single call, so
    # "Foo Group Holdings" and "Foo Group" and "Foo" all converge to one node.
    c = AliasCanonicalizer()
    assert c("Foo Group Holdings") == "Foo"
    assert c("Bar Co Ltd") == "Bar"
    assert c("Acme Holdings Inc.") == "Acme"


def test_bare_suffix_is_not_reduced_to_empty():
    c = AliasCanonicalizer()
    assert c("Inc.") == "Inc."
    assert c("Group") == "Group"


def test_alias_value_with_suffix_converges_with_stripped_variants():
    # An alias target carrying a corporate suffix is reduced too, so the aliased
    # form and plainly-extracted forms land on the SAME node (the whole point).
    c = AliasCanonicalizer({"aapl": "Apple Inc."})
    assert c("aapl") == "Apple"
    assert c("Apple Inc.") == "Apple"
    assert c("Apple") == "Apple"


def test_idempotent():
    c = AliasCanonicalizer({"the fed": "Federal Reserve", "aapl": "Apple Inc."})
    for surface in ["Apple Inc.", "The Fed", "Cisco", "Bar LLC",
                    "Foo Group Holdings", "aapl", "Acme Holdings Inc."]:
        once = c(surface)
        assert c(once) == once


def test_empty_and_whitespace_return_empty():
    c = AliasCanonicalizer()
    assert c("") == ""
    assert c("   ") == ""
