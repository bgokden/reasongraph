"""Entity canonicalization -- the light stand-in for coreference.

Extracted entity strings vary across chunks and articles: "Apple Inc.", "Apple",
"the Fed", "Federal Reserve". Left as-is, each becomes its own node and the
shared-entity bridge that reconnects facts never forms. A canonicalizer maps every
surface form to one canonical name so those facts bridge.

This is NOT coreference: it unifies *named-entity variants*, not document-local
pronouns or definite noun phrases ("the company" -> Northwind). Real coref needs a
model (deferred to an isolated sidecar); canonicalization is the cheap, dependency-
free lever that closes the cross-article half of the gap.

The graph provides the mechanism (a pluggable ``canonicalizer`` callable); the
caller provides the policy (the alias map). ``AliasCanonicalizer`` is the default
policy: whitespace/case normalization + corporate-suffix stripping + an alias map.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping

# A canonicalizer maps one extracted entity surface form to its canonical name.
CanonicalizerFn = Callable[[str], str]

# Corporate / legal suffixes to strip so "Apple Inc." and "Apple" converge without
# needing an explicit alias entry. Anchored to the end, requires a separator before
# the suffix so a name that merely ends in these letters is untouched.
_SUFFIX_RE = re.compile(
    r"[,\s]+"
    r"(?:inc|incorporated|corp|corporation|co|company|ltd|limited|llc|l\.l\.c|"
    r"plc|gmbh|ag|sa|s\.a|nv|n\.v|se|spa|s\.p\.a|pty|bv|b\.v|oyj|ab|as|kk|kgaa|"
    r"holdings|holding|group)"
    r"\.?$",
    re.IGNORECASE,
)
_WS_RE = re.compile(r"\s+")


class AliasCanonicalizer:
    """Default canonicalization policy: normalize, strip corporate suffixes, alias.

    Resolution order for an entity string:
      1. Normalize whitespace (collapse runs, strip ends).
      2. If the normalized (case-folded) form is an alias key, return its canonical
         value, itself reduced (step 4) so an alias target carrying a suffix
         ("AAPL" -> "Apple Inc.") converges with plainly-stripped variants.
      3. Otherwise reduce the form: strip trailing corporate/legal suffixes to a
         fixed point, then re-check the alias map on the reduced form.
      4. Return the reduced form (original casing kept).

    Idempotent (``canon(canon(x)) == canon(x)``) for a well-formed alias map -- one
    whose canonical values are not themselves surface keys (i.e. no alias
    chains/cycles). Suffix stripping reduces to a fixed point, so stacked suffixes
    ("Foo Group Holdings" -> "Foo") are already stable after the first pass.

    Args:
        aliases: Optional mapping of surface form -> canonical name. Keys are matched
            case-insensitively and after whitespace normalization, so
            ``{"the fed": "Federal Reserve"}`` matches "The Fed" and "the  fed".
        strip_suffixes: When True (default), strip trailing corporate/legal suffixes
            ("Inc.", "Corp.", "GmbH", ...) so legal variants of one name converge.
    """

    def __init__(
        self,
        aliases: Mapping[str, str] | None = None,
        strip_suffixes: bool = True,
    ) -> None:
        self.strip_suffixes = strip_suffixes
        self._aliases: dict[str, str] = {}
        if aliases:
            for surface, canonical in aliases.items():
                self._aliases[self._norm_key(surface)] = canonical

    @staticmethod
    def _norm_key(s: str) -> str:
        return _WS_RE.sub(" ", s.strip()).casefold()

    def _reduce(self, s: str) -> str:
        """Whitespace-normalize and strip trailing corporate suffixes to a fixed point.

        Loops so stacked suffixes fully reduce ("Foo Group Holdings" -> "Foo"); never
        reduces to empty (a bare "Inc." is left as-is). Does no alias lookup, so it
        cannot cycle -- this is what makes a single ``__call__`` idempotent.
        """
        s = _WS_RE.sub(" ", s.strip())
        if not self.strip_suffixes:
            return s
        while True:
            stripped = _SUFFIX_RE.sub("", s).strip()
            if not stripped or stripped == s:
                return s
            s = stripped

    def __call__(self, entity: str) -> str:
        base = _WS_RE.sub(" ", entity.strip())
        if not base:
            return ""
        # Alias values are themselves reduced so an alias target that carries a
        # corporate suffix ("AAPL" -> "Apple Inc.") converges with plainly-stripped
        # variants ("Apple Inc." -> "Apple") instead of landing on a separate node.
        hit = self._aliases.get(base.casefold())
        if hit is not None:
            return self._reduce(hit)
        reduced = self._reduce(base)
        hit = self._aliases.get(reduced.casefold())
        return self._reduce(hit) if hit is not None else reduced
