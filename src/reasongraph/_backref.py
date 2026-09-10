"""Back-reference detection: when a note points at the sentence before it.

People writing notes chain causes by pointing backwards rather than repeating the subject::

    The billing migration finished on Tuesday.
    This broke invoice delivery for about forty accounts.

Extraction reads one sentence at a time, so "This" refers to nothing it can see and the link is
lost. The device differs by language and it is emphatically not always a pronoun:

* English, Spanish, French use a demonstrative subject ("This caused", "Esto provocó", "Cela a
  entraîné") or a connective ("por ello", "par conséquent").
* German and Dutch mostly use a pronominal adverb that carries the antecedent inside the word:
  "dadurch", "daraufhin", "infolgedessen"; "daardoor", "hierdoor", "vandaar". Closed class, and
  German word order puts it first, so it is easier to spot than the English case.
* Turkish marks the demonstrative with case suffixes, so match the stem: "bu", "bunun", "bundan",
  "bu nedenle", "bu yüzden".

Measured limits, worth knowing before relying on this: Spanish and Turkish drop subjects, so the
back-reference is sometimes absent from the surface entirely and no rule can see it; Turkish also
tends to express cause inside one sentence, so it has less of this to find. On real incident
write-ups about a fifth of causal links between sentences are surface-marked back-references.
"""

from __future__ import annotations

import re

# Two kinds of opener, and the difference matters for recall.
#
# A CONNECTIVE already says "because of what I just told you" on its own: "as a result",
# "dadurch", "por ello", "bu nedenle". Requiring a causal verb as well would throw away the
# clearest cases, since the connective IS the causal claim.
_CAUSAL_MARKERS = {
    "en": ["as a result", "consequently", "therefore", "because of this", "because of that",
           "following this", "as a consequence", "hence", "thus", "so"],
    "de": ["dadurch", "daraufhin", "infolgedessen", "deshalb", "deswegen", "somit", "daher",
           "aus diesem grund"],
    "nl": ["daardoor", "hierdoor", "daarom", "vandaar", "dientengevolge", "als gevolg hiervan"],
    "es": ["por ello", "por eso", "por lo tanto", "por consiguiente", "como consecuencia",
           "debido a esto", "a raíz de esto"],
    "fr": ["par conséquent", "de ce fait", "en conséquence", "ainsi", "d'où", "à cause de cela"],
    "tr": ["bu nedenle", "bu yüzden", "bundan dolayı", "bunun sonucunda", "bu sebeple"],
}

# A bare DEICTIC only points; it does not claim causation. "This is fine" is not a causal link.
# These count only when the sentence also asserts a cause.
_DEICTICS = {
    "en": ["this", "that", "these", "those", "it"],
    "de": ["dies", "das", "diese", "dieser"],
    "nl": ["dit", "dat", "deze"],
    "es": ["esto", "eso", "esta", "ese"],
    "fr": ["cela", "ceci", "ce"],
    "tr": ["bu", "bunun", "buna", "bundan", "şu", "şunun"],
}

# The relation verbs that make a bare demonstrative a causal claim rather than a comment.
_CAUSAL = {
    "en": ["caused", "led to", "resulted in", "broke", "triggered", "meant", "forced", "made",
           "prevented", "delayed", "produced", "drove", "lost"],
    "de": ["führte", "verursachte", "löste aus", "bewirkte", "sorgte", "brachte", "verhinderte",
           "stieg", "fiel"],
    "nl": ["leidde", "veroorzaakte", "zorgde", "brak", "verhinderde", "viel"],
    "es": ["causó", "provocó", "produjo", "generó", "llevó", "impidió", "rompió", "obligó"],
    "fr": ["a causé", "a provoqué", "a entraîné", "a conduit", "a empêché", "a cassé", "a obligé",
           "est tombé"],
    "tr": ["neden oldu", "yol açtı", "sebep oldu", "bozdu", "engelledi", "kesildi", "durdu"],
}

_WORD = re.compile(r"\w+", re.UNICODE)


def _norm(text: str) -> str:
    return text.strip().lower()


def _starts_with(text: str, phrases) -> bool:
    t = _norm(text)
    for p in phrases:
        if t.startswith(p) and (len(t) == len(p) or not t[len(p)].isalnum()):
            return True
    return False


def _flat(table, langs):
    return [p for lang, ps in table.items() if not langs or lang in langs for p in ps]


def opens_with_back_reference(text: str, langs=None) -> bool:
    """True when the sentence starts by pointing at whatever came before it."""
    return (_starts_with(text, _flat(_CAUSAL_MARKERS, langs))
            or _starts_with(text, _flat(_DEICTICS, langs)))


def asserts_a_cause(text: str, langs=None) -> bool:
    """True when the sentence states that something caused something."""
    t = _norm(text)
    return any(v in t for v in _flat(_CAUSAL, langs))


def is_back_referenced_cause(text: str, langs=None) -> bool:
    """A sentence whose cause is the sentence before it, said only by pointing backwards.

    A causal connective ("as a result", "dadurch", "bu nedenle") is enough on its own: it already
    asserts the link. A bare demonstrative ("This ...") needs a causal verb as well, or "This is
    fine" would become an edge. That pairing is the whole precision mechanism, and it is what kept
    this from inventing cross-case links when it was measured; a similarity check cannot help here,
    because a pure back-reference shares no words with what it refers to, by definition.
    """
    if _starts_with(text, _flat(_CAUSAL_MARKERS, langs)):
        return True
    return _starts_with(text, _flat(_DEICTICS, langs)) and asserts_a_cause(text, langs)
