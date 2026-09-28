
from __future__ import annotations
import re
from dataclasses import dataclass

# detecteaza orice forma de nr financiar
_NUMBER = re.compile(
    r"(?<![\w.])(?:\$?\d{1,3}(?:,\d{3})+(?:\.\d+)?|\$?\d+(?:\.\d+)?)(?:%|x|\s*(?:million|billion|trillion|[MBT]))?",
    re.IGNORECASE,
)
# cauta locurile unde se termina o prop sau unde incepe o noua linie dintr o lista
# pentru a putea sparge raspunsul AI ului in fraze individuale
_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+(?=-|\d+\.)")


@dataclass(frozen=True)
class FaithfulnessResult:
    score: int | None
    checked_claims: int
    supported_claims: int
    unsupported_claims: list[str]
    evidence_available: bool

    def as_dict(self) -> dict:
        return {
            "score": self.score,
            "checked_claims": self.checked_claims,
            "supported_claims": self.supported_claims,
            "unsupported_claims": self.unsupported_claims,
            "evidence_available": self.evidence_available,
        }


def _normalise_number(value: str) -> str:  # facem 1200$ comparabil cu 1200
    return re.sub(r"[,$\s]", "", value).lower()


# toate cifrele dintr o propozitie trebuie sa se gaseasca in textul brut returnat de unelte
def _claim_is_supported(claim: str, evidence: str) -> bool:
    values = [_normalise_number(m.group(0)) for m in _NUMBER.finditer(claim)]
    normalised_evidence = _normalise_number(evidence)
    return bool(values) and all(value in normalised_evidence for value in values)


def evaluate_faithfulness(answer: str, tool_outputs: list[dict[str, str]]) -> FaithfulnessResult:
   # procentaj de enunturi validate din totalul celor verificate
    evidence = "\n".join(output.get("output", "") for output in tool_outputs)
    if not evidence.strip():
        return FaithfulnessResult(None, 0, 0, [], False)

    claims = []
    for sentence in _SENTENCE.split(answer):
        sentence = sentence.strip().lstrip("-• ")
        if _NUMBER.search(sentence):
            claims.append(sentence)

    if not claims:
        return FaithfulnessResult(None, 0, 0, [], True)

    supported = []
    unsupported = []

    for claim in claims:
        if _claim_is_supported(claim, evidence):
            supported.append(claim)
        else:
            unsupported.append(claim)

    total_claims = len(claims)
    supported_count = len(supported)

    score = round(100 * supported_count / total_claims)

    return FaithfulnessResult(
        score=score,
        checked_claims=total_claims,
        supported_claims=supported_count,
        unsupported_claims=unsupported,
        evidence_available=True,
    )
