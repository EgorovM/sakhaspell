"""sakhaspell — проверка орфографии якутского языка.

    from sakhaspell import SpellChecker, Lexicon

    checker = SpellChecker(Lexicon.load())
    checker.correct("Ого уорэгэ кисиэхэ")   # -> 'Оҕо үөрэҕэ киһиэхэ'

Словарь на 300 304 словоформы встроен в пакет, скачивать ничего не нужно.
Контекстная модель ставится отдельно: pip install "sakhaspell[tagger]".
"""
from .checker import Issue, SpellChecker, Suggestion
from .grammar import ALL_RULES, RULES, SAFE_RULES, harmony_violations, is_wellformed, violations
from .lexicon import Lexicon, Verdict
from .lm import BigramLM
from .userdict import UserDict
from .norm import normalize
from .pipeline import Correction, Pipeline
from .tokenize import Token, tokenize, words

__version__ = "0.4.0"
__all__ = [
    "SpellChecker", "Lexicon", "Pipeline", "Correction",
    "Issue", "Suggestion", "Verdict",
    "normalize", "tokenize", "words", "Token",
    "harmony_violations", "is_wellformed", "violations",
    "RULES", "SAFE_RULES", "ALL_RULES",
    "BigramLM", "UserDict",
    "__version__",
]
