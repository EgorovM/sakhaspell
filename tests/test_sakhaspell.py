"""Тесты компонентов, не требующие корпуса.

Проверяют то, что легко сломать незаметно: границы токенов, обратимость
нормализации, выравнивание меток тэггера.

    python -m pytest tests/ -q
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pytest

from sakhaspell.errors import denormalize, restoration_variants
from sakhaspell.fuzzy import Trie, sub_cost
from sakhaspell.lexicon import Lexicon
from sakhaspell.norm import fix_confusables, fix_mixed_script, normalize, strip_invisible
from sakhaspell.tagger import apply_tags, make_tags
from sakhaspell.tokenize import sentences, tokenize, words, sakha_letter_share


# --- токенизация -------------------------------------------------------------
def test_offsets_point_at_original_text():
    s = "Оҕо, үөрэҕэ — киһиэхэ!"
    for t in tokenize(s):
        assert s[t.start:t.end] == t.text


def test_hyphenated_compound_is_one_token():
    ts = words("сайыҥҥыттан-күһүҥҥүттэн")
    assert len(ts) == 1
    assert ts[0].text == "сайыҥҥыттан-күһүҥҥүттэн"


def test_token_kinds():
    kinds = {t.text: t.kind for t in tokenize("Саха 1990-с Linux")}
    assert kinds["Саха"] == "word"
    assert kinds["Linux"] == "latin"
    assert kinds["1990-с"] in ("number", "word")


def test_abbreviation_does_not_split_sentence():
    s = "Ол 1990 с. буолбута. Иккис этии."
    assert len(sentences(s)) == 2


def test_sakha_letter_share():
    # в живом якутском тексте доля ҕҥөһү около 7.6%
    assert 0.0 < sakha_letter_share("Оҕо үөрэҕэ киһиэхэ саҥа") < 0.4
    assert sakha_letter_share("текст без якутских букв") == 0.0


# --- нормализация ------------------------------------------------------------
@pytest.mark.parametrize("bad,good", [
    ("ѳс", "өс"), ("саңа", "саҥа"), ("аға", "аҕа"), ("ұс", "үс"), ("ӊ", "ҥ"),
])
def test_confusables_fixed(bad, good):
    assert fix_confusables(bad) == good


def test_confusables_preserve_length():
    s = "ѳң ғұ ӊҋ"
    assert len(fix_confusables(s)) == len(s)


def test_invisible_removed_with_offset_map():
    s = "оҕо­лор"
    out, idx = strip_invisible(s)
    assert out == "оҕолор"
    assert all(s[i] == c for c, i in zip(out, idx))


def test_latin_homoglyph_fixed_only_inside_cyrillic_word():
    assert fix_mixed_script("Cаха") == "Саха"      # латинская C
    assert fix_mixed_script("CD-ROM") == "CD-ROM"  # трогать нельзя


def test_normalize_is_idempotent():
    s = "Cаха­ сирэ ѳс саңа"
    assert normalize(normalize(s)) == normalize(s)


# --- модель ошибок -----------------------------------------------------------
def test_denormalize_removes_special_letters():
    out = denormalize("Оҕо үөрэҕэ киһиэхэ саҥа")
    assert not any(c in "ҕҥөүһ" for c in out.lower())


@pytest.mark.parametrize("wrong,right", [
    ("ого", "оҕо"), ("киhи", "киһи"), ("уьу", "уһу"),
    ("сана", "саҥа"), ("хацалас", "хаҥалас"), ("кор", "көр"),
])
def test_restoration_contains_correct_variant(wrong, right):
    assert right in restoration_variants(wrong)


def test_restoration_includes_input_unchanged():
    assert "ого" in restoration_variants("ого")


# --- поиск по расстоянию -----------------------------------------------------
def test_cheap_substitutions_are_cheaper_than_typos():
    assert sub_cost("г", "ҕ") < sub_cost("г", "б")
    assert sub_cost("ь", "һ") < sub_cost("ь", "б")


def test_trie_finds_within_cost():
    t = Trie.from_freq({"оҕо": 100, "оҕолор": 50, "киһи": 90})
    got = {c.form for c in t.search("ого", max_cost=100)}
    assert "оҕо" in got


def test_trie_contains():
    t = Trie.from_freq({"оҕо": 1})
    assert "оҕо" in t and "оҕол" not in t


def test_transposition_costs_one_edit():
    t = Trie.from_freq({"киһи": 10})
    got = t.search("икһи", max_cost=100)
    assert any(c.form == "киһи" for c in got)


# --- метки тэггера -----------------------------------------------------------
@pytest.mark.parametrize("src,tgt", [
    ("ого", "оҕо"), ("киhи", "киһи"), ("уьу", "уһу"), ("санга", "саҥа"),
    ("хацалас", "хаҥалас"), ("тонгус", "тоҥус"),
    ("Ого уорэгэ", "Оҕо үөрэҕэ"),
])
def test_tags_roundtrip(src, tgt):
    tags = make_tags(src, tgt)
    assert tags is not None, f"не размечено: {src} -> {tgt}"
    assert apply_tags(src, tags) == tgt


def test_tags_reject_incompatible_pair():
    # порча не сводится к восстановлению спецбукв — такую пару учить нельзя
    assert make_tags("оҕо", "биэрэр") is None


def test_tags_length_matches_source():
    src, tgt = "Ого уорэгэ кисиэхэ", "Оҕо үөрэҕэ киһиэхэ"
    assert len(make_tags(src, tgt)) == len(src)


def test_clean_text_gets_all_keep():
    s = "Оҕо үөрэҕэ"
    tags = make_tags(s, s)
    assert set(tags) == {0}


# --- лексикон ----------------------------------------------------------------
def test_lexicon_hyphen_parts():
    lex = Lexicon(core={"оҕо": 10, "уруу": 10})
    assert lex.check_form("оҕо-уруу").ok
    assert not lex.check_form("оҕо-ххх").ok


def test_lexicon_shadow_rejected():
    lex = Lexicon(core={"саҥа": 100}, shadow={"сана": "саҥа"})
    assert not lex.check_form("сана").ok
    assert lex.check_form("саҥа").ok


def test_lexicon_tail_needs_corroboration():
    lex = Lexicon(core={}, tail={"редкое": 1}, other={"редкое": 5})
    assert lex.check_form("редкое").ok
    lex2 = Lexicon(core={}, tail={"опечатка": 1}, other={})
    assert not lex2.check_form("опечатка").ok


def test_numbers_and_latin_are_skipped():
    lex = Lexicon(core={})
    toks = {t.text: t for t in tokenize("2024 Linux")}
    assert lex.check_token(toks["2024"]).ok
    assert lex.check_token(toks["Linux"]).ok


# --- правила грамматики ------------------------------------------------------
from sakhaspell.grammar import (cluster_violations, harmony_violations,
                                is_loanword, parse_nuclei, positional_violations)


@pytest.mark.parametrize("word,expected", [
    ("оҕо", ["о", "о"]),
    ("үөрэх", ["үө", "э"]),            # дифтонг разбирается как одно ядро
    ("кыаҕы", ["ыа", "ы"]),
    ("сүһүөх", ["ү", "үө"]),
    ("уонна", ["уо", "а"]),
    ("кыыс", ["ыы"]),                  # долгота — одно ядро, не две гласные
    ("оскуола", ["о", "уо", "а"]),
])
def test_nuclei_parsing(word, expected):
    assert [n for _, n in parse_nuclei(word)] == expected


@pytest.mark.parametrize("word", [
    "оҕо", "үөрэх", "киһи", "туһунан", "уонна", "кыаҕы", "олоҥхо",
    "оҕолорбутугар", "үөрэммитэ", "биһиги", "түмүгэр", "сайын",
])
def test_correct_words_obey_harmony(word):
    assert harmony_violations(word) == []


@pytest.mark.parametrize("word", [
    "сурэ",      # должно быть сүрэ: у→э невозможно
    "уорэх",     # үөрэх
    "тумугэр",   # түмүгэр
    "болла",     # буолла
])
def test_denormalized_words_break_harmony(word):
    assert harmony_violations(word), f"{word} должно нарушать гармонию"


@pytest.mark.parametrize("word", [
    "дьон-сэргэ", "күүс-көмө", "дьиэ-уот", "ас-үөл", "үп-харчы",
])
def test_hyphen_parts_checked_separately(word):
    # гармония действует внутри слова; через дефис ряд меняется законно
    assert harmony_violations(word) == []


def test_loanwords_are_exempt():
    # буквы, которые в якутском бывают только в заимствованиях
    assert is_loanword("революция")
    assert not is_loanword("оҕо")
    # заимствование не должно давать нарушений, даже если ряд не выдержан
    assert harmony_violations("революция") == []


def test_positional_rule():
    assert positional_violations("ҥа")          # слово не начинается на ҥ
    assert not positional_violations("оҕо")


def test_cluster_rule_ignores_loanwords():
    assert cluster_violations("млрд")
    assert cluster_violations("оҕолор") == []


# --- пользовательский словарь -------------------------------------------------
from sakhaspell.userdict import UserDict


def test_userdict_accepts_and_replaces(tmp_path):
    ud = UserDict()
    ud.add("Ньургуйаана")
    ud.add_replacement("сахалыы", "саха тылынан")
    assert ud.accepts("ньургуйаана")
    assert ud.accepts("НЬУРГУЙААНА")
    assert not ud.accepts("сахалыы")           # слово, которое велено править
    assert ud.correction("сахалыы") == "саха тылынан"


def test_userdict_roundtrip(tmp_path):
    p = tmp_path / "ud.txt"
    ud = UserDict()
    ud.add("скайраннинг")
    ud.add_replacement("неверно", "верно")
    ud.save(p)
    back = UserDict.load(p)
    assert back.accept == ud.accept
    assert back.replace == ud.replace


def test_userdict_ignores_comments(tmp_path):
    p = tmp_path / "ud.txt"
    p.write_text("# комментарий\n\nслово\nа -> б\n", encoding="utf-8")
    ud = UserDict.load(p)
    assert ud.accept == {"слово"}
    assert ud.replace == {"а": "б"}


def test_userdict_suppresses_flag():
    lex = Lexicon(core={"оҕо": 10})
    from sakhaspell.checker import SpellChecker
    ud = UserDict()
    ud.add("калибулин")
    ch = SpellChecker(lex, userdict=ud)
    assert [i.token.text for i in ch.check("калибулин оҕо")] == []
    assert [i.token.text for i in SpellChecker(lex).check("калибулин оҕо")] == ["калибулин"]


# --- контекстная модель -------------------------------------------------------
from sakhaspell.lm import BOS, BigramLM


def _toy_lm() -> BigramLM:
    uni = {"бу": 100, "сайын": 30, "ыйын": 60, "үлэлээбит": 20}
    bi = {"бу": {"сайын": 25}, "сайын": {"үлэлээбит": 15}}
    return BigramLM(uni, bi, sum(uni.values()))


def test_lm_prefers_seen_bigram():
    lm = _toy_lm()
    # «бу сайын» в модели есть, «бу ыйын» нет — несмотря на то, что «ыйын»
    # вдвое частотнее само по себе
    assert lm.score("сайын", "бу") > lm.score("ыйын", "бу")


def test_lm_falls_back_to_unigram():
    lm = _toy_lm()
    # без контекста побеждает частотное
    assert lm.logp_unigram("ыйын") > lm.logp_unigram("сайын")
    # неизвестное слово получает конечную, но малую вероятность
    assert lm.logp_unigram("такогонет") < lm.logp_unigram("сайын")


def test_lm_uses_right_context():
    lm = _toy_lm()
    with_right = lm.score("сайын", None, "үлэлээбит")
    without = lm.score("сайын", None, None)
    assert with_right != without


def test_lm_decides_between_equally_cheap_candidates():
    """Контекст решает там, где правки равны по цене.

    Если один кандидат дешевле по расстоянию, цена правки перевешивает: так и
    задумано, контекст не должен переставлять заметно более близкий вариант.
    Поэтому кандидаты здесь равноудалены — обе правки по одной замене.
    """
    from sakhaspell.checker import SpellChecker
    uni = {"бу": 100, "сайын": 30, "тайын": 60}
    bi = {"бу": {"сайын": 25}}
    lm = BigramLM(uni, bi, sum(uni.values()))
    lex = Lexicon(core={"сайын": 30, "тайын": 60})

    # без контекста побеждает частотное
    assert SpellChecker(lex).suggest("байын")[0].form == "тайын"
    # с контекстом «бу ___» — то, что встречалось после «бу»
    smart = SpellChecker(lex, lm=lm)
    assert smart.suggest("байын", "бу", None)[0].form == "сайын"


def test_lm_does_not_override_cheaper_edit():
    """Контекст не должен побеждать заметно более дешёвую правку."""
    from sakhaspell.checker import SpellChecker
    lex = Lexicon(core={"сайын": 30, "ыйын": 60})
    smart = SpellChecker(lex, lm=_toy_lm())
    # «ыйын» на одну замену соседних по раскладке букв, «сайын» — на вставку;
    # контекст сдвигает оценку, но порядок не переворачивает
    assert smart.suggest("сйын", "бу", None)[0].form == "ыйын"


# --- правила: реестр ----------------------------------------------------------
from sakhaspell.grammar import ALL_RULES, RULES, SAFE_RULES


def test_rules_registry_complete():
    assert len(RULES) == 8
    assert set(SAFE_RULES) <= set(RULES)
    assert set(ALL_RULES) == set(RULES)


def test_final_rule_narrowed_to_ghe():
    from sakhaspell.grammar import rule_final
    # корпус показал, что «тыһ» — законное слово, а не нарушение
    assert rule_final("тыһ") == []
    assert rule_final("буоллаҕ")


def test_soft_sign_rule_catches_real_error():
    from sakhaspell.grammar import rule_soft_sign
    assert rule_soft_sign("уьу")        # ь вместо һ — самая частая ошибка
    assert rule_soft_sign("сылдьар") == []
    assert rule_soft_sign("аньыы") == []
