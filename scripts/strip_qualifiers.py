#!/usr/bin/env python3
"""Strip participle qualifiers from noun phrases, keeping the noun part.

"overriding debt" is not a noun, but "debt" is. Rather than discard the row,
this removes the qualifier and keeps what is left, then dedupes against
phrases that already exist in the dataset.

A word is treated as a strippable qualifier when all of these hold:
  * it is not the last word (only pre-head modifiers qualify)
  * spaCy tags it VBG (continuous) or VBN/VBD (past tense)
  * WordNet knows no noun sense for it

That last test is what separates a qualifier from a noun modifier of the same
shape: "overriding" has no noun sense, so "overriding debt" -> "debt", while
"reporting", "packaging" and "quarrying" all do, so "reporting channels" is
left alone.

A hyphenated word is one unit: it is tagged whole rather than split into parts,
and is never stripped, so "healthcare-associated infections" survives intact
and "overriding debt-service costs" -> "debt-service costs".

Usage:
  python3 scripts/strip_qualifiers.py --dry-run   # report only
  python3 scripts/strip_qualifiers.py             # rewrite the CSV
"""

import argparse
import csv
import json
import os
import random
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import spacy
from nltk.corpus import wordnet as wn
from spacy.tokens import Doc

QUALIFIER_TAGS = {"VBG", "VBN", "VBD"}

# WordNet only knows the compounds it happens to list, so it protects
# "registered nurse" but not "grounded theory", "gated community" or
# "nonperforming loans". This second opinion catches the rest.
VERIFY_PROMPT = """Each term below is a noun preceded by a participle \
(an -ing or -ed word). Decide whether the participle is part of the thing's \
established name.

Answer 1 if the term is a recognised name for something and would lose its \
meaning if the participle were removed: grounded theory, gated community, \
nonperforming loans, suspended solids, saturated zone, registered nurse, \
organized crime, armed robbery, developing country, processed food, \
running water, dried fruit, boiling point, graduating class, sparkling water.

Answer 0 if the participle merely describes the noun and the noun stands \
perfectly well alone: overriding debt (debt), increased crisis risk (crisis \
risk), forced land acquisition (land acquisition), alleged negligence \
(negligence), renewed push (push), attempted assassination (assassination), \
stalled negotiations (negotiations), leaking roof (roof), forged documents \
(documents).

The test is whether the phrase names a distinct kind of thing (1) or is just \
a noun with a description attached (0). When genuinely unsure, answer 1 so \
the term is left intact.

Answer with one integer (1 or 0) per term, in the same order as the input."""


class WhitespaceTokenizer:
    """Split on whitespace only, so a hyphenated word stays a single token.

    spaCy's default tokenizer splits "debt-service" into three tokens, which
    both breaks word/tag alignment and invites tagging the pieces of a
    lexicalised compound as if they stood alone.
    """

    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        return Doc(self.vocab, words=text.split())

_noun_cache = {}


def has_noun_sense(word):
    """True if WordNet lists a noun sense (so the word can modify as a noun)."""
    w = word.lower()
    if w not in _noun_cache:
        try:
            _noun_cache[w] = bool(wn.synsets(w, pos=wn.NOUN))
        except LookupError:
            sys.exit("WordNet data missing: python3 -m nltk.downloader wordnet")
    return _noun_cache[w]


def is_known_term(words):
    """True if these words form a noun entry in WordNet ("registered nurse").

    Catches lexicalised participle compounds, where the participle is part of
    the thing's name and must not be stripped.
    """
    if len(words) < 2:
        return False
    lemma = "_".join(words).lower()
    if lemma not in _noun_cache:
        _noun_cache[lemma] = bool(wn.synsets(lemma, pos=wn.NOUN))
    return _noun_cache[lemma]


def strip_qualifiers(phrase, doc):
    """Return the phrase with participle qualifiers removed."""
    words = phrase.split()
    if len(words) < 2:
        return phrase

    tokens = list(doc)
    if len(tokens) != len(words):
        return phrase  # tokenisation disagrees; leave it alone

    kept = []
    for i, (word, tok) in enumerate(zip(words, tokens)):
        last = i == len(words) - 1
        strippable = (
            not last
            and "-" not in word          # hyphenated units are left whole
            and tok.tag_ in QUALIFIER_TAGS
            and not has_noun_sense(word)
            # "registered nurse" / "armed robbery": the participle belongs to
            # the name, so check the compound before removing it.
            and not is_known_term(words[i:])
            and not is_known_term(words[i:i + 2])
            and not is_known_term([word, words[-1]])
        )
        if strippable:
            continue
        # A leading adverb left behind by a stripped participle is not a noun.
        # Hyphenated units are exempt here too ("eight-day journey" tags as an
        # adverb but is a unit).
        if (not kept and not last and tok.pos_ == "ADV"
                and "-" not in word):
            continue
        kept.append(word)

    return " ".join(kept) if kept else phrase


def verify_with_gemini(rewrites, args):
    """Drop rewrites where the participle turns out to belong to the name."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from clean_nouns import Gemini

    if not args.api_key:
        sys.exit("--verify needs an API key: --api-key or GEMINI_API_KEY")

    verdicts = {}
    if os.path.exists(args.verify_cache):
        with open(args.verify_cache, encoding="utf-8") as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                verdicts[rec["phrase"]] = rec["v"]
        print(f"  verify cache: {len(verdicts):,} phrases")

    todo = [p for p in rewrites if p not in verdicts]
    batches = [todo[i:i + 60] for i in range(0, len(todo), 60)]
    print(f"  verifying {len(todo):,} candidates in {len(batches):,} batches "
          f"(~{len(batches) / max(args.rpm, 1):.1f} min)", flush=True)

    if batches:
        gem = Gemini(args.api_key, args.rpm, system_prompt=VERIFY_PROMPT)
        done = {"n": 0}
        with open(args.verify_cache, "a", encoding="utf-8") as cache_fh:
            def work(batch):
                digits = gem.judge(batch)
                for phrase, digit in zip(batch, digits):
                    cache_fh.write(json.dumps({"phrase": phrase,
                                               "v": digit}) + "\n")
                cache_fh.flush()
                done["n"] += 1
                if done["n"] % 50 == 0:
                    print(f"    {done['n']:,}/{len(batches):,}", flush=True)
                return batch, digits

            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                for batch, digits in pool.map(work, batches):
                    for phrase, digit in zip(batch, digits):
                        verdicts[phrase] = digit

    # "1" means the participle is part of the name: leave the phrase alone.
    protected = [p for p in rewrites if verdicts.get(p) == "1"]
    print(f"  Gemini protected {len(protected):,} established terms "
          f"({len(protected) / max(len(rewrites), 1):.1%} of candidates)")
    random.seed(0)
    if protected:
        print("    e.g. " + " | ".join(random.sample(protected,
                                                     min(12, len(protected)))))
    return {p: n for p, n in rewrites.items() if verdicts.get(p) != "1"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/ghana-nouns.csv")
    ap.add_argument("--output", default="data/ghana-nouns.csv")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--processes", type=int, default=1,
                    help="spaCy workers; each one copies the model, so more "
                         "processes need more RAM")
    ap.add_argument("--verify", action="store_true",
                    help="ask Gemini whether each participle belongs to the "
                         "name, so established terms are not stripped")
    ap.add_argument("--verify-cache", default="data/.qualifier-verdicts.jsonl")
    ap.add_argument("--rpm", type=int, default=300)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--api-key", default=os.environ.get("GEMINI_API_KEY"))
    args = ap.parse_args()

    # Two streaming passes. Materialising 519k rows as dicts alongside the
    # tagger needs several GB, which this machine does not have.
    print("pass 1: reading phrases", flush=True)
    with open(args.input, newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        phrase_col = header.index("phrase")
        phrases = [row[phrase_col] for row in reader]
    print(f"  {len(phrases):,} phrases")

    nlp = spacy.load("en_core_web_sm", exclude=["parser", "ner", "lemmatizer"])
    nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)

    print("pass 1: tagging", flush=True)
    rewrites = {}          # only phrases that actually change
    for i, (p, doc) in enumerate(
        zip(phrases, nlp.pipe(phrases, batch_size=1000,
                              n_process=args.processes))
    ):
        new = strip_qualifiers(p, doc)
        if new != p:
            rewrites[p] = new
        if (i + 1) % 100_000 == 0:
            print(f"  tagged {i + 1:,}", flush=True)
    print(f"  {len(rewrites):,} phrases have a qualifier to strip")

    if args.verify and rewrites:
        rewrites = verify_with_gemini(rewrites, args)

    existing = set(phrases)
    del phrases

    print("pass 2: rewriting rows", flush=True)
    changes, examples, claimed = Counter(), [], set()
    out = None if args.dry_run else open(args.output + ".tmp", "w",
                                        newline="", encoding="utf-8")
    writer = csv.writer(out, lineterminator="\n") if out else None
    if writer:
        writer.writerow(header)

    n_in = n_out = 0
    with open(args.input, newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        next(reader)
        for row in reader:
            n_in += 1
            old = row[phrase_col]
            new = rewrites.get(old, old)
            if new != old:
                if new in existing or new in claimed:
                    # The noun part is already in the dataset: drop as duplicate.
                    changes["deduped_away"] += 1
                    if len(examples) < 400:
                        examples.append((old, new, "dedup"))
                    continue
                changes["rewritten"] += 1
                if len(examples) < 400:
                    examples.append((old, new, "rewrite"))
                row[phrase_col] = new
                claimed.add(new)
            else:
                claimed.add(old)
            n_out += 1
            if writer:
                writer.writerow(row)
    if out:
        out.close()

    print(f"\nqualifier stripped from {sum(changes.values()):,} phrases:")
    print(f"  rewritten to noun part  {changes['rewritten']:>8,}")
    print(f"  removed as duplicate    {changes['deduped_away']:>8,}")
    print(f"\n{n_in:,} rows -> {n_out:,} rows")

    random.seed(0)
    for kind in ("rewrite", "dedup"):
        sample = [e for e in examples if e[2] == kind]
        print(f"\n--- {kind} examples ---")
        for old, new, _ in random.sample(sample, min(15, len(sample))):
            print(f"  {old:44s} -> {new}")

    if args.dry_run:
        print("\n(dry run: nothing written)")
        return

    # Written to a temp file first so a failure mid-write cannot truncate the
    # dataset, which is also the input.
    os.replace(args.output + ".tmp", args.output)
    print(f"\nwrote {n_out:,} rows -> {args.output}")


if __name__ == "__main__":
    main()
