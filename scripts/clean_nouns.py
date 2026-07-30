#!/usr/bin/env python3
"""Clean ghana-nouns.csv: drop entries that are not real standalone nouns.

Two stages:
  1. Rule prefilter  - removes high-confidence junk with zero API cost
                       (conjoined phrases, disfluencies, adverb qualifiers,
                       stray determiners, single-letter OCR debris, ligatures).
  2. Gemini pass      - everything left goes to gemini-3.5-flash-lite in
                       batches of 60; the model answers with a single integer
                       per phrase (1 = keep, 0 = drop), so output tokens are
                       about one per phrase. Scored 95.4% accuracy (97% of real
                       nouns kept, 94% of junk dropped) on a held-out dev set.

Writes data/ghana-nouns-clean.csv (kept) and data/ghana-nouns-rejected.csv
(dropped, with a reason column). Gemini verdicts are checkpointed to
--cache so an interrupted run resumes instead of re-paying for batches.

Usage:
  python3 scripts/clean_nouns.py --sample 2000     # trial run
  python3 scripts/clean_nouns.py                   # full run
  python3 scripts/clean_nouns.py --rules-only      # no API calls
"""

import argparse
import csv
import json
import os
import random
import re
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import httpx

MODEL = "gemini-3.5-flash-lite"
ENDPOINT = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent"
# Reasoning is not needed for this judgement and it dominated both latency and
# cost when left at the default.
THINKING_LEVEL = "low"
# Answers come back as a schema-constrained array of exactly N integers.
# Free-form digit strings drifted (59 or 62 answers for 60 terms), which would
# silently misalign verdicts against phrases.
BATCH_DEFAULT = 60

# ---------------------------------------------------------------- rule prefilter

CONJUNCTIONS = {"and", "or", "but", "nor", "plus", "versus", "vs"}

# Speech disfluencies and interjections - never part of a noun.
FILLERS = {
    "uh", "um", "erm", "hmm", "mmm", "mhm", "yeah", "yep", "yup", "okay", "ok",
    "eh", "ah", "oh", "huh", "hey", "wow", "oops", "alright", "right",
}

# Adverbs with no legitimate use inside a lexical noun. Deliberately excludes
# risky ones ("still", "only", "even", "now", "well", "just", "near") which can
# head real compounds (still water, only child, well water) - those go to Gemini.
ADVERBS = {
    "very", "really", "quite", "actually", "certainly", "usually", "particularly",
    "maybe", "perhaps", "probably", "possibly", "definitely", "simply", "merely",
    "always", "never", "often", "sometimes", "rather", "somewhat", "extremely",
    "totally", "completely", "absolutely", "basically", "literally", "apparently",
    "clearly", "obviously", "generally", "specifically", "especially", "mainly",
    "mostly", "largely", "fully", "partly", "partially", "previously", "formerly",
    "already", "likewise", "exactly", "virtually", "hardly", "occasionally",
    "increasingly", "relatively", "significantly", "substantially", "notably",
    "arguably", "ultimately", "eventually", "finally", "initially", "originally",
    "essentially", "primarily", "purely", "solely", "seemingly", "supposedly",
    "allegedly", "highly", "newly", "recently", "currently", "moreover",
    "furthermore", "however", "therefore", "nevertheless", "nonetheless",
    "consequently", "accordingly", "meanwhile", "instead", "besides",
    # Manner adverbs. Note this list is explicit rather than a "-ly" test:
    # supply, family, assembly, monthly, friendly and costly all end in -ly
    # without being adverbs.
    "badly", "poorly", "wrongly", "closely", "widely", "heavily", "strongly",
    "deeply", "greatly", "slightly", "seriously", "severely", "reportedly",
    "jointly", "freshly", "locally", "nationally", "internationally",
    "globally", "politically", "economically", "socially", "culturally",
    "physically", "mentally", "legally", "formally", "informally",
    "directly", "indirectly", "publicly", "privately", "openly", "secretly",
    "quickly", "slowly", "rapidly", "gradually", "steadily", "constantly",
    "continuously", "regularly", "frequently", "rarely", "remotely",
    "successfully", "properly", "carefully", "deliberately", "unduly",
    "adequately", "efficiently", "effectively", "duly", "hugely", "fiercely",
}

# Determiners/possessives that mark a phrase as a fragment of running text.
DETERMINERS = {
    "the", "a", "an", "this", "that", "these", "those",
    "his", "her", "its", "their", "our", "your", "my",
}

# Prepositions that can legitimately appear inside a fixed noun phrase
# ("man of the year"). Their presence exempts a phrase from the determiner
# rule so Gemini can judge it instead.
PREPOSITIONS = {"of", "in", "on", "for", "at", "to", "with", "from"}

LIGATURES = "ﬀﬁﬂﬃﬄﬅﬆ"


def rule_verdict(phrase):
    """Return a rejection reason, or None to pass the phrase on to Gemini."""
    tokens = phrase.split()

    if any(ch in LIGATURES for ch in phrase):
        return "ocr_ligature"

    if any(len(t) == 1 and t.isalpha() for t in tokens):
        return "single_letter_debris"

    tokenset = set(tokens)

    if FILLERS & tokenset:
        return "disfluency"

    if ADVERBS & tokenset:
        return "adverb_qualifier"

    # Conjunction joining two nouns ("crop residue and fertilizer additions").
    if CONJUNCTIONS & set(tokens[1:-1]):
        return "conjoined_nouns"

    if DETERMINERS & tokenset and not (PREPOSITIONS & tokenset):
        return "stray_determiner"

    return None


# --------------------------------------------------------------- gemini prefilter

# Framed as fault-detection rather than "is this dictionary-worthy". The
# dictionary framing made the model reject valid but specific compounds
# (engine capacity, polling station), costing ~17% of real nouns.
SYSTEM_PROMPT = """Each term below was auto-extracted as a noun phrase from \
Ghanaian English text. Your only job is to spot extraction faults.

Answer 0 (DROP) only if the term has one of these four faults:
1. REMOVABLE QUALIFIER - an adjective, participle or adverb that merely \
describes the noun and can be deleted leaving a complete noun behind: \
newly graduated teachers, burnt electricity poles, politically motivated \
interference, remotely sensed bands, involved parties, suspected substance, \
revised form, reduced impact, traded papers, newly engaged workers
2. CONJOINED - two nouns joined by and/or: industry and police forces
3. NOT A NOUN - a verb, gerund or verb phrase: crave, achieving, shying, \
serves, haunted, actually work, location matters, politicising state \
logistics procurement
4. BROKEN TEXT - a misspelling, words run together, word salad, or an \
incoherent fragment: collition, withfull, ownviewis, completion completion, \
registration exercisewhich, ilack iof icommon igoals, jails educationist, \
obtainment officials, mood court facility, three-page ruling captures, \
commission six listed documents

Answer 1 (KEEP) for everything else.

Do NOT judge whether the term is common, important or dictionary-worthy. \
Rare, technical, bureaucratic and very specific terms are all valid. A noun \
modifying a noun is always valid, however long the chain, and the head word \
may be an abstract nominalisation: power supply shortage, power sector \
upgrades, master cylinder seals, equipment seizures, price swings, \
market strategy planning, programme selection, engine capacity, \
project webpage, quality mark, restless legs syndrome, mango road. Fixed \
adjective+noun names are valid: civil servant, only child, blood bank.

If a term shows none of the four faults, answer 1. When unsure, answer 1.

Answer with one integer (1 or 0) per term, in the same order as the input."""


class RateLimiter:
    """Simple sliding-window limiter: at most `rpm` acquisitions per minute."""

    def __init__(self, rpm):
        self.rpm = rpm
        self.times = []
        self.lock = threading.Lock()

    def acquire(self):
        while True:
            with self.lock:
                now = time.monotonic()
                self.times = [t for t in self.times if now - t < 60.0]
                if len(self.times) < self.rpm:
                    self.times.append(now)
                    return
                wait = 60.0 - (now - self.times[0]) + 0.01
            time.sleep(wait)


class Gemini:
    def __init__(self, api_key, rpm, timeout=120.0):
        self.api_key = api_key
        self.limiter = RateLimiter(rpm)
        self.client = httpx.Client(timeout=timeout)

    def _call(self, phrases):
        numbered = "\n".join(f"{i + 1}. {p}" for i, p in enumerate(phrases))
        body = {
            "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": f"{len(phrases)} terms. Answer 1 or 0 for "
                                 f"each, in order.\n{numbered}"}
                    ],
                }
            ],
            "generationConfig": {
                "temperature": 0,
                "maxOutputTokens": len(phrases) * 6 + 300,
                "thinkingConfig": {"thinkingLevel": THINKING_LEVEL},
                # A free-form digit string drifted by one or two answers on a
                # noticeable fraction of batches; pinning the array length makes
                # a verdict/phrase mismatch impossible.
                "responseMimeType": "application/json",
                "responseSchema": {
                    "type": "ARRAY",
                    "minItems": len(phrases),
                    "maxItems": len(phrases),
                    "items": {"type": "INTEGER"},
                },
            },
        }
        self.limiter.acquire()
        r = self.client.post(
            ENDPOINT, params={"key": self.api_key}, json=body,
            headers={"Content-Type": "application/json"},
        )
        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
        data = r.json()
        cands = data.get("candidates") or []
        if not cands:
            raise RuntimeError(f"no candidates: {json.dumps(data)[:300]}")
        parts = cands[0].get("content", {}).get("parts") or []
        answers = json.loads("".join(p.get("text", "") for p in parts))
        if len(answers) != len(phrases):
            raise ValueError(
                f"expected {len(phrases)} answers, got {len(answers)} "
                f"(finish={cands[0].get('finishReason')})"
            )
        return "".join("1" if a else "0" for a in answers)

    def judge(self, phrases, attempts=4, depth=0):
        """Return a digit string aligned to `phrases`, splitting on misalignment."""
        last = None
        for attempt in range(attempts):
            try:
                return self._call(phrases)
            except Exception as exc:  # noqa: BLE001 - retry everything
                last = exc
                msg = str(exc)
                transient = any(
                    s in msg for s in ("429", "500", "502", "503", "504", "timed out",
                                       "timeout", "Connection", "no candidates",
                                       "RESOURCE_EXHAUSTED", "UNAVAILABLE")
                )
                if transient and attempt < attempts - 1:
                    time.sleep(min(2 ** attempt * 2, 30))
                    continue
                break

        # Misalignment or persistent failure: halve the batch and retry.
        if len(phrases) > 1 and depth < 8:
            mid = len(phrases) // 2
            return (self.judge(phrases[:mid], attempts, depth + 1)
                    + self.judge(phrases[mid:], attempts, depth + 1))

        # Single phrase still failing: keep it rather than lose data silently.
        print(f"  ! keeping unjudged {phrases!r}: {last}", file=sys.stderr)
        return "1" * len(phrases)


# ------------------------------------------------------------------------ driver

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/ghana-nouns.csv")
    ap.add_argument("--clean-out", default="data/ghana-nouns-clean.csv")
    ap.add_argument("--rejected-out", default="data/ghana-nouns-rejected.csv")
    ap.add_argument("--cache", default="data/.gemini-verdicts.jsonl",
                    help="checkpoint of completed batches; enables resume")
    ap.add_argument("--batch-size", type=int, default=BATCH_DEFAULT)
    ap.add_argument("--rpm", type=int, default=300)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--sample", type=int, default=0,
                    help="judge only N randomly sampled phrases (trial run)")
    ap.add_argument("--rules-only", action="store_true",
                    help="skip Gemini; only apply the rule prefilter")
    ap.add_argument("--api-key", default=os.environ.get("GEMINI_API_KEY"))
    args = ap.parse_args()

    with open(args.input, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fields = reader.fieldnames
        rows = list(reader)
    print(f"loaded {len(rows):,} rows from {args.input}")

    # Stage 1: rules.
    reasons = {}
    for row in rows:
        reason = rule_verdict(row["phrase"])
        if reason:
            reasons[row["phrase"]] = reason
    counts = Counter(reasons.values())
    print(f"\nrule prefilter dropped {len(reasons):,}:")
    for reason, n in counts.most_common():
        print(f"  {reason:22s} {n:>8,}")

    todo = [r["phrase"] for r in rows if r["phrase"] not in reasons]
    # Dedupe while preserving order - identical phrases need judging once.
    seen, uniq = set(), []
    for p in todo:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    print(f"\n{len(uniq):,} unique phrases remain for Gemini")

    if args.sample and args.sample < len(uniq):
        order = {p: i for i, p in enumerate(uniq)}
        random.seed(42)
        uniq = sorted(random.sample(uniq, args.sample), key=order.__getitem__)
        print(f"sampling {len(uniq):,} of them (trial run)")
        # A sample batches differently from a full run, so verdicts must not
        # share a cache file with it.
        args.cache += f".sample{args.sample}"

    verdicts = {}
    if not args.rules_only:
        if not args.api_key:
            sys.exit("no API key: pass --api-key or set GEMINI_API_KEY")

        # Cache is keyed by phrase, not by batch index. Keying it by index
        # would make any rule or batch-size change shift every boundary and
        # discard all previously paid-for verdicts.
        if os.path.exists(args.cache):
            with open(args.cache, encoding="utf-8") as fh:
                for line in fh:
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    verdicts[rec["phrase"]] = rec["v"]
            print(f"cache: {len(verdicts):,} phrases already judged")

        pending = [p for p in uniq if p not in verdicts]
        batches = [pending[i:i + args.batch_size]
                   for i in range(0, len(pending), args.batch_size)]
        print(f"{len(pending):,} phrases to judge -> {len(batches):,} batches "
              f"of {args.batch_size} at {args.rpm} rpm "
              f"(~{len(batches) / max(args.rpm, 1):.1f} min)\n")

        gem = Gemini(args.api_key, args.rpm)
        cache_lock = threading.Lock()
        progress = {"n": 0}
        started = time.monotonic()
        cache_fh = open(args.cache, "a", encoding="utf-8")

        def work(phrases):
            digits = gem.judge(phrases)
            with cache_lock:
                for phrase, digit in zip(phrases, digits):
                    cache_fh.write(json.dumps({"phrase": phrase, "v": digit}) + "\n")
                cache_fh.flush()
                progress["n"] += 1
                if progress["n"] % 50 == 0 or progress["n"] == len(batches):
                    elapsed = time.monotonic() - started
                    rate = progress["n"] / elapsed * 60
                    left = (len(batches) - progress["n"]) / max(rate, 1e-9)
                    print(f"  {progress['n']:,}/{len(batches):,} batches "
                          f"({rate:.0f}/min, ~{left:.1f} min left)", flush=True)
            return phrases, digits

        if batches:
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                for phrases, digits in pool.map(work, batches):
                    for phrase, digit in zip(phrases, digits):
                        verdicts[phrase] = digit
        cache_fh.close()

        # Only phrases that survived the current rules get a model verdict
        # applied; cached verdicts for now-rule-dropped phrases are ignored.
        live = set(uniq)
        verdicts = {p: v for p, v in verdicts.items() if p in live}

        dropped = sum(1 for v in verdicts.values() if v == "0")
        print(f"\ngemini dropped {dropped:,} of {len(verdicts):,} judged "
              f"({dropped / max(len(verdicts), 1):.1%})")
        for phrase, digit in verdicts.items():
            if digit == "0":
                reasons[phrase] = "gemini_not_a_noun"

    # Write outputs.
    kept, rejected = [], []
    for row in rows:
        reason = reasons.get(row["phrase"])
        if reason:
            rejected.append({**row, "reject_reason": reason})
        else:
            kept.append(row)

    # lineterminator="\n": csv defaults to CRLF, but the source dataset is LF
    # and a wholesale line-ending flip would show up as a diff on every row.
    with open(args.clean_out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        w.writerows(kept)
    with open(args.rejected_out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields + ["reject_reason"],
                           lineterminator="\n")
        w.writeheader()
        w.writerows(rejected)

    print(f"\nkept     {len(kept):,} -> {args.clean_out}")
    print(f"rejected {len(rejected):,} -> {args.rejected_out}")
    for reason, n in Counter(reasons.values()).most_common():
        print(f"  {reason:22s} {n:>8,}")


if __name__ == "__main__":
    main()
