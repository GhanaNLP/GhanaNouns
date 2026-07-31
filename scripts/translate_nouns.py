#!/usr/bin/env python3
"""Translate each noun into Twi, Ewe, Ga and Dagbani, three variants each.

Uses gemini-3.6-flash with a response schema that pins the exact JSON shape,
so a batch always comes back with one object per input noun and three variants
per language - no parsing guesswork and no silent misalignment.

Progress is checkpointed per phrase to --cache after every batch, so an
interrupted run resumes instead of re-paying for work already done.

Usage:
  python3 scripts/translate_nouns.py --limit 400      # pilot
  python3 scripts/translate_nouns.py                  # full run
  python3 scripts/translate_nouns.py --csv-only       # rebuild CSV from cache
"""

import argparse
import csv
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import httpx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from clean_nouns import RateLimiter  # noqa: E402

MODEL = "gemini-3.6-flash"
ENDPOINT = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent"
# Reasoning adds latency and tokens without helping a translation lookup.
THINKING_LEVEL = "minimal"

LANGUAGES = ["twi", "ewe", "ga", "dagbani"]
VARIANTS = 3

SYSTEM_PROMPT = """You are a translator for Ghanaian languages. For each \
English noun given, provide three natural alternative ways to say it in each \
of Twi (Asante Twi), Ewe, Ga and Dagbani.

Rules:
- Give three DISTINCT variants per language: different wording, dialectal
  choice or synonym. Never repeat the same string twice in one list.
- Use each language's standard orthography, including diacritics and special
  characters.
- Translate the meaning as a fluent speaker would say it, not word by word.
- If the term has no native equivalent, give the naturalised or borrowed form
  as it is actually spoken.
- Keep the "phrase" field exactly as it was given to you.

The input is a JSON array of English nouns. Return ONLY JSON in exactly this
structure, with one object per input noun, in the same order as the input:

{"translations":[{"phrase":"<the english noun>","twi":["variant 1","variant 2","variant 3"],"ewe":["variant 1","variant 2","variant 3"],"ga":["variant 1","variant 2","variant 3"],"dagbani":["variant 1","variant 2","variant 3"]}]}"""

_LANG_SCHEMA = {"type": "ARRAY", "minItems": VARIANTS, "maxItems": VARIANTS,
                "items": {"type": "STRING"}}


def response_schema(n):
    """Pin the array to exactly n objects, each with 3 variants per language."""
    return {
        "type": "OBJECT",
        "required": ["translations"],
        "properties": {
            "translations": {
                "type": "ARRAY", "minItems": n, "maxItems": n,
                "items": {
                    "type": "OBJECT",
                    "required": ["phrase"] + LANGUAGES,
                    "properties": dict({"phrase": {"type": "STRING"}},
                                       **{L: _LANG_SCHEMA for L in LANGUAGES}),
                },
            }
        },
    }


class Translator:
    def __init__(self, api_key, rpm, timeout=300.0):
        self.api_key = api_key
        self.limiter = RateLimiter(rpm)
        self.client = httpx.Client(timeout=timeout)

    def _call(self, phrases):
        body = {
            "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            "contents": [{"role": "user", "parts": [
                {"text": json.dumps(phrases, ensure_ascii=False)}]}],
            "generationConfig": {
                "temperature": 0.3,
                "maxOutputTokens": 300 * len(phrases) + 1024,
                "thinkingConfig": {"thinkingLevel": THINKING_LEVEL},
                "responseMimeType": "application/json",
                "responseSchema": response_schema(len(phrases)),
            },
        }
        self.limiter.acquire()
        r = self.client.post(ENDPOINT, params={"key": self.api_key}, json=body,
                             headers={"Content-Type": "application/json"})
        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
        data = r.json()
        cands = data.get("candidates") or []
        if not cands:
            raise RuntimeError(f"no candidates: {json.dumps(data)[:200]}")
        finish = cands[0].get("finishReason")
        parts = cands[0].get("content", {}).get("parts") or []
        text = "".join(p.get("text", "") for p in parts)
        if not text:
            raise RuntimeError(f"empty response (finish={finish})")
        out = json.loads(text)["translations"]
        if len(out) != len(phrases):
            raise ValueError(f"got {len(out)} objects for {len(phrases)} nouns")
        # Trust position, but flag a mismatch: if the echoed phrase differs the
        # model has re-ordered or invented rows and the batch is unusable.
        for phrase, obj in zip(phrases, out):
            if obj.get("phrase", "").strip().lower() != phrase.strip().lower():
                raise ValueError(f"phrase mismatch: {phrase!r} vs "
                                 f"{obj.get('phrase')!r}")
            obj["phrase"] = phrase
        return out

    def translate(self, phrases, attempts=4, depth=0):
        """Return one translation object per phrase, splitting on failure."""
        last = None
        for attempt in range(attempts):
            try:
                return self._call(phrases)
            except Exception as exc:  # noqa: BLE001 - retry everything
                last = exc
                msg = str(exc)
                transient = any(s in msg for s in (
                    "429", "500", "502", "503", "504", "timeout", "timed out",
                    "Connection", "no candidates", "RESOURCE_EXHAUSTED",
                    "UNAVAILABLE", "empty response"))
                if transient and attempt < attempts - 1:
                    time.sleep(min(2 ** attempt * 2, 30))
                    continue
                break
        if len(phrases) > 1 and depth < 8:
            mid = len(phrases) // 2
            return (self.translate(phrases[:mid], attempts, depth + 1)
                    + self.translate(phrases[mid:], attempts, depth + 1))
        print(f"  ! giving up on {phrases!r}: {last}", file=sys.stderr)
        return []


def cached_phrases(path):
    """Set of phrases already in the cache. Only the keys are held in memory -
    keeping every translation resident costs about a gigabyte at full size."""
    done = set()
    if not os.path.exists(path):
        return done
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue  # partial final line from a killed run
            if rec.get("phrase"):
                done.add(rec["phrase"])
    return done


def write_csv(cache_path, wanted, path):
    """Stream the cache into the CSV, skipping duplicates and unwanted rows."""
    cols = ["phrase"] + [f"{L}_{i}" for L in LANGUAGES
                         for i in range(1, VARIANTS + 1)]
    seen, n = set(), 0
    with open(path, "w", newline="", encoding="utf-8") as out, \
            open(cache_path, encoding="utf-8") as fh:
        w = csv.writer(out, lineterminator="\n")
        w.writerow(cols)
        for line in fh:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            p = rec.get("phrase")
            if not p or p in seen or p not in wanted:
                continue
            seen.add(p)
            row = [p]
            for L in LANGUAGES:
                vals = [str(v) for v in (rec.get(L) or [])][:VARIANTS]
                vals += [""] * (VARIANTS - len(vals))
                row += vals
            w.writerow(row)
            n += 1
    return n, cols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/ghana-nouns.csv")
    ap.add_argument("--output", default="data/ghana-nouns-translated.csv")
    ap.add_argument("--cache", default="data/.translations.jsonl")
    ap.add_argument("--batch-size", type=int, default=20)
    ap.add_argument("--rpm", type=int, default=300)
    ap.add_argument("--workers", type=int, default=100,
                    help="a batch takes ~20-40s, so many concurrent requests "
                         "are needed to actually reach --rpm")
    ap.add_argument("--limit", type=int, default=0, help="pilot on first N nouns")
    ap.add_argument("--csv-only", action="store_true",
                    help="rebuild the CSV from the cache without any API calls")
    ap.add_argument("--api-key", default=os.environ.get("GEMINI_API_KEY"))
    args = ap.parse_args()

    with open(args.input, newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        col = header.index("phrase")
        phrases = [row[col] for row in reader]
    if args.limit:
        phrases = phrases[:args.limit]
    print(f"{len(phrases):,} nouns to translate into "
          f"{', '.join(LANGUAGES)} x {VARIANTS} variants")

    done = cached_phrases(args.cache)
    if done:
        print(f"cache: {len(done):,} already translated")

    if not args.csv_only:
        if not args.api_key:
            sys.exit("no API key: pass --api-key or set GEMINI_API_KEY")

        todo = [p for p in phrases if p not in done]
        batches = [todo[i:i + args.batch_size]
                   for i in range(0, len(todo), args.batch_size)]
        print(f"{len(todo):,} remaining -> {len(batches):,} batches of "
              f"{args.batch_size} at {args.rpm} rpm "
              f"(~{len(batches) / max(args.rpm, 1):.0f} min)\n", flush=True)

        tr = Translator(args.api_key, args.rpm)
        lock = threading.Lock()
        state = {"n": 0, "ok": 0, "lost": 0}
        started = time.monotonic()
        cache_fh = open(args.cache, "a", encoding="utf-8")

        def work(batch):
            out = tr.translate(batch)
            with lock:
                for obj in out:
                    cache_fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
                cache_fh.flush()
                state["n"] += 1
                state["ok"] += len(out)
                state["lost"] += len(batch) - len(out)
                if state["n"] % 25 == 0 or state["n"] == len(batches):
                    el = time.monotonic() - started
                    rate = state["ok"] / el * 60
                    left = (len(todo) - state["ok"]) / max(rate, 1e-9)
                    print(f"  {state['n']:,}/{len(batches):,} batches, "
                          f"{state['ok']:,} nouns ({rate:.0f}/min, "
                          f"~{left:.0f} min left, {state['lost']} lost)",
                          flush=True)
            return out

        try:
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                # Results are consumed for their side effect on the cache only;
                # holding them all in memory would exhaust it at full size.
                for _ in pool.map(work, batches):
                    pass
        except KeyboardInterrupt:
            print("\ninterrupted - cache holds everything done so far",
                  file=sys.stderr)
        finally:
            cache_fh.close()
        print(f"\ntranslated {state['ok']:,}; {state['lost']:,} could not be "
              f"translated")

    if not os.path.exists(args.cache):
        sys.exit("nothing translated yet: no cache to build a CSV from")
    n, cols = write_csv(args.cache, set(phrases), args.output)
    print(f"wrote {n:,} rows x {len(cols)} columns -> {args.output}")
    missing = len(phrases) - n
    if missing:
        print(f"{missing:,} nouns still untranslated - rerun to resume")


if __name__ == "__main__":
    main()
