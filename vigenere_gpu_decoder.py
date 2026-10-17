#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""Wordlist-based Vigenere decoder for Python 2.7; PyCUDA backend is optional."""
from __future__ import print_function

import argparse
import os
import sys
import string

ALPHABET = string.ascii_uppercase
# Lightweight frequency weights; heuristic only, not a language model.
LETTER_WEIGHTS = [8.2, 1.5, 2.8, 4.3, 12.7, 2.2, 2.0, 6.1, 7.0,
                  0.2, 0.8, 4.0, 2.4, 6.7, 7.5, 1.9, 0.1, 6.0,
                  6.3, 9.1, 2.8, 1.0, 2.4, 0.2, 2.0, 0.1]


def _key_shifts(key):
    shifts = []
    for char in key.upper():
        if char in ALPHABET:
            shifts.append(ord(char) - ord("A"))
    if not shifts:
        raise ValueError("key must contain at least one letter A-Z")
    return shifts


def decrypt_vigenere(ciphertext, key):
    """Decrypt A-Z letters; preserve punctuation, spaces, and letter case."""
    shifts = _key_shifts(key)
    output = []
    key_index = 0
    for char in ciphertext:
        upper = char.upper()
        if upper in ALPHABET:
            cipher_value = ord(upper) - ord("A")
            plain_value = (cipher_value - shifts[key_index % len(shifts)]) % 26
            plain_char = chr(ord("A") + plain_value)
            if char.islower():
                plain_char = plain_char.lower()
            output.append(plain_char)
            key_index += 1
        else:
            output.append(char)
    return "".join(output)


def english_score(text):
    """Return a simple English-likeness score; higher is better."""
    letters = [c.upper() for c in text if c.upper() in ALPHABET]
    if not letters:
        return float("-inf")
    score = 0.0
    for char in letters:
        score += LETTER_WEIGHTS[ord(char) - ord("A")]
    # Common English bigrams and trigrams add a small ranking bonus.
    upper = text.upper()
    for token in ("TH", "HE", "IN", "ER", "AN", "RE", "ON", "AT",
                  "EN", "ND", "THE", "AND", "ING", "ION"):
        score += upper.count(token) * (2.0 if len(token) == 2 else 3.0)
    # Penalize very rare letters slightly; this is intentionally basic.
    for char in ("J", "Q", "X", "Z"):
        score -= letters.count(char) * 1.5
    return score / float(len(letters))


def load_wordlist(path):
    words = []
    with open(path, "rb") as handle:
        for line in handle:
            word = line.strip()
            if not word or word.startswith("#"):
                continue
            try:
                word = word.decode("utf-8")
            except AttributeError:
                pass
            except UnicodeDecodeError:
                word = word.decode("latin-1")
            if any(char.upper() in ALPHABET for char in word):
                words.append(word)
    # Remove duplicate keys case-insensitively while preserving order.
    seen = set()
    unique = []
    for word in words:
        normalized = word.upper()
        if normalized not in seen:
            seen.add(normalized)
            unique.append(word)
    return unique


def score_candidates_cpu(ciphertext, keys):
    results = []
    for key in keys:
        try:
            plaintext = decrypt_vigenere(ciphertext, key)
            results.append((english_score(plaintext), key, plaintext))
        except ValueError:
            continue
    results.sort(key=lambda item: item[0], reverse=True)
    return results


def score_candidates_gpu(ciphertext, keys):
    """Use a CUDA kernel to score candidate keys; requires a working PyCUDA."""
    try:
        import numpy
        import pycuda.autoinit
        import pycuda.driver as cuda
        from pycuda.compiler import SourceModule
    except Exception as exc:
        raise RuntimeError("PyCUDA GPU backend unavailable: %s" % exc)

    # Filter to ASCII keys because the CUDA kernel operates on bytes.
    ascii_keys = []
    for key in keys:
        try:
            encoded = key.upper().encode("ascii")
        except UnicodeEncodeError:
            continue
        encoded = "".join([c for c in encoded if c in string.ascii_uppercase])
        if encoded:
            ascii_keys.append(encoded)
    if not ascii_keys:
        return []

    try:
        cipher = ciphertext.upper().encode("ascii")
    except UnicodeEncodeError:
        raise RuntimeError("GPU backend currently requires ASCII ciphertext")

    cipher = "".join([c for c in cipher if c in string.ascii_uppercase])
    if not cipher:
        return []

    max_key_len = max(len(key) for key in ascii_keys)
    key_matrix = numpy.zeros((len(ascii_keys), max_key_len), dtype=numpy.uint8)
    key_lengths = numpy.zeros(len(ascii_keys), dtype=numpy.int32)
    for row, key in enumerate(ascii_keys):
        key_lengths[row] = len(key)
        for col, char in enumerate(key):
            key_matrix[row, col] = ord(char)

    # Each thread scores one candidate key. The kernel mirrors the simple
    # frequency component of english_score; CPU ranking adds n-gram bonuses.
    source = r"""
    __device__ float weight(int c) {
      const float w[26] = {8.2f,1.5f,2.8f,4.3f,12.7f,2.2f,2.0f,6.1f,7.0f,
        0.2f,0.8f,4.0f,2.4f,6.7f,7.5f,1.9f,0.1f,6.0f,6.3f,9.1f,2.8f,1.0f,2.4f,0.2f,2.0f,0.1f};
      return w[c];
    }
    __global__ void score_keys(const unsigned char *cipher, int cipher_len,
      const unsigned char *keys, const int *key_lengths, int key_width,
      int count, float *scores) {
      int idx = blockIdx.x * blockDim.x + threadIdx.x;
      if (idx >= count) return;
      int key_len = key_lengths[idx];
      if (key_len <= 0) { scores[idx] = -1.0e20f; return; }
      float total = 0.0f;
      for (int i = 0; i < cipher_len; ++i) {
        int c = (int)cipher[i] - 65;
        int k = (int)keys[idx * key_width + (i % key_len)] - 65;
        int p = (c - k + 26) % 26;
        total += weight(p);
        if (p == 9 || p == 16 || p == 23 || p == 25) total -= 1.5f;
      }
      scores[idx] = total / (float)cipher_len;
    }
    """
    try:
        module = SourceModule(source)
        kernel = module.get_function("score_keys")
        scores = numpy.zeros(len(ascii_keys), dtype=numpy.float32)
        kernel(cuda.In(cipher), numpy.int32(len(cipher)),
              cuda.In(key_matrix), cuda.In(key_lengths),
              numpy.int32(max_key_len), numpy.int32(len(ascii_keys)),
              cuda.Out(scores), block=(128, 1, 1),
              grid=((len(ascii_keys) + 127) // 128, 1))
    except Exception as exc:
        raise RuntimeError("PyCUDA kernel execution failed: %s" % exc)

    # Decrypt on CPU and use the full heuristic for final ordering.
    candidates = []
    for index, key in enumerate(ascii_keys):
        plain = decrypt_vigenere(ciphertext, key)
        candidates.append((float(scores[index]), key, plain))
    candidates.sort(key=lambda item: item[0], reverse=True)
    # Refine the shortlist with the same CPU scorer, including n-gram bonuses.
    shortlist = [item[1] for item in candidates[:min(100, len(candidates))]]
    return score_candidates_cpu(ciphertext, shortlist)


def read_ciphertext(args):
    if args.ciphertext is not None:
        return args.ciphertext
    if args.ciphertext_file:
        with open(args.ciphertext_file, "rb") as handle:
            data = handle.read()
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError:
            return data.decode("latin-1")
    return None


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Wordlist-based Vigenere decoder (Python 2.7; optional PyCUDA)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--ciphertext", help="ciphertext string")
    group.add_argument("--ciphertext-file", help="read ciphertext from a file")
    parser.add_argument("--wordlist", required=True, help="one candidate key per line")
    parser.add_argument("--top", type=int, default=10, help="number of results to print")
    parser.add_argument("--backend", choices=("cpu", "gpu"), default="cpu",
                        help="CPU is default; GPU requires compatible PyCUDA")
    args = parser.parse_args(argv)

    if args.top < 1:
        parser.error("--top must be at least 1")
    if not os.path.isfile(args.wordlist):
        parser.error("wordlist file does not exist: %s" % args.wordlist)

    ciphertext = read_ciphertext(args)
    keys = load_wordlist(args.wordlist)
    if not keys:
        parser.error("wordlist contains no usable keys")

    if args.backend == "gpu":
        print("Backend requested: GPU (PyCUDA).")
        try:
            results = score_candidates_gpu(ciphertext, keys)
        except RuntimeError as exc:
            print("ERROR: %s" % exc, file=sys.stderr)
            return 2
    else:
        print("Backend: CPU")
        results = score_candidates_cpu(ciphertext, keys)

    for rank, result in enumerate(results[:args.top], 1):
        score, key, plaintext = result
        print("%2d. score=%8.3f  key=%s  plaintext=%s" %
              (rank, score, key, plaintext))
    return 0


if __name__ == "__main__":
    sys.exit(main())
