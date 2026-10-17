# -*- coding: utf-8 -*-
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import vigenere_gpu_decoder as decoder


class VigenereDecoderTests(unittest.TestCase):
    def test_decrypt_classic_example(self):
        self.assertEqual(
            decoder.decrypt_vigenere("LXFOPVEFRNHR", "LEMON"),
            "ATTACKATDAWN"
        )

    def test_preserves_case_and_punctuation(self):
        self.assertEqual(
            decoder.decrypt_vigenere("Lxfopv, Ef Rnhr!", "Lemon"),
            "Attack, At Dawn!"
        )

    def test_empty_key_is_rejected(self):
        self.assertRaises(ValueError, decoder.decrypt_vigenere, "ABC", "")

    def test_score_returns_number(self):
        self.assertTrue(isinstance(decoder.english_score("THIS IS A TEST"), (int, float)))


if __name__ == "__main__":
    unittest.main()
