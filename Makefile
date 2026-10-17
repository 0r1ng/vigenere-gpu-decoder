PYTHON ?= python2.7

.PHONY: test example

test:
	$(PYTHON) -m unittest discover -s tests -v

example:
	$(PYTHON) vigenere_gpu_decoder.py --ciphertext "LXFOPVEFRNHR" --wordlist examples/wordlist.txt --top 10
