# Vigenere GPU Decoder

A Python 2.7-compatible, wordlist-based Vigenère cipher decoder with an optional PyCUDA scoring backend.

**Compatibility warning:** Python 2.7 is end-of-life, and legacy PyCUDA builds may not work with CUDA 12.4. The CPU backend is the dependable baseline. GPU support is experimental and reports import/runtime failures rather than pretending acceleration is active.

## Features

- Wordlist-based candidate-key search for standard Vigenère ciphertexts.
- Preserves punctuation, spaces, and letter case in plaintext output.
- Ranks candidates using a lightweight English-frequency heuristic.
- Optional PyCUDA kernel to score candidate keys in parallel.
- Python 2.7 syntax; CPU mode uses only the standard library.
- Example inputs and unit tests.

## Requirements

- Python 2.7.18 (legacy, unsupported upstream).
- GPU mode: a working PyCUDA build compatible with your Python, compiler, CUDA toolkit, and NVIDIA driver.
- An NVIDIA CUDA-capable GPU for GPU mode.

A GTX 1050 Ti is a target device, but that does not guarantee compatibility with Python 2.7 and CUDA 12.4. Read [GPU compatibility notes](docs/GPU_COMPATIBILITY.md).

## Quick start

Run the example using the CPU backend:

```bash
python2.7 vigenere_gpu_decoder.py --ciphertext "LXFOPVEFRNHR" --wordlist examples/wordlist.txt --top 10
```

Or read ciphertext from a file:

```bash
python2.7 vigenere_gpu_decoder.py --ciphertext-file examples/ciphertext.txt --wordlist examples/wordlist.txt --top 10
```

## GPU mode (experimental)

```bash
python2.7 vigenere_gpu_decoder.py --ciphertext "LXFOPVEFRNHR" --wordlist examples/wordlist.txt --backend gpu --top 10
```

GPU mode is opt-in. If PyCUDA cannot be imported or the CUDA kernel fails, the program exits with a clear error. Use `--backend cpu` for the reliable baseline. Confirm the program reports a successful GPU run before assuming acceleration is active.

## Tests

```bash
bash scripts/run_tests.sh
```

Or directly:

```bash
python2.7 -m unittest discover -s tests -v
```

## Limitations

- This is a heuristic decoder, not a guaranteed cryptanalysis solution.
- It expects standard Vigenère encryption over A-Z; non-letters are not encrypted.
- Wordlist quality strongly affects results.
- Short or unusual plaintexts may be ranked poorly by the simple scoring heuristic.
- Python 2.7 is obsolete; use an isolated environment and avoid processing untrusted files.

## License

MIT. See [LICENSE](LICENSE).
