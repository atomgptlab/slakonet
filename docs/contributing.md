# Contributing

Contributions are welcome — bug reports, fixes, new examples,
documentation improvements, and feature work.

## Ways to help

- **Report bugs** — open a [GitHub issue](https://github.com/atomgptlab/slakonet/issues)
  with a minimal reproducer.
- **Improve the docs** — every page has an edit link (top right); fixes
  and clarifications are appreciated.
- **Add examples** — a clear, self-contained script for a new use case
  is one of the most useful contributions.
- **Submit code** — bug fixes and features via pull request.

## Development setup

```bash
git clone https://github.com/atomgptlab/slakonet.git
cd slakonet
conda create --name slakonet-dev python=3.10 -y
conda activate slakonet-dev
pip install -e .
```

## Building the documentation locally

The docs are built with [MkDocs](https://www.mkdocs.org/) and the
[Material](https://squidfunk.github.io/mkdocs-material/) theme:

```bash
pip install mkdocs-material
mkdocs serve
```

Then open <http://127.0.0.1:8000>. The site rebuilds live as you edit
files under `docs/`.

## Pull request guidelines

- Keep changes focused — one logical change per PR.
- Match the surrounding code style.
- Add or update an example or doc page when you add a feature.
- Make sure existing examples still run.
- Describe *why* the change is needed in the PR description.

## Reporting bugs effectively

A good bug report includes:

1. What you ran (a minimal code snippet).
2. What you expected.
3. What happened (full error / traceback).
4. Your environment — OS, Python, PyTorch and SlaKoNet versions, CPU/GPU.

## Code of conduct

Please be respectful and constructive in all project spaces. Assume good
intent and help newcomers.

## Questions

For usage questions, open a
[GitHub issue](https://github.com/atomgptlab/slakonet/issues) — chances
are someone else has the same question, and the answer then helps them
too.
