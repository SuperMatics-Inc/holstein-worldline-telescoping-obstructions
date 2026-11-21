# Build and verification record

Version 1.0.0, 21 November 2026. This record covers the released bytes of this repository: the
manuscript `paper/main.pdf`, its sources and the replay supplement. Every digest below was
computed on the released files themselves with Python's `hashlib`, and `MANIFEST.sha256` plus
`replay/SHA256SUMS` pin the same bytes.

## Released bytes

| File | Size (bytes) | SHA-256 |
| --- | --- | --- |
| `paper/main.tex` | 84536 | `c1eb13f13ee826280361b1fa98487030c0e6e49b12579b36ad37a71cea70de9b` |
| `paper/refs.bib` | 5542 | `0594895e946ba594ea23120f20a39908aeb12fbaf25f34e32c42250f5bbd0690` |
| `paper/main.pdf` | 256283 | `52a80ea0b6b837a0286eeb467a4ad1b5af9afcaaf0601192fb4a1f5cb4d14dc9` |
| `paper/CLAIMS.md` | 7724 | `24b0233a6a86d6d8ac113e2c97bee43576798b9bef4ee478ba1548a4b4e249c8` |
| `replay/replay-charged-skeleton.py` | 33731 | `4fe93b2bd0fdf3c11a75aef7cbe415ed9a4015a8a682a38c00adba5c65d8368d` |
| `replay/replay-telescope.py` | 17046 | `800d6e12d196de3f585ad7d0c295b3be20006292eaeddc56b951e919d95b094b` |
| `replay/replay-annealed-spectral.py` | 19892 | `d33e788fe95ed1139f34372b50e0c6c879e2f41eee734db598f494b00d74e8fd` |
| `replay/replay-annealed-endpoint.py` | 58475 | `d013cda90dde118358b489e382a587bfc1d740c0586427f6057fd827d2582595` |
| `replay/replay-neutral-dipole.py` | 70552 | `e4f07397205b0343686369f0665c4648ed289eeaffe16aa0683adbe933aeea6a` |
| `replay/run_all.py` | 2621 | `45647e15f8b6a38cbb2593341fda6b9fc6350671172e2cb71a37ec1657acde7b` |
| `replay/README.md` | 10590 | `9a6bc0303bcf2e452cb273038850b181aa23dd7c05b9aac99b2e6ea1de9f9ce6` |
| `replay/SHA256SUMS` | 611 | `205ae07333ad8892ccbc1bfcfaa24f73c9c441d39ddff431f02228430874eced` |

The PDF has 20 pages. The MD5 of `paper/main.pdf` is `ff836f2076f5ae64d4d86a22330b074d`. All
files are UTF-8 or binary as appropriate and use LF line endings. The five replay scripts,
`run_all.py`, `replay/README.md` and `replay/SHA256SUMS` are byte copies of the reviewed
supplement (`sha256sum -c --strict SHA256SUMS` gives 7/7 OK); the manuscript sources and the PDF
are the reviewed bytes unchanged.

## Manuscript toolchain

- Compiler: Tectonic 0.17.0 (release asset
  `tectonic-0.17.0-x86_64-unknown-linux-musl.tar.gz`, SHA-256
  `8533d07f9ccbd7a65824b9e0459041bca34af1eb33daba48f59215593753a3b7`, downloaded and checked
  for this record), driving its bundled XeTeX engine and `xdvipdfmx`; no local TeX installation
  is used. The released PDF was produced by the same version on Windows and reproduced byte for
  byte by continuous integration on Linux.
- Document class `article`, 11 pt, margin 1 in, with `geometry`, `amsmath`, `amssymb`, `amsthm`,
  `mathtools`, `hyperref` (`hidelinks,pdfusetitle`), `microtype`, `enumitem`, `array` and
  `booktabs`, all from the bundled distribution.
- Bibliography: `plain` style, `refs.bib`, run by Tectonic's own BibTeX pass.
- Fixed build epoch: `SOURCE_DATE_EPOCH=1795132800` (2026-11-20T00:00:00Z), so that the PDF
  carries no build time; its declared creation date is `D:20261120000000-00'00'`.
- PDF metadata: title equal to the paper title and author `Davis Rens, SuperMatics Inc., ORCID
  0000-0001-5839-5197`, both set by `pdfusetitle` from the manuscript.

Recipe, from the `paper/` directory:

```sh
SOURCE_DATE_EPOCH=1795132800 tectonic -X compile --keep-logs main.tex
```

Observed result: exit code 0; the log holds 0 `Overfull` lines, 0 `undefined` references or
citations, 0 `Warning` lines, and exactly 2 `Underfull` lines, which are two short paragraphs of
the reviewed source and not defects of the build:

```
Underfull \hbox (badness 1189) in paragraph at lines 875--880
Underfull \hbox (badness 1466) in paragraph at lines 1349--1355
```

A second run from the same inputs reproduces the PDF byte for byte, digest
`52a80ea0b6b837a0286eeb467a4ad1b5af9afcaaf0601192fb4a1f5cb4d14dc9`. The released PDF is the
reviewed build: the review's findings were applied to `paper/main.tex` before it, and the
released source, bibliography and PDF are the reviewed bytes unchanged.

## Replay

The five scripts need an interpreter of Python 3.12 or later with NumPy, SciPy, mpmath and
sympy. From the `replay/` directory of an unpacked release:

```sh
python -B replay-charged-skeleton.py                 # exit 0, ALL CHARGED-SKELETON CHECKS PASSED (EXIT 0)
python -B replay-charged-skeleton.py --mutants        # exit 0, ALL 8 MUTANTS CAUGHT WITH EXIT 1; DRIVER EXIT 0
python -B replay-telescope.py                        # exit 0, ALL_CHECKS_PASSED
python -B replay-telescope.py --mutants               # exit 0, MUTANTS all 7/7 caught with exit!=0; driver exit=0
python -B replay-annealed-spectral.py                # exit 0, REPLAY SUCCESS
python -B replay-annealed-spectral.py --mutants       # exit 0, ALL 7/7 MUTANTS CAUGHT
python -B replay-annealed-endpoint.py                # exit 0, ALL REPLAY CHECKS PASSED.
python -B replay-annealed-endpoint.py --mutants       # exit 0, ALL 13 MUTANTS KILLED (exit 0).
python -B replay-annealed-endpoint.py --tables         # exit 0, the V = 6 Hubbard-limit row of Table 2
python -B replay-neutral-dipole.py                   # exit 0, [PASS] Step 5C: 4-site ring worldline MC ...
python -B replay-neutral-dipole.py --mutants          # exit 0, Mutation summary: 10/10 mutants caught
python -B run_all.py                                 # exit 0, SUMMARY: 5/5 scripts passed in ...s
```

Elapsed times vary by more than an order of magnitude with the machine and its load (about six
minutes for the whole battery on one laptop core); the closing text does not. `replay/README.md`
maps every script to the paper section it checks, states the approximate single-threaded times
and lists the same closing lines.

## Continuous integration

`.github/workflows/ci.yml` has two jobs, both on Python 3.12.

The `replay` job installs NumPy, SciPy, mpmath and sympy with `pip`, checks the digests of the
released sources and scripts, verifies `MANIFEST.sha256` and `replay/SHA256SUMS`, checks that the
manifest lists every tracked file except itself, builds the release archive with `git archive`
from the commit, unpacks it, and runs all five scripts in plain mode and with `--mutants`,
`replay-annealed-endpoint.py` again with `--tables`, and `run_all.py`, on the unpacked copy,
requiring each documented closing line.

The `paper` job installs the pinned Tectonic release after checking the digest of its archive,
checks the digests of the released sources and PDF, rebuilds the manuscript with the epoch above,
requires zero overfull boxes, exactly the two documented underfull boxes, zero undefined
references and no warning in the log, requires the rebuilt PDF to equal the released PDF byte for
byte, and checks the PDF metadata and the markers of `ci/verify_text.py`, including the
disclosure sentence and its single occurrence.

## Byte custody

```sh
sha256sum -c MANIFEST.sha256
(cd replay && sha256sum -c SHA256SUMS)
```

`MANIFEST.sha256` lists every tracked file except itself; `replay/SHA256SUMS` lists the seven
other files of the replay directory. Both lists were compared against the released tree with
Python's `hashlib` and report zero mismatches, and continuous integration reruns `sha256sum -c`
on both. The deposited supplement archive is `git archive` of the release tag, so its members
are exactly the files listed by `git ls-files` at that commit.

## Custody notes

The released bytes are copies of the reviewed files. Two files carry descriptive working labels
rather than paths: `replay/run_all.py` names the paper's position in its series in the first line
of its module docstring, and `paper/CLAIMS.md` is the release version of the internal claim map,
rewritten so that it cites the manuscript anchors and the replay scripts instead of internal
packet and review files. No released file contains a filesystem path, a lane or agent name, or a
review, note, fix or backup file, and none reads a label at run time.

## Scope

The scripts check finite arithmetic, in exact rational or outward-interval arithmetic, with
finite-volume exact diagonalization, determinant quantum Monte Carlo and worldline Monte Carlo.
They are not certificates for any thermodynamic statement, and passing them proves none of the
analytic steps of the paper. No pairing ceiling and no bound on any critical temperature is
proved or disproved, and no released result changes status.
