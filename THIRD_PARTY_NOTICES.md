# Third-party notices

This project is Apache-2.0. It includes the following third-party material, used under the
licence shown.

## typesafe-ai/system-one-adapter-python

- Source: https://github.com/typesafe-ai/system-one-adapter-python, commit `fb52b10`
  (`src/system_one_adapter/_utils/confidence_metrics.py` and
  `tests/utils/test_confidence_metrics.py`).
- Used in: `_choice_confidence`, `_score_confidence` and `_normalize_probs` in
  `src/granite_switch_decision/plugin.py` (adapted), and the confidence test vectors in
  `tests/test_plugin.py`.

```
MIT License

Copyright (c) 2026 TypeSafe AI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Obtained at test time, not included

- **OpenAI's OpenAPI specification** (`openapi.json`, https://github.com/openai/openai-openapi,
  MIT, Copyright (c) OpenAI) is downloaded by `tests/specs.py` at a pinned commit and
  checked against a pinned hash. It is not part of this repository.
- **typesafe-sdk** (https://pypi.org/project/typesafe-sdk/, MIT) is a dev dependency whose
  generated wire models supply the jev schemas for `tests/test_systemone_spec.py`. It is
  installed, not copied.
- **TypeSafe's API specification** (https://api.typesafe.ai/openapi.json) carries no licence.
  It is not used, downloaded or included.
- **BoolQ** (https://huggingface.co/datasets/google/boolq, CC BY-SA 3.0, Google) is fetched
  from Hugging Face's datasets server each time `scripts/accuracy_check.py` runs. No rows are
  stored in this repository or shipped in the package.
- **Decision Index** (https://github.com/apolinario/decision-index, MIT) is a benchmark kit
  that a user may install and run against a server, as described in `docs/TESTING.md`. It is not
  part of this repository or the package. The benchmark suite it builds is made from third-party
  datasets under their own licences and is never stored here.
