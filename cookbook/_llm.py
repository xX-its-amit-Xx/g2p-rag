"""Tiny 3-tier LLM fallback adapter for the g2p-rag cookbook.

Why this module exists
----------------------
Cookbook scripts want a single, uniform ``llm(prompt, max_tokens) -> str``
callable so the retrieval-grounded synthesis step does not need to special-case
which backend is reachable on a given machine. We try three backends in order:

  1. **Anthropic** — if ``ANTHROPIC_API_KEY`` is set, use the official
     ``anthropic`` SDK with ``claude-sonnet-4-6`` (overridable).
  2. **Local Llama (llama.cpp)** — if ``llama_cpp`` is importable AND a GGUF
     model file exists at ``LLAMA_MODEL_PATH`` (or the default
     ``C:/llama-models/Llama-3.2-3B-Instruct-Q4_K_M.gguf``), load it on CPU
     with a modest context window.
  3. **NoOpLLM** — a sentinel that returns a clear diagnostic string instead
     of crashing. Cookbook scripts are expected to print their retrieval
     results in a structured way first and only call the LLM for a final
     paraphrase; with NoOpLLM the retrieval-grounded output is still useful,
     just without the prose summary.

Public API
----------
  - ``get_llm(model_anthropic, model_llama_path, verbose) -> Callable``
  - ``class NoOpLLM``
  - ``is_llm_available() -> bool``

Import safety
-------------
Heavy SDKs (``anthropic``, ``llama_cpp``) are imported lazily inside the
relevant branch of ``get_llm`` so this module is safe to import on a fresh
clone without those packages installed. Only the ``get_llm()`` call can raise,
and even there we fall through to ``NoOpLLM`` rather than propagating an
ImportError.
"""

from __future__ import annotations

import os
from typing import Callable, Optional

DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-6"
DEFAULT_LLAMA_PATH = "C:/llama-models/Llama-3.2-3B-Instruct-Q4_K_M.gguf"

NO_LLM_MESSAGE = (
    "[no LLM available; set ANTHROPIC_API_KEY or LLAMA_MODEL_PATH]"
)


# ---------------------------------------------------------------------------
# NoOpLLM sentinel
# ---------------------------------------------------------------------------

class NoOpLLM:
    """Sentinel LLM that returns a clear "no LLM available" diagnostic.

    Cookbook scripts that depend on an LLM should print retrieval results in
    a structured way (tables, bullet lists, citation tags) and ONLY call the
    LLM for the final paraphrase. With ``NoOpLLM`` in place, the
    retrieval-grounded synthesis is still fully usable; only the optional
    prose summary is replaced by the diagnostic message.
    """

    backend = "noop"

    def __call__(self, prompt: str, max_tokens: int = 512) -> str:  # noqa: ARG002
        return NO_LLM_MESSAGE

    def __repr__(self) -> str:
        return "NoOpLLM()"


# ---------------------------------------------------------------------------
# Backend probes (cheap, no heavy imports)
# ---------------------------------------------------------------------------

def _anthropic_available() -> bool:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return False
    try:
        import anthropic  # noqa: F401
    except Exception:
        return False
    return True


def _resolve_llama_path(model_llama_path: Optional[str]) -> Optional[str]:
    """Return a filesystem path to a GGUF model, or None if none is reachable."""
    candidates = []
    if model_llama_path:
        candidates.append(model_llama_path)
    env_path = os.environ.get("LLAMA_MODEL_PATH")
    if env_path:
        candidates.append(env_path)
    candidates.append(DEFAULT_LLAMA_PATH)
    for p in candidates:
        if p and os.path.isfile(p):
            return p
    return None


def _llama_available(model_llama_path: Optional[str] = None) -> bool:
    if _resolve_llama_path(model_llama_path) is None:
        return False
    try:
        import llama_cpp  # noqa: F401
    except Exception:
        return False
    return True


def is_llm_available() -> bool:
    """Return True if either Anthropic or local Llama is reachable.

    Useful for cookbook scripts that want to conditionally include an
    LLM-paraphrased section: ``if is_llm_available(): ...``.
    """
    return _anthropic_available() or _llama_available()


# ---------------------------------------------------------------------------
# Backend factories
# ---------------------------------------------------------------------------

def _make_anthropic_llm(model: str, verbose: bool) -> Callable[[str, int], str]:
    import anthropic  # lazy
    client = anthropic.Anthropic()  # picks up ANTHROPIC_API_KEY from env

    if verbose:
        print(f"[_llm] backend=anthropic model={model}")

    def _call(prompt: str, max_tokens: int = 512) -> str:
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        # response.content is a list of blocks; the first is the text block.
        block = response.content[0]
        text = getattr(block, "text", None)
        if text is None:
            # Defensive: stringify whatever we got back.
            text = str(block)
        return str(text)

    _call.backend = "anthropic"  # type: ignore[attr-defined]
    _call.model = model  # type: ignore[attr-defined]
    return _call


def _make_llama_llm(model_path: str, verbose: bool) -> Callable[[str, int], str]:
    from llama_cpp import Llama  # lazy

    cpu_count = os.cpu_count() or 2
    n_threads = max(1, cpu_count // 2)

    if verbose:
        print(
            f"[_llm] backend=llama model_path={model_path} "
            f"n_ctx=4096 n_threads={n_threads}"
        )

    llm = Llama(
        model_path=model_path,
        n_ctx=4096,
        n_threads=n_threads,
        verbose=False,
    )

    def _call(prompt: str, max_tokens: int = 512) -> str:
        out = llm(
            prompt,
            max_tokens=max_tokens,
        )
        # llama_cpp returns a dict with a "choices" list; first choice text is
        # the completion.
        try:
            text = out["choices"][0]["text"]
        except (KeyError, IndexError, TypeError):
            text = str(out)
        return str(text)

    _call.backend = "llama"  # type: ignore[attr-defined]
    _call.model_path = model_path  # type: ignore[attr-defined]
    return _call


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def get_llm(
    model_anthropic: str = DEFAULT_ANTHROPIC_MODEL,
    model_llama_path: Optional[str] = None,
    verbose: bool = True,
):
    """Return an LLM callable with a unified interface.

    The returned object is callable as ``llm(prompt: str, max_tokens: int = 512) -> str``.

    Fallback order:
      1. Anthropic (requires ``ANTHROPIC_API_KEY``)
      2. Local Llama via ``llama_cpp`` (requires a reachable GGUF file)
      3. ``NoOpLLM`` — returns the diagnostic string above

    Parameters
    ----------
    model_anthropic:
        Anthropic model ID to use when the Anthropic branch is selected.
    model_llama_path:
        Optional explicit path to a GGUF model file. If not given, falls back
        to ``$LLAMA_MODEL_PATH`` then to ``DEFAULT_LLAMA_PATH``.
    verbose:
        If True (default), print which backend is selected at construction time.
    """
    # 1. Anthropic
    if _anthropic_available():
        try:
            return _make_anthropic_llm(model_anthropic, verbose=verbose)
        except Exception as exc:  # pragma: no cover - defensive
            if verbose:
                print(f"[_llm] anthropic init failed: {exc!r}; trying llama")

    # 2. Local Llama
    llama_path = _resolve_llama_path(model_llama_path)
    if llama_path is not None:
        try:
            import llama_cpp  # noqa: F401  (lazy availability check)
            return _make_llama_llm(llama_path, verbose=verbose)
        except Exception as exc:  # pragma: no cover - defensive
            if verbose:
                print(f"[_llm] llama init failed: {exc!r}; falling back to NoOp")

    # 3. NoOp
    if verbose:
        print("[_llm] backend=noop (no Anthropic key and no reachable Llama model)")
    return NoOpLLM()


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== _llm.py self-test ===")
    print(f"is_llm_available(): {is_llm_available()}")
    llm = get_llm()
    backend = getattr(llm, "backend", "unknown")
    print(f"selected backend: {backend}")

    prompt = "Reply with the single word OK."
    print(f"prompt: {prompt!r}")
    response = llm(prompt, max_tokens=8)
    print(f"response: {response!r}")
    print("=== done ===")
