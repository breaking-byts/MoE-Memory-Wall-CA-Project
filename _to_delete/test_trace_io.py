"""Tests for expert-trace I/O and D4 correlation statistics.

The oracle for trace parsing is a faithful port of LLMSimulator's C++ reader
(src/scheduler/scheduler.cpp:560-608, Sum branch at :570-583). Any Python
loader or writer that disagrees with it will silently produce a trace the
simulator mis-reads -- which is exactly the bug this suite exists to catch.

Run: ./.venv/bin/python -m pytest test_trace_io.py -v
  or ./.venv/bin/python test_trace_io.py
"""
import os
import runpy
import shutil
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TRACE = os.path.join(HERE, "LLMSimulator/expert_data/experts_mixtral_stmoe.csv")

sys.path.insert(0, HERE)
import analysis_d4  # noqa: E402

N_EXPERT, TOPK, N_LAYER = 8, 2, 32


# --------------------------------------------------------------------------
# Oracle: the C++ parser, ported verbatim.
# --------------------------------------------------------------------------
def cpp_parse(pairs, lin, num_layer):
    """Port of scheduler.cpp Sum branch (scheduler.cpp:570-583).

    C++ reads line-by-line with a running idx:
        token_id = idx % Lin;  layer_id = idx / Lin;
        expert_list[token_id * num_layer + layer_id] = {top_1, top_2}
    so file row `idx` must be (token=idx%Lin, layer=idx/Lin).
    The Gen branch instead push_backs in file order (scheduler.cpp:585), so
    decode rows are read token-major and need no reshape.
    """
    expert_list = [None] * (lin * num_layer)
    for idx, pair in enumerate(pairs):
        token_id = idx % lin
        layer_id = idx // lin
        expert_list[token_id * num_layer + layer_id] = pair
    arr = np.zeros((lin, num_layer, TOPK), dtype=int)
    for tok in range(lin):
        for lay in range(num_layer):
            arr[tok, lay] = expert_list[tok * num_layer + lay]
    return arr


def read_csv_blocks(path):
    """Split a trace CSV into [(lin, [prefill_pairs], [decode_pairs]), ...]."""
    seqs, cur = [], None
    with open(path) as f:
        for line in f:
            p = line.split()
            if not p:
                continue
            if p[0].startswith("seq_"):
                cur = {"lin": None, "prefill": [], "decode": [], "section": None}
                seqs.append(cur)
            elif cur is None:
                continue
            elif p[0] == "Lin":
                cur["lin"] = int(p[1])
            elif p[0] == "Lout":
                pass
            elif p[0] == "Sum":
                cur["section"] = "prefill"
            elif p[0] == "Gen":
                cur["section"] = "decode"
            else:
                cur[cur["section"]].append((int(p[0]), int(p[1])))
    return seqs


def cpp_read_trace(path):
    """Full (N_TOK, N_LAYER, TOPK) array as the simulator would see it."""
    blocks = []
    for s in read_csv_blocks(path):
        pre = cpp_parse(s["prefill"], s["lin"], N_LAYER)
        post = np.array(s["decode"], dtype=int).reshape(-1, N_LAYER, TOPK)
        blocks.append(np.concatenate([pre, post], axis=0))
    return np.concatenate(blocks, axis=0)


def gen_trace_truth():
    """Re-run gen_trace.py's RNG and return its in-memory (N_SEQ, T, L, K) array.

    Runs inside a sandbox so the committed trace on disk is untouched.
    """
    tmp = _sandbox()
    cwd = os.getcwd()
    try:
        os.chdir(tmp)
        g = runpy.run_path(os.path.join(tmp, "gen_trace.py"))
    finally:
        os.chdir(cwd)
        shutil.rmtree(tmp, ignore_errors=True)
    seqs, lin, lout = g["seqs"], g["LIN"], g["LOUT"]
    arr = np.zeros((len(seqs), lin + lout, N_LAYER, TOPK), dtype=int)
    for s, toks in enumerate(seqs):
        for t in range(lin + lout):
            arr[s, t] = toks[t]
    return arr


# --------------------------------------------------------------------------
def overlap(arr):
    """1-step predictor overlap: next layer's top-2 == current layer's top-2."""
    hits = sum(len(set(arr[t, l]) & set(arr[t, l + 1]))
               for t in range(arr.shape[0]) for l in range(N_LAYER - 1))
    return hits / (arr.shape[0] * (N_LAYER - 1) * TOPK)


def distinct_per_layer(arr, batch):
    """Mean distinct experts touched per layer by a batch of `batch` tokens."""
    n_tok = arr.shape[0] // batch * batch
    x = arr[:n_tok].reshape(-1, batch, N_LAYER, TOPK)
    return float(np.mean([len(set(x[b, :, l, :].ravel()))
                          for b in range(x.shape[0]) for l in range(N_LAYER)]))


def _sandbox():
    """Temp dir with gen_trace.py + analysis_d4.py available and a writable
    LLMSimulator/expert_data, so we never clobber the committed trace."""
    tmp = tempfile.mkdtemp()
    for name in ("gen_trace.py", "analysis_d4.py"):
        os.symlink(os.path.join(HERE, name), os.path.join(tmp, name))
    os.makedirs(os.path.join(tmp, "LLMSimulator/expert_data"))
    return tmp


def _run_gen_trace(cwd):
    proc = subprocess.run([sys.executable, "gen_trace.py"], cwd=cwd,
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    return os.path.join(cwd, "LLMSimulator/expert_data/experts_mixtral_stmoe.csv")


def test_writer_output_parses_to_the_truth_under_the_cpp_parser():
    """gen_trace.py's CSV must round-trip to its own in-memory array.

    This is the check that catches a transposed writer: the Python loader in
    analysis_d4 and the writer in gen_trace.py must not share a convention
    that the C++ parser disagrees with.
    """
    truth = gen_trace_truth()
    tmp = _sandbox()
    try:
        path = _run_gen_trace(tmp)

        blocks = read_csv_blocks(path)
        assert len(blocks) == truth.shape[0], "seq count mismatch"
        assert all(b["lin"] > 0 for b in blocks), "every seq_N block needs a Lin"

        got = cpp_read_trace(path)
        n_seq, t_per_seq = truth.shape[0], truth.shape[1]
        assert got.shape[0] == n_seq * t_per_seq, \
            f"token count {got.shape[0]} != {n_seq * t_per_seq}"

        mismatch = []
        for s in range(n_seq):
            got_s = got[s * t_per_seq:(s + 1) * t_per_seq]
            if not np.array_equal(got_s, truth[s]):
                bad = int((got_s != truth[s]).any(axis=2).sum())
                mismatch.append((s, bad))
        assert not mismatch, (
            f"{len(mismatch)}/{n_seq} seq blocks differ from the generator's "
            f"in-memory array; first few: {mismatch[:5]}. "
            "The Sum section is written in the wrong order for "
            "scheduler.cpp:577-579 (token=idx%Lin, layer=idx/Lin)."
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_python_loader_agrees_with_the_cpp_parser():
    """analysis_d4.load_trace_csv must read traces the same way the sim does."""
    tmp = _sandbox()
    try:
        path = _run_gen_trace(tmp)
        got = analysis_d4.load_trace_csv(path)
        want = cpp_read_trace(path)
        assert got.shape == want.shape, f"shape {got.shape} != {want.shape}"
        assert np.array_equal(got, want), (
            "load_trace_csv disagrees with the C++ parser on "
            f"{int((got != want).any(axis=2).sum())} token-layers"
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_stats_ignores_the_module_level_token_count():
    """stats() must measure the array it is given.

    The bug this guards: stats() and plot() iterated `range(N_TOK)` -- a
    module-level constant meant for the synthetic generators -- so loading a
    5120-token trace silently measured only its first 2000 tokens (39%), all of
    them prefill. Shrinking N_TOK below the trace length makes any such
    truncation show up as a wrong answer.
    """
    rng = np.random.default_rng(3)
    small = analysis_d4.gen_correlated(rng, n_tok=64)
    # Identical halves, so the correct answer for the whole array is the same
    # as for one half -- any deviation is pure truncation.
    big = np.concatenate([small, small], axis=0)

    ov_small = analysis_d4.stats(small)[3]

    saved = analysis_d4.N_TOK
    analysis_d4.N_TOK = 10          # far below big.shape[0]
    try:
        ov_big = analysis_d4.stats(big)[3]
    finally:
        analysis_d4.N_TOK = saved

    assert abs(ov_big - ov_small) < 0.05, \
        f"overlap({ov_big:.3f}) does not reflect all {big.shape[0]} tokens " \
        f"(expected ~{ov_small:.3f}); stats() is truncating to N_TOK"


def test_persistence_is_reported_for_the_decode_phase():
    """Token persistence is the temporal-across-tokens metric; decode is the
    phase the prefetcher cares about, so stats() must surface it."""
    rng = np.random.default_rng(11)
    trace = analysis_d4.gen_correlated(rng, n_tok=400, persist=0.9)
    out = analysis_d4.stats(trace)
    names = analysis_d4.stats.__doc__ or ""
    del out, names
    persist = analysis_d4.stats(trace)[4]
    assert isinstance(persist, dict), \
        f"stats() persist must be a per-phase mapping, got {type(persist).__name__}"
    assert {"prefill", "decode"} <= set(persist), \
        f"persist must cover prefill and decode, got {sorted(persist)}"
    assert persist["decode"] > 0.5, \
        f"decode persistence {persist['decode']:.3f} should reflect persist=0.9"


def test_coverage_saturates_and_reports_distinct_experts_per_layer():
    """Coverage (distinct experts/layer at batch 8) drives the prefetcher's
    traffic argument, so it must be computed by code, not by hand."""
    rng = np.random.default_rng(5)
    uniform = analysis_d4.gen_uniform(rng, n_tok=512)
    cov = analysis_d4.coverage(uniform, batch=8)
    assert 1.0 <= cov["mean_distinct"] <= N_EXPERT
    assert abs(cov["mean_distinct"] - distinct_per_layer(uniform, 8)) < 1e-9
    # 8 tokens x top-2 = 16 draws over 8 experts -> most experts touched.
    assert cov["mean_distinct"] > 6.0, \
        f"coverage {cov['mean_distinct']:.2f} too low to demonstrate saturation"


def test_zipf_skew_alone_produces_overlap_above_the_random_null():
    """Guards the guide's 'skew != predictability' claim.

    The generator must expose skew and correlation as independent knobs so the
    ablation can be run; if pure skew already yields high overlap, the claim
    cannot be made.
    """
    rng = np.random.default_rng(7)
    skew_only = analysis_d4.gen_skew_only(rng, n_tok=1024, zipf_s=1.0)
    _, _, _, ov, _ = analysis_d4.stats(skew_only)
    assert ov > analysis_d4.RANDOM_OVERLAP, \
        f"skew-only overlap {ov:.3f} <= null {analysis_d4.RANDOM_OVERLAP}"


def test_skew_only_and_correlated_are_distinguishable():
    """The ablation must show correlation adds signal on top of skew."""
    a = analysis_d4.stats(
        analysis_d4.gen_skew_only(np.random.default_rng(7), n_tok=1024, zipf_s=1.0))[3]
    b = analysis_d4.stats(
        analysis_d4.gen_correlated(np.random.default_rng(7), n_tok=1024, stay=0.6))[3]
    assert b - a > 0.15, \
        f"correlated ({b:.3f}) must beat skew-only ({a:.3f}) by >0.15"


def test_uniform_generator_matches_real_routing_and_hits_its_null():
    """Real MoE routing picks top_k DISTINCT experts (std::set in
    sequence.cpp:190-197). gen_uniform must do the same, and then the 0.25
    analytic null must actually hold."""
    rng = np.random.default_rng(2)
    trace = analysis_d4.gen_uniform(rng, n_tok=4000)
    for t in range(trace.shape[0]):
        for l in range(N_LAYER):
            assert trace[t, l, 0] != trace[t, l, 1], "top-2 must be distinct"
    ov = analysis_d4.stats(trace)[3]
    assert abs(ov - analysis_d4.RANDOM_OVERLAP) < 0.05, \
        f"uniform overlap {ov:.3f} should sit at the {analysis_d4.RANDOM_OVERLAP} null"


if __name__ == "__main__":
    fns = [(n, f) for n, f in sorted(globals().items())
           if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in fns:
        try:
            fn()
        except AssertionError as e:
            failed += 1
            print(f"FAIL {name}\n     {e}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"ERROR {name}\n     {type(e).__name__}: {e}")
        else:
            print(f"ok   {name}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
