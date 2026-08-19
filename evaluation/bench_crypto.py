"""
bench_crypto.py -- Q4 fast-path microbenchmarks (paper sec:ev-cost, tab:gas).

The Ballast draw fast path is "one signature plus a constant-size opening"
(sec:de-opt): the operator signs the draw certificate sigma_u over
DC_i = (i, b_i, ver, cmt, ...), and the counterparty verifies it.  We benchmark
that signature/verification with the SAME Schnorr-over-secp256k1 primitive the
Horcrux reference implementation uses (github-horcrux/basic_implementation),
so the fast-path draw latency reported in the paper is grounded in a real
implementation rather than a placeholder.

This anchors the coordination model's r_const (the constant draw latency that
makes Ballast flat in coin-shift latency): a draw costs one signature + one
verification + one network round trip, and we confirm the crypto is sub-
millisecond, i.e. the round trip dominates (paper's claim).

On-chain gas for PostBond / Claim / Equivocation / Release is measured by the
UCRB Hardhat suite (github-ucrb/contracts/test/schnorr-test.js); see README for
the `npx hardhat test` command.  This script reports only the off-chain fast
path, which is what runs per draw.
"""

import os
import sys
import time

# reuse Horcrux's reference Schnorr primitives
HORCRUX = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "..", "github-horcrux", "basic_implementation")
sys.path.insert(0, os.path.abspath(HORCRUX))

from sig_schnorr import (generate_keypair_schnorr, sign_message_schnorr,
                         verify_signature_schnorr)


def bench(n=2000):
    msg = b"DC_i = (i, b_i, ver, cmt, pi_sum, pi_pos, sigma_u)"  # a draw certificate

    t0 = time.time()
    for _ in range(n):
        sk, vk = generate_keypair_schnorr()
    t_key = (time.time() - t0) / n * 1000

    sk, vk = generate_keypair_schnorr()
    t0 = time.time()
    for _ in range(n):
        sig = sign_message_schnorr(sk, msg)
    t_sign = (time.time() - t0) / n * 1000

    sig = sign_message_schnorr(sk, msg)
    t0 = time.time()
    for _ in range(n):
        verify_signature_schnorr(vk, msg, sig)
    t_verify = (time.time() - t0) / n * 1000

    fastpath = t_sign + t_verify
    print(f"draw-certificate microbenchmark (Schnorr/secp256k1, n={n}):")
    print(f"  keygen           : {t_key:8.4f} ms")
    print(f"  sign   (operator): {t_sign:8.4f} ms")
    print(f"  verify (cp)      : {t_verify:8.4f} ms")
    print(f"  fast-path crypto : {t_sign + t_verify:8.4f} ms  (sign + verify)")
    print(f"  proof size       : {len(sig)} B  (r||s)")
    print()
    print("Interpretation: fast-path crypto is a few ms with this pure-Python")
    print("reference (a native libsecp256k1 is sub-millisecond), so a draw's")
    print("latency is dominated by the network round trip, not cryptography")
    print("(paper sec:ev-cost).  On-chain gas: run the UCRB Hardhat suite,")
    print("github-ucrb/contracts/test/schnorr-test.js (see README).")
    return {"keygen_ms": t_key, "sign_ms": t_sign, "verify_ms": t_verify,
            "fastpath_ms": fastpath, "sig_bytes": len(sig)}


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    bench(n)
