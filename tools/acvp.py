#!/usr/bin/env python3
"""Convert NIST ACVP ML-DSA vectors into ParallelPQ's key = value vector files.

usage: tools/acvp.py ACVP_SERVER_CHECKOUT [--out-dir tests/vectors/acvp]

Reads gen-val/json-files/ML-DSA-{sigVer,keyGen}-FIPS204/internalProjection.json from a checkout
of https://github.com/usnistgov/ACVP-Server and writes mldsa_sigver.txt and mldsa_keygen.txt.

Pre-hash (HashML-DSA) groups are dropped: ParallelPQ verifies pure ML-DSA only. Each sigVer
vector gains the verdict ParallelPQ must report, from decoding the signature here (FIPS 204
Algorithms 21 and 27): bad-hint if HintBitUnpack fails, z-norm if ||z|| >= gamma1 - beta,
otherwise pass or bad-commitment from ACVP's testPassed.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

# id: (k, l, c-tilde bytes, gamma1 bits, omega, beta)
PARAMS = {
    44: (4, 4, 32, 17, 80, 78),
    65: (6, 5, 48, 19, 55, 196),
    87: (8, 7, 64, 19, 75, 120),
}
N = 256

HEADER_NOTE = "# converted by tools/acvp.py; do not edit by hand\n"


def param_id(name):
    if not name.startswith("ML-DSA-") or int(name[7:]) not in PARAMS:
        raise ValueError(f"unknown parameter set {name!r}")
    return int(name[7:])


def hint_valid(y, k, omega):
    """HintBitUnpack (Algorithm 21) succeeds"""
    index = 0
    for i in range(k):
        if y[omega + i] < index or y[omega + i] > omega:
            return False
        first = index
        while index < y[omega + i]:
            if index > first and y[index - 1] >= y[index]:
                return False
            index += 1
    return all(b == 0 for b in y[index:omega])


def z_max_norm(sig, l, ctilde, gamma1_bits):
    """max |z| over the decoded z polynomials (Algorithm 27 with BitUnpack)"""
    z_bits = gamma1_bits + 1
    poly_bytes = N * z_bits // 8
    gamma1 = 1 << gamma1_bits
    mask = (1 << z_bits) - 1
    worst = 0
    for j in range(l):
        chunk = sig[ctilde + j * poly_bytes : ctilde + (j + 1) * poly_bytes]
        v = int.from_bytes(chunk, "little")
        for i in range(N):
            z = gamma1 - ((v >> (i * z_bits)) & mask)
            worst = max(worst, abs(z))
    return worst


def verdict(pid, sig, passed):
    k, l, ctilde, gamma1_bits, omega, beta = PARAMS[pid]
    hint = sig[len(sig) - omega - k :]
    if not hint_valid(hint, k, omega):
        v = "bad-hint"
    elif z_max_norm(sig, l, ctilde, gamma1_bits) >= (1 << gamma1_bits) - beta:
        v = "z-norm"
    else:
        v = "pass" if passed else "bad-commitment"
    if passed and v != "pass":
        raise ValueError(f"a passing vector decodes as {v}")
    return v


def load_projection(root, mode):
    path = root / "gen-val" / "json-files" / f"ML-DSA-{mode}-FIPS204" / "internalProjection.json"
    with open(path) as f:
        return json.load(f), path.relative_to(root)


def source_commit(root):
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def write_records(path, header, records):
    with open(path, "w") as f:
        f.write(header)
        for rec in records:
            f.write("\n")
            for key, value in rec:
                value = str(value)
                f.write(f"{key} = {value}\n" if value else f"{key} =\n")


def convert_sigver(root, commit):
    data, rel = load_projection(root, "sigVer")
    records = []
    counts = {}

    for group in data["testGroups"]:
        if group.get("preHash") == "preHash":
            continue
        pid = param_id(group["parameterSet"])
        if group["signatureInterface"] == "external":
            interface = "external"
        elif group.get("externalMu"):
            interface = "mu"
        else:
            interface = "internal"

        for t in group["tests"]:
            sig = bytes.fromhex(t["signature"])
            v = verdict(pid, sig, t["testPassed"])
            rec = [
                ("tcId", t["tcId"]),
                ("params", pid),
                ("interface", interface),
                ("passed", int(t["testPassed"])),
                ("verdict", v),
                ("reason", t["reason"]),
                ("pk", t["pk"].lower()),
            ]
            if interface == "mu":
                rec.append(("mu", t["mu"].lower()))
            else:
                rec.append(("msg", t["message"].lower()))
            if interface == "external":
                rec.append(("ctx", t["context"].lower()))
            rec.append(("sig", t["signature"].lower()))
            records.append(rec)
            counts[(interface, v)] = counts.get((interface, v), 0) + 1

    header = (
        "# ML-DSA signature verification vectors (FIPS 204) from NIST ACVP\n"
        f"# source: usnistgov/ACVP-Server {commit}\n"
        f"#         {rel}\n"
        + HEADER_NOTE
        + "#\n"
        "# interface: external = ML-DSA.Verify(pk, msg, sig, ctx), pure (no pre-hash)\n"
        "#            internal = ML-DSA.Verify_internal(pk, msg, sig); msg is M' itself\n"
        "#            mu       = Verify_internal given mu directly (ACVP externalMu)\n"
        "# verdict:   what ParallelPQ reports: pass, bad-hint, z-norm or bad-commitment\n"
    )
    return header, records, counts


def convert_keygen(root, commit):
    data, rel = load_projection(root, "keyGen")
    records = []
    for group in data["testGroups"]:
        pid = param_id(group["parameterSet"])
        for t in group["tests"]:
            records.append(
                [
                    ("tcId", t["tcId"]),
                    ("params", pid),
                    ("seed", t["seed"].lower()),
                    ("pk", t["pk"].lower()),
                    ("sk", t["sk"].lower()),
                ]
            )
    header = (
        "# ML-DSA.KeyGen_internal(seed) vectors (FIPS 204) from NIST ACVP\n"
        f"# source: usnistgov/ACVP-Server {commit}\n"
        f"#         {rel}\n" + HEADER_NOTE
    )
    return header, records


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("acvp_server", type=Path, help="checkout of usnistgov/ACVP-Server")
    ap.add_argument("--out-dir", type=Path, default=here.parent / "tests" / "vectors" / "acvp")
    args = ap.parse_args()

    root = args.acvp_server.resolve()
    commit = source_commit(root)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    header, records, counts = convert_sigver(root, commit)
    write_records(args.out_dir / "mldsa_sigver.txt", header, records)
    print(f"mldsa_sigver.txt: {len(records)} vectors")
    for (interface, v), n in sorted(counts.items()):
        print(f"  {interface:9} {v:15} {n}")

    header, records = convert_keygen(root, commit)
    write_records(args.out_dir / "mldsa_keygen.txt", header, records)
    print(f"mldsa_keygen.txt: {len(records)} vectors")
    return 0


if __name__ == "__main__":
    sys.exit(main())
