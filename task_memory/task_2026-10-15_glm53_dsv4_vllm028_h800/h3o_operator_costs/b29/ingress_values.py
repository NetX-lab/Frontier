"""Check B29's driver values against the retained printed measurement means."""
from decimal import Decimal
from pathlib import Path
import re
import sys

A10 = Path(__file__).resolve().parent.parent / "a10"
TERMS = (
    ("client_dispatch_to_proxy", "remainder/reference_split_stdout.txt", "client dispatch -> proxy arrival", "n 379", "0.6"),
    ("proxy_to_api", "k19/ingress_split_wave-20261030-01.txt", "proxy_arrival->api_arrival", "n  128", "3.11"),
    ("api_to_add_enqueue", "k19/ingress_split_wave-20261030-01.txt", "api_arrival->add_enqueue", "n  128", "6.59"),
    ("prefill_update_to_response", "remainder/remainder_stdout.txt", "p_update->prefill_response", "n 128", "2.3"),
    ("prefill_response_to_decode_enqueue", "remainder/pairs_stdout.txt", "prefill_response->d_add_enqueue", "n 128", "9.1"),
)


def main():
    values = []
    for name, file, span, population, quoted in TERMS:
        matches = [(i, line) for i, line in enumerate((A10 / file).read_text().splitlines(), 1)
                   if span in line and population in line]
        if len(matches) != 1:
            raise ValueError(f"Missing or ambiguous measurement {name}: {A10 / file}: {matches}")
        number, line = matches[0]
        mean = Decimal(re.search(r"mean\s+([0-9.]+)", line).group(1))
        values.append(mean)
        print(f"{name} = {mean} ms; a10/{file}:{number}: {line}")
        print(f"  difference from quoted {quoted} ms: {mean - Decimal(quoted)} ms")
    ingress, handoff = sum(values[:3]), sum(values[3:])
    assert ingress == Decimal("10.3")
    assert handoff == Decimal("11.4")
    sys.path.insert(0, str(A10))
    from a10_driver import REQUEST_INGRESS_DELAY_MS, REQUEST_HANDOFF_DELAY_MS
    assert REQUEST_INGRESS_DELAY_MS == float(ingress)
    assert REQUEST_HANDOFF_DELAY_MS == float(handoff)
    print(f"request_ingress_delay_ms = {ingress} ms")
    print(f"request_handoff_delay_ms = {handoff} ms")
    print("P2-32 shares P2-64 values: no add_enqueue capture, same engine build/proxy/API server (A10 section 5.4).")


if __name__ == "__main__":
    main()
