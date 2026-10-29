import csv
import sys

for path in sys.argv[1:]:
    print("==", path.split("G3H_")[1])
    for row in csv.DictReader(open(path)):
        print("  rank {ep_rank:>2} L {gating_num_tokens:>4} tokens {num_tokens:>6} local {local_routed_rows:>5} | "
              "gemm n={c} k={k} median {m:.4f} ms | gate n={gc} k={gk} {g:.4f} ms".format(
                  c=row["time_stats.moe_grouped_gemm.count"], k=row["time_stats.moe_grouped_gemm.kernel_count"],
                  m=float(row["time_stats.moe_grouped_gemm.median"]),
                  gc=row["time_stats.moe_gating_linear.count"], gk=row["time_stats.moe_gating_linear.kernel_count"],
                  g=float(row["time_stats.moe_gating_linear.median"]), **row))
