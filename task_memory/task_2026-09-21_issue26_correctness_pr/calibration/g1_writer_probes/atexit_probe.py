import atexit, multiprocessing, os, signal, sys, time
OUT = sys.argv[1]

def child(tag, handle_term):
    atexit.register(lambda: open(OUT, "a").write(f"{tag} atexit ran\n"))
    if handle_term:
        def handler(signum, frame):
            raise SystemExit()
        signal.signal(signal.SIGTERM, handler)
    try:
        time.sleep(30 if tag.endswith("term") else 0.1)
    except SystemExit:
        pass

if __name__ == "__main__":
    open(OUT, "w").close()
    for method in ("fork", "spawn"):
        ctx = multiprocessing.get_context(method)
        for tag, handle in (("return", False), ("term_default_term", False), ("term_handled_term", True)):
            p = ctx.Process(target=child, args=(f"{method}:{tag}", handle))
            p.start()
            if "term" in tag:
                time.sleep(1.0)
                p.terminate()
            p.join()
    print(open(OUT).read() or "(no atexit ran)")
