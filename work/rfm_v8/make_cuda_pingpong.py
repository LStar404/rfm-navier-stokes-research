"""Generate a parity-buffer pilot from the frozen prepared CUDA march."""

import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "rfm_v7" / "galerkin_cuda_prepared.py"
TARGET = HERE / "galerkin_cuda_pingpong.py"
MANIFEST = HERE / "cuda_pingpong_generation.json"


def main():
    if TARGET.exists() or MANIFEST.exists():
        raise FileExistsError("Preserve the existing parity-buffer generation")
    raw = SOURCE.read_bytes()
    text = raw.decode("utf-8")
    span = text[text.index('        multiply = rt.launch("products"'):text.index('        rt.sync()\n        march =')]
    new = '''        # n=1 writes transformed, and the other buffer starts at the exact zero F_0.
        history = previous, transformed
        multiply = rt.launch("products", q*n, physical, transformed, products, previous, q*n, 0)
        observe = rt.launch("observe_absolute", 1, state, amps, index, slots, p_slots, proj, pl,
                            saved, p_saved, frame, 1)
        first = rt.launch("observe_absolute", 1, state, amps, index, slots, p_slots, proj, pl,
                          saved, p_saved, frame, 0)

        def advances(pack):
            return tuple(rt.launch("advance", frame, state, amps, index, history[p], history[1-p],
                                   *pack, w, k, q, half) for p in range(2))

        old_advance, late_advance = advances(old_pack), advances(late_pack)
        bridge_advance = tuple(bridge_launch(rt, frame, state, amps, index, history[p], history[1-p],
                                             bridge_pack, w, k, q, half, coefficients) for p in range(2))

        def body(advance, parity):
            form()
            call(rt.z2d, inverse, spectrum, physical)
            multiply()
            call(rt.d2z, forward, products, history[parity])
            advance[parity]()
            observe()

        call(rt.z2d, inverse, spectrum, physical)
        call(rt.d2z, forward, products, transformed)
        first()
        rt.sync()
        plan = time.perf_counter()-t
        t = time.perf_counter()
        before_steps, after_steps = split-1, steps-split-1

        def capture(advance, first_index, count):
            chunk = min(graph_steps, count)
            # Replay begins on the same parity only when the captured block is even.
            chunk -= chunk % 2
            if not chunk:
                return None, 0

            def pair():
                parity = first_index % 2
                body(advance, parity)
                body(advance, 1-parity)

            return rt.capture(resources, pair, chunk//2), chunk

        old_graph, old_chunk = capture(old_advance, 1, before_steps)
        late_graph, late_chunk = capture(late_advance, split+1, after_steps)
        graph = time.perf_counter()-t

        def enqueue(advance, first_index, executable, chunk, count):
            blocks, tail = divmod(count, chunk) if chunk else (0, count)
            for _ in range(blocks):
                call(rt.driver.cuGraphLaunch, executable, rt.driver.stream)
            for offset in range(tail):
                body(advance, (first_index+blocks*chunk+offset)%2)
            return blocks*chunk+tail

        t = time.perf_counter()
        executed = enqueue(old_advance, 1, old_graph, old_chunk, before_steps)
        body(bridge_advance, split%2)
        executed += 1+enqueue(late_advance, split+1, late_graph, late_chunk, after_steps)
'''
    replacements = (
        ("def solve_cuda_prepared(","def solve_cuda_pingpong("),
        (span,new),
        ('               preparation=preparation, phase_preparation_shared=',
         '               history_copy_elements_per_evaluation=0, history_buffer_mode="parity",\n'
         '               graph_chunk_by_phase=dict(startup=old_chunk,late=late_chunk),\n'
         '               preparation=preparation, phase_preparation_shared='),
    )
    for old, replacement in replacements:
        if text.count(old) != 1:
            raise ValueError("Frozen CUDA source no longer matches the audited replacement")
        text = text.replace(old,replacement)
    with TARGET.open("x",encoding="utf-8",newline="\n") as handle:
        handle.write(text)
    digest = lambda value:hashlib.sha256(value).hexdigest()
    manifest = dict(source=SOURCE.relative_to(HERE.parent).as_posix(),source_sha256=digest(raw),target=TARGET.name,
                    target_sha256=digest(TARGET.read_bytes()),replacement_sha256=[
                        dict(before=digest(a.encode()),after=digest(b.encode())) for a,b in replacements],
                    scope="Host pointer parity and graph replay only. No change to CUDA kernels, plans, stage arithmetic or preparation.")
    with MANIFEST.open("x",encoding="utf-8") as handle:
        json.dump(manifest,handle,indent=2)


if __name__ == "__main__":
    main()
