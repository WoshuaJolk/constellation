"""Deterministic proposal order spanning epochs and distance hypotheses early.

Planning only: it does not silently replace the frozen production graph, remove
caps, or claim equal coverage under unequal task costs. Each task needs a separate
completion receipt. An interrupted task stays pending, never marked searched.
"""
import hashlib
import math


def coverage_plan(anchors, distances, radial_fractions, seed='distant-coverage-v1'):
    anchors = tuple(float(x) for x in anchors)
    distances = tuple(float(x) for x in distances)
    radial = tuple(float(x) for x in radial_fractions)
    for name, values in [('anchors', anchors), ('distances', distances), ('radial', radial)]:
        if len(set(values)) != len(values) or not all(math.isfinite(x) for x in values):
            raise ValueError(name+' must be finite and unique')
    if any(x <= 0 for x in distances):
        raise ValueError('Distances must be positive')
    if not anchors or not distances or not radial:
        return []
    # Hash timestamps, never fit outcomes, to avoid chronological truncation.
    epochs = sorted(anchors, key=lambda x: hashlib.sha256((seed+':'+x.hex()).encode()).digest())
    # Bit reversal spreads early tasks across the full ordered distance grid.
    distances = tuple(sorted(distances))
    bits = max(1, (len(distances)-1).bit_length())
    order = sorted(range(len(distances)), key=lambda i: int(format(i, f'0{bits}b')[::-1], 2))
    hypotheses = [(distances[i], f) for f in radial for i in order]
    tasks = []
    for round_index in range(len(hypotheses)):
        for j, epoch in enumerate(epochs):
            distance, fraction = hypotheses[(round_index+j) % len(hypotheses)]
            key = '|'.join(x.hex() for x in (epoch, distance, fraction))
            tasks.append(dict(task_id=hashlib.sha256(key.encode()).hexdigest(),
                epoch=epoch, distance_au=distance, radial_fraction=fraction))
    return tasks
