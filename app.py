import argparse
import json
from pathlib import Path


def solve(data):
    from collections import Counter, defaultdict
    from math import log
    seeds = list(data["seeds"])
    pool = list(enumerate(data["pool"]))
    trace = []
    for _ in range(min(data["budget"], len(pool))):
        counts = defaultdict(Counter)
        docs = Counter()
        for sentence, label in seeds:
            docs[label] += 1
            counts[label].update(sentence.lower().split())
        if len(docs) < 2:
            raise ValueError("Seed at least two classes")
        vocabulary = set().union(*(set(c) for c in counts.values()))
        def margin(entry):
            tokens = entry[1][0].lower().split()
            scores = []
            for label, count in counts.items():
                denom = sum(count.values()) + len(vocabulary) + 1
                scores.append(log(docs[label] / len(seeds)) + sum(
                    log((count[token] + 1) / denom) for token in tokens))
            scores.sort(reverse=True)
            return scores[0] - scores[1], entry[0]
        selected = min(pool, key=margin)
        gap, index = margin(selected)
        sentence, revealed = selected[1]
        trace.append({"index": index, "text": sentence, "margin": gap, "revealed": revealed})
        seeds.append((sentence, revealed))
        pool.remove(selected)
    return {"queries": trace, "training_size": len(seeds)}

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Use active learning to choose which text examples a human should label next.')
    parser.add_argument("input", nargs="?", type=Path, default=Path(__file__).with_name("sample.json"))
    args = parser.parse_args()
    print(json.dumps(solve(json.loads(args.input.read_text())), indent=2))
