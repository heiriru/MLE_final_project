# Parses coins and kills per round from a game log.
import json
import re
import sys

ROUND = re.compile(r"STARTING ROUND #(\d+)")
COIN = re.compile(r"Agent <([^>]+)> picked up coin")
KILL = re.compile(r"blown up by agent <([^>]+)>'s bomb")


# Reads a game log and counts, per round and agent, collected coins and kills (for the round-
# paired estimate).
def main(log_path, out_path):
    rounds, current = [], None
    with open(log_path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if ROUND.search(line):
                current = {}
                rounds.append(current)
                continue
            if current is None:
                continue
            match = COIN.search(line)
            if match:
                entry = current.setdefault(match.group(1), [0, 0])
                entry[0] += 1
                continue
            match = KILL.search(line)
            if match:
                entry = current.setdefault(match.group(1), [0, 0])
                entry[1] += 1
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(rounds, handle)
    print(f"{len(rounds)} rounds -> {out_path}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
