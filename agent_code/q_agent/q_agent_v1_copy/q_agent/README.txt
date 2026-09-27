Agent Version 1  --  Snapshot vom 2026-09-07
================================================
Trainierter Stand VOR der Bomben-Reward-Staffelung.

  train.py     = ALTE Rewards (flaches GOOD_BOMB=2.0, COIN_COLLECTED=3.0),
                 also der Stand, mit dem die q_table trainiert wurde.
  callbacks.py = unveraendert (Fix 1: Fluchtweg sichtbar).
  q_table.pkl  = trainierter Stand (voller 7-Phasen-Lauf, siehe training_all_phases.png).
  stats_*.json = Lernkurven pro Phase; training_stats.json = letzte Phase.

Spielen (read-only, aendert die q_table NICHT):
  python main.py play --agents agent_version_1 --scenario classic --no-gui --n-rounds 100
NICHT mit --train laufen lassen, sonst wird die gespeicherte q_table ueberschrieben.
