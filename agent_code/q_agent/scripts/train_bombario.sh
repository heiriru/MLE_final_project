#!/usr/bin/env bash
# ==========================================================================
#  Trainings-Curriculum fuer den BOMBARIO (Fix 1: Fluchtweg ist im
#  Feature-Vektor sichtbar). Wie das escape-Curriculum, ergaenzt um die
#  zwei Fight-Phasen im Szenario "empty" (4b/4c).
#
#  Leitgedanke aus der Analyse:
#    - Abdeckung ist NICHT der Engpass (Zustandsraum nur 4^4*5*3 = 3840).
#    - Der Engpass war Flucht/Suizid. Fix 1 macht den Rettungsweg lernbar.
#  Deshalb: Schwierigkeit ueber die KISTENDICHTE hochziehen, Gegner erst
#  spaet, und in jeder Phase, in der eine NEUE Faehigkeit dazukommt
#  (Bomben+Flucht ab crates-easy), VIEL Exploration geben.
#
#  Pro Phase werden Hyperparameter ueber Umgebungsvariablen gesetzt, die der
#  Agent liest (BM_ALPHA / BM_EPS_START / BM_EPS_END / BM_EPS_DECAY).
#  BM_EPS_DECAY = ~70% der Phasenlaenge -> laengere Phasen senken epsilon
#  langsamer ab.
#
#  BENUTZUNG (bomberman-Env muss aktiv sein):
#     conda activate bomberman
#     MINI=1  FRESH=1 ./scripts/train_bombario.sh   # ~10 min Logiktest (bombt+flieht+sammelt?)
#     QUICK=1 FRESH=1 ./scripts/train_bombario.sh   # ~1 h Schnelltest (laeuft es besser?)
#     FRESH=1        ./scripts/train_bombario.sh    # voller Lauf, frisch (empfohlen)
#                    ./scripts/train_bombario.sh    # voller Lauf, auf vorhandener q_table weiter
#     SCALE=10       ./scripts/train_bombario.sh    # voller Lauf mit 10% der Runden
#     ./scripts/train_bombario.sh mein_agent        # anderen Agenten trainieren
# ==========================================================================
set -euo pipefail

AGENT="${1:-q_agent/BOMBARIO}"
SCALE="${SCALE:-100}"          # Prozent der Basis-Runden (100 = voll)
FRAMEWORK="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$FRAMEWORK"

if [ "${FRESH:-0}" = "1" ]; then
  rm -f "agent_code/$AGENT/q_table.pkl" \
        "agent_code/$AGENT/train_meta.json" \
        "agent_code/$AGENT/training_stats.json" \
        "agent_code/$AGENT"/stats_*.json
  echo ">>> FRESH: q_table / meta / stats geloescht -- starte bei 0."
fi

# run_phase <name> <basis_runden> <szenario> <eps_start> <alpha> <gegner...>
run_phase() {
  local name="$1" base="$2" scenario="$3" eps_start="$4" alpha="$5"; shift 5
  local agents=("$AGENT" "$@")           # unser Agent zuerst -> wird mit --train 1 trainiert
  local rounds=$(( base * SCALE / 100 )); [ "$rounds" -lt 1 ] && rounds=1
  local decay=$(( rounds * 70 / 100 ));  [ "$decay" -lt 1 ] && decay=1
  echo "=============================================================="
  echo ">>> Phase $name | $scenario | Runden=$rounds eps_start=$eps_start alpha=$alpha eps_decay=$decay"
  echo "    Gegner: ${*:-(keine)}"
  echo "=============================================================="
  BM_EPS_START="$eps_start" BM_EPS_END="0.05" BM_EPS_DECAY="$decay" BM_ALPHA="$alpha" \
    python main.py play --agents "${agents[@]}" --train 1 \
      --scenario "$scenario" --no-gui --n-rounds "$rounds"
  # Stats dieser Phase sichern (training_stats.json wird sonst naechste Phase ueberschrieben)
  cp "agent_code/$AGENT/training_stats.json" "agent_code/$AGENT/stats_${name}.json" 2>/dev/null || true
}

if [ "${MINI:-0}" = "1" ]; then
  # ---- MINI-Logiktest (~10 min): greift Fix 1? Bombt + flieht + sammelt der Agent? ----
  #            name             runden  szenario       eps_start alpha  gegner...
  run_phase "1_coins"            500    coin-heaven    0.3       0.10
  run_phase "2_crates_easy"     1200    crates-easy    0.5       0.10
  run_phase "3_crates_mid"      1200    crates-middle  0.5       0.06

elif [ "${QUICK:-0}" = "1" ]; then
  # ---- Schnelltest (~1 h): kurze, aber vollstaendige Leiter ----
  #            name             runden  szenario       eps_start alpha  gegner...
  run_phase "1_coins"           1000    coin-heaven    0.3       0.10
  run_phase "2_crates_easy"     2500    crates-easy    0.5       0.10
  run_phase "3_crates_mid"      3000    crates-middle  0.5       0.06
  run_phase "4_lootcrate"       2500    loot-crate     0.5       0.05

else
  # ---- Voller Lauf (mehrere Stunden). Dichte steigt, Gegner erst spaet. ----
  #            name             runden  szenario       eps_start alpha  gegner...
  run_phase "1_coins"           3000    coin-heaven    0.3       0.10                                                                        # reine Bewegung + Muenzen
  run_phase "2_crates_easy"     4000    crates-easy    0.6       0.10                                                                        # Bomben+Flucht NEU -> viel Exploration
  run_phase "3_crates_mid"      6000    crates-middle  0.6       0.06   peaceful_agent peaceful_agent peaceful_agent                                     # dichter -> Flucht schwerer
  run_phase "4_lootcrate"       6000    loot-crate     0.5       0.05   coin_collector_agent coin_collector_agent coin_collector_agent       # volle Dichte -> Flucht MUSS sitzen
  run_phase "4b_hunt_peaceful"  2000    empty          0.8       0.10   peaceful_agent peaceful_agent peaceful_agent                         # Kessel/Bombe gegen bewegtes Ziel, kein Rueckfeuer
  run_phase "4c_hunt_rule"      2000    empty          0.7       0.08   rule_based_agent  "$AGENT"  "$AGENT"                    # gegen fliehende, bombende Gegner
  run_phase "5_coincollector"   6000    classic        0.4       0.04   coin_collector_agent coin_collector_agent coin_collector_agent                    # volles Spiel, alle Faehigkeiten zugleich
  run_phase "9_selfplay"        10000    classic        0.30      0.03   rule_based_agent rule_based_agent "$AGENT"                            # Selbstspiel-Abschluss, niedriges epsilon
fi

echo
echo "=============================================================="
echo ">>> Fertig. Abschluss-Eval (train OFF)."
echo "=============================================================="
if [ "${MINI:-0}" = "1" ]; then
  # MINI: prueft die Logik auf den Kisten-Szenarien (bombt + ueberlebt + sammelt?)
  for sc in crates-easy crates-middle; do
    python main.py play --agents "$AGENT" --scenario "$sc" --no-gui --n-rounds 60 \
      --save-stats "/tmp/escape_mini_${sc}.json" || true
    python - "$AGENT" "$sc" <<'PY' || true
import json,sys,numpy as np
a,sc=sys.argv[1],sys.argv[2]; d=json.load(open(f'/tmp/escape_mini_{sc}.json'))
ba=d['by_agent'][a]; byr=d['by_round']; r=len(byr)
suic=sum(x.get('suicides',0) for x in byr.values())
st=np.array([x['steps'] for x in byr.values()]); co=np.array([x['coins'] for x in byr.values()])
print(f'{sc:14} bombs/Rd {ba.get("bombs",0)/r:5.2f}  crates/Rd {ba.get("crates",0)/r:5.2f}  '
      f'coins/Rd {co.mean():4.2f}  Suizide {100*suic/r:3.0f}%  bisEnde {100*(st>=395).mean():3.0f}%')
PY
  done
  echo "(Ziel: bombs>0, crates hoch, coins hoch, Suizide DEUTLICH unter dem 82%-Altstand)"
else
  echo ">>> 1) SOLO auf classic -- isoliert Suizide/Ueberleben (Vergleich: Altstand 82% Suizide):"
  python main.py play --agents "$AGENT" --scenario classic --no-gui --n-rounds 100 \
    --save-stats /tmp/escape_solo.json || true
  python - "$AGENT" <<'PY' || true
import json,sys,numpy as np
a=sys.argv[1]; d=json.load(open('/tmp/escape_solo.json')); byr=d['by_round']; r=len(byr)
suic=sum(x.get('suicides',0) for x in byr.values())
st=np.array([x['steps'] for x in byr.values()]); co=np.array([x['coins'] for x in byr.values()])
inv=d['by_agent'][a].get('invalid',0)
print(f'SOLO classic: Suizide {suic}/{r}={100*suic/r:.0f}%  Ø steps {st.mean():.0f}  '
      f'bisEnde {100*(st>=395).mean():.0f}%  Ø coins {co.mean():.2f}  invalid/Rd {inv/r:.2f}')
PY
  echo ">>> 2) gegen 3x rule_based (Score-Vergleich):"
  python "$ROOT/scripts/evaluate.py" "$AGENT" rule_based_agent rule_based_agent rule_based_agent \
    -n 100 --scenario classic || true
fi
echo
echo "Lernkurven aller Phasen plotten:"
echo "   python scripts/plot_all_phases.py $AGENT"
