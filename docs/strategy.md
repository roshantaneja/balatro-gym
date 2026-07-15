# Balatro strategy notes (for heuristics and reward design)

Distilled from community consensus (primarily DrBoomMD's "How to Win Chips and
Influence Mult" guide — Completionist+, ~119h — plus thread replies from other
gold-stake players), adapted to the v1 content catalog. These rules drive
`StrategistAgent` in `balatro_gym/env/baselines.py`.

## Core doctrine: score with jokers, not hands

1. **The flush fallacy.** Flushes are the best ante-1 hand but a trap as a
   strategy: hard to find (≤1-2 per round), no jokers scale them, and many
   bosses (suit debuffs, The Water, The Arm, The Eye) counter them.
2. **Scaling is king.** The most run-winning jokers gain value every hand or
   round: Supernova, Green Joker, Ride the Bus, Castle (chips), plus xmult
   scalers outside v1 (Constellation, Vampire, Campfire, Hologram).
   A winning board ≈ **1 big-chip joker + 1-2 +mult jokers + 1-2 xmult jokers**.
3. **High Card / Pair spam.** Joker builds don't care what hand you play, so
   play the cheapest hand every turn: it triggers per-hand scalers 4x/round,
   levels one hand type deeply, and shrugs off most boss effects.
   Corollaries: Half Joker (≤3 cards) and The Duo (contains Pair) become
   near-unconditional; Ride the Bus wants *no scoring face cards* (face cards
   may still be PLAYED if they don't score); Green Joker wants *no discards*.
4. **Tempo vs scaling.** Tempo jokers (flat +mult/+chips, decaying jokers like
   Ice Cream/Popcorn/Gros Michel) exist to survive antes 1-3 while scalers
   grow. Buy them early, sell them late when slots are needed.
5. **Joker order matters**: +mult before xmult (left to right). (v1 heuristic
   cannot reorder; buy order approximates it.)

## Economy doctrine

6. **Interest is a build engine.** +$1 per $5 held, capped at $5/round ($25).
   Reach and stay above $25 as fast as possible; each round at cap ≈ one free
   reroll or half a pack.
7. **Early spending exception**: high-tempo or premium scaling jokers are
   worth dipping below the cap in antes 1-2; nearly nothing else is.
8. **Rerolls > speculative packs** when no committed build. Celestial packs
   only pay with a committed hand type (or Constellation); tarot packs are
   good value if they don't break the interest threshold; buffoon packs are
   almost always worth it early.
9. **The Hermit doubles money (max +$20)** — best used at $20+; Temperance
   pays total joker sell value (max $50) — best with a full board.
10. **Skips**: mostly play small blinds for the money + shop visit; skip only
    for premium tags (not in v1's tag set) — so v1: never skip.

## Planet doctrine

11. Planets scale linearly while blind requirements scale super-exponentially:
    strong early, weak late. Level the ONE hand you spam (Pair/High Card);
    levels on hands you don't play are wasted money.

## v1 catalog tiering (43 jokers; for shop priority)

- **S (buy always, dip below reserve):** j_supernova, j_green_joker,
  j_ride_the_bus, j_castle (scalers) · j_duo (x2 on pair-spam) · j_blueprint,
  j_brainstorm (copy the best scaler/xmult)
- **A (buy if half-reserve stays):** j_space (levels spam hand), j_photograph,
  j_half, j_sly, j_banner (chips), j_hack (retrigger low ranks, pairs with
  low-rank spam), j_golden (economy)
- **B (buy early / cheap):** j_joker, j_jolly, j_abstract, j_scholar,
  j_odd_todd, j_fortune_teller, j_bloodstone, j_mime, j_dusk, j_baron
- **C (tempo only, antes 1-3):** j_ice_cream, j_popcorn, j_gros_michel,
  j_business, suit jokers (j_greedy/…), j_shoot_the_moon, j_raised_fist,
  j_mail_in, j_8ball, j_egg
- **D (situational/avoid):** j_baseball (needs 2+ uncommons), j_dna, j_splash,
  j_four_fingers, j_pareidolia (breaks Ride the Bus: every card becomes a
  scoring face)

## Anti-synergies the agent must respect

- Pareidolia + Ride the Bus (all cards are faces → RTB never scales)
- Green Joker + discard-hungry play (each discard = -1 mult)
- Castle wants discards / Green Joker hates them: prefer Green side (guide
  consensus: don't discard unless the hand is truly dead)
