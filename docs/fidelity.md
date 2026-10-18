# Fidelity notes

What the engine implements, which numbers were verified, the judgment calls
made where the rules are terse, and the known simplifications. The goal is a
faithful re-implementation of Balatro 1.0 for RL research; every rule is
written from scratch in Python.

## Content coverage

| Category | Count | Notes |
|---|---|---|
| Jokers | 150 | all rarities incl. the 5 Legendaries (The Soul only) |
| Tarots | 22 | |
| Planets | 12 | secret planets gated on discovering their hand |
| Spectrals | 18 | The Soul / Black Hole only via pack odds |
| Boss blinds | 28 | 23 regular + 5 ante-8 finishers |
| Vouchers | 32 | 16 base/upgrade pairs, upgrade requires the base |
| Decks | 15 | |
| Stakes | 8 | cumulative modifiers |
| Tags | 24 | |
| Booster packs | 15 | 5 types x Normal / Jumbo / Mega |
| Enhancements / editions / seals / stickers | all | Eternal, Perishable, Rental |

## Verified numbers

Costs, rarities, boss antes/multipliers, voucher effects, tag availability,
deck configs, pack sizes/weights, the three ante scaling tables, interest
caps, shop kind weights, edition odds and sticker odds were checked against
the game's data tables:

- Shop: Joker 20 / Tarot 4 / Planet 4 / playing card 0 (Magic Trick 4) /
  Spectral 0 (Ghost Deck 2). Tarot & Planet Merchant x2.4, Tycoon x8.
- Editions in the shop (base rate): Negative 0.3%, Polychrome 0.3%,
  Holographic 1.4%, Foil 2.0%. Hone x2, Glow Up x4 (Negative unscaled).
- Stickers: Eternal 30% (Black+), Perishable 30% (Orange+; exclusive with
  Eternal), Rental 30% (Gold; independent). Rental jokers cost $1 and charge
  $3 at the end of every round.
- Interest: $1 per $5 held, capped at $25 held (Seed Money $50, Money Tree
  $100); To the Moon adds $1 per $5.
- Ante requirements: White/Red/Black/Blue 300 ... 50k; Green (and above)
  300 ... 100k; Purple/Orange/Gold 300 ... 200k. Antes 9+ use the endless
  formula. Plasma Deck doubles the requirement.
- Stake modifiers: Red = no Small Blind reward; Green = faster scaling;
  Black = Eternals; Blue = -1 discard; Purple = faster still; Orange =
  Perishables; Gold = Rentals. Each stake includes the ones below it.

## Activation sequence

`engine/scoring.py` runs: boss (pre-base) -> base chips/mult -> on-played
jokers -> scored cards left to right (base, enhancement, gold seal, edition,
on-scored jokers; retriggers: red seal first then jokers left to right) ->
held cards (steel, on-held jokers, retriggers) -> per joker left to right
(foil/holo, independent effect, Baseball-style pass, polychrome) ->
Observatory -> Plasma balancing -> floor(chips x mult).

Blueprint/Brainstorm resolve through copy chains with a cycle guard. Copies
read the target's state; `update_*` hooks only run for originals, and
scoring hooks that must mutate (Vampire, Wee Joker) skip the mutation for
copies. Consequence: a Blueprint left of Vampire eats enhancements without
growing anything, matching the known in-game interaction.

## Judgment calls

- **The Flint** halves base chips and mult rounded up.
- **The Arm** permanently lowers the hand level (minimum 1) before base is read.
- **The Ox** fixes its target hand when the blind starts; ties go to the
  lowest hand type.
- **Suit debuffs** use the printed suit (a debuffed Wild reverts).
- **The Plant / The Mark** treat Pareidolia-faces as faces.
- **The Pillar** debuffs every card played earlier this ante, including
  earlier in the boss round.
- **Cerulean Bell** forces a random card after every draw; both plays and
  discards must include it.
- **Crimson Heart** re-picks its victim after every hand (it may repeat).
- **Amber Acorn** shuffles the board and hides joker ids from the observation;
  reordering is disabled for the round.
- **Verdant Leaf** debuffs every card until any joker is sold.
- **Matador** pays when a per-hand boss ability fires (Hook, Fish, Serpent,
  Tooth, Crimson Heart always; Arm/Ox/Flint when they act; debuff bosses when
  a played card is debuffed).
- **Loyalty Card** pays x4 on the 6th, 12th, ... hand played since acquired.
- **Obelisk** compares against the most-played hand(s) before counting the
  current hand.
- **Ice Cream / Seltzer / Ramen** are destroyed right after the hand (or
  discard) that empties them; Popcorn, Turtle Bean, Gros Michel and Cavendish
  resolve at round end.
- **Perishable** debuffs after 5 completed rounds; a debuffed joker fires no
  hooks at all (and copies that resolve to it resolve to nothing).
- **Stuck rounds** (no legal play or discard, e.g. The Psychic with fewer
  than five cards, or an emptied hand) are resolved as a lost blind, Mr. Bones
  permitting.
- **Pack consumables** are used immediately when picked; Arcana and Spectral
  packs deal a hand of your deck for targeting, as in the game.
- **Skip tags and the boss** are pre-rolled at the start of each ante so the
  agent sees them on the blind-select screen.
- **Double Tag** copies the next tag acquired (never another Double); the copy
  is applied immediately after the original.
- **Hidden spectrals**: The Soul replaces a pack option with 0.3% per slot in
  Arcana/Spectral packs; Black Hole likewise in Celestial/Spectral packs.

## Known simplifications

- The RNG is seeded per stream (shuffle, shop, packs, chance, boss, ...);
  it does not reproduce real game seeds.
- Vouchers appear one per ante (the pre-rolled ante voucher persists across
  that ante's shops); the Voucher Tag adds one for a single shop.
- The action codec exposes the 12 highest-ranked cards of the hand; hand
  sizes above 12 are legal in the engine but the extra low cards are not
  selectable through the Gymnasium action space.
- Card-targeting consumables take up to 3 targets through the env (the
  engine supports the same arities as the game).
- Endless mode is not part of the win condition (ante 8 boss wins the run),
  but requirements for antes 9+ are implemented.
- Shop playing cards from Magic Trick cost $1 (before discounts).
- The hand dealt by a pack is a uniform sample of the deck rather than the
  top of the current draw pile.
