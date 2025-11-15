# game_logic.py
import itertools
import random
from collections import defaultdict
import math

# Game constants
CARDS = set(range(1, 11))
PLAYERS = 3
CARDS_PER_PLAYER = 3

QUERY_CARDS = [
    {1, 2, 3, 4},
    {1, 5, 6, 7},
    {2, 5, 8, 9},
    {3, 6, 8, 10},
    {4, 7, 9, 10},
]


# ---------------------------
# Possible worlds generator
# ---------------------------
def possible_worlds(known_hands, answers):
    """
    known_hands: dict player -> set(known cards for that player)
    answers: list of (asker, target, q_index, response)
    Returns: list of tuples (hidden_card, hands) where hands is dict player->set
    """
    unknown_cards = CARDS - set().union(*known_hands.values())
    hidden_candidates = list(unknown_cards)
    possible = []

    def fill(players, cards):
        if not players:
            yield {}
            return
        p = players[0]
        need = CARDS_PER_PLAYER - len(known_hands[p])
        for combo in itertools.combinations(cards, need):
            rest = cards - set(combo)
            for r in fill(players[1:], rest):
                yield {p: set(combo), **r}

    for hidden in hidden_candidates:
        remaining_cards = unknown_cards - {hidden}
        unknown_players = [p for p in range(PLAYERS) if len(known_hands[p]) < CARDS_PER_PLAYER]
        for assignment in fill(unknown_players, remaining_cards):
            hands = {p: set(known_hands[p]) | assignment.get(p, set()) for p in range(PLAYERS)}
            valid = True
            for (asker, target, q_idx, response) in answers:
                q = QUERY_CARDS[q_idx]
                if len(hands[target] & q) != response:
                    valid = False
                    break
            if valid:
                possible.append((hidden, hands))
    return possible


# ---------------------------
# Deal
# ---------------------------
def deal_cards():
    all_cards = list(CARDS)
    random.shuffle(all_cards)
    hands = {}
    for p in range(PLAYERS):
        hands[p] = set(all_cards[p * CARDS_PER_PLAYER: (p + 1) * CARDS_PER_PLAYER])
    hidden = all_cards[PLAYERS * CARDS_PER_PLAYER]
    return hands, hidden


# ---------------------------
# Scoring functions
# ---------------------------
def score_max_split(worlds, query_index, asker, target):
    answer_groups = defaultdict(set)
    for hidden, hands in worlds:
        ans = len(hands[target] & QUERY_CARDS[query_index])
        answer_groups[ans].add(hidden)

    total = set().union(*answer_groups.values()) if answer_groups else set()
    score = sum(len(g) * (len(total) - len(g)) for g in answer_groups.values())
    return score


def entropy_from_counts(counts):
    total = sum(counts.values())
    if total == 0:
        return 0.0
    ent = 0.0
    for c in counts.values():
        p = c / total
        if p > 0:
            ent -= p * math.log2(p)
    return ent


def score_shannon(worlds, query_index, asker, target):
    """
    Return negative expected entropy (so larger is better), and also return
    the expected entropy itself (for use in UI). We keep API returning only a number
    for ranking (negated expected entropy) but provide helper functions to compute distribution.
    """
    # prior distribution over hidden (from worlds)
    hidden_counts = defaultdict(int)
    for h, _ in worlds:
        hidden_counts[h] += 1
    prior_ent = entropy_from_counts(hidden_counts)

    # For each possible answer, compute entropy over hidden conditioned on that answer
    answer_counts = defaultdict(int)
    subsets_hidden_counts = {}
    for h, hands in worlds:
        ans = len(hands[target] & QUERY_CARDS[query_index])
        answer_counts[ans] += 1
        subsets_hidden_counts.setdefault(ans, defaultdict(int))[h] += 1

    total = sum(answer_counts.values())
    expected_entropy = 0.0
    for ans, cnt in answer_counts.items():
        p_ans = cnt / total
        sub_counts = subsets_hidden_counts[ans]
        ent_sub = entropy_from_counts(sub_counts)
        expected_entropy += p_ans * ent_sub

    info_gain = prior_ent - expected_entropy
    # For comparison with max-split where larger is better, return info_gain (larger better)
    return info_gain, expected_entropy


# ---------------------------
# Helper: hidden distribution given a player's POV
# ---------------------------
def hidden_distribution_for_player(known_hands, answers):
    worlds = possible_worlds(known_hands, answers)
    if not worlds:
        return {}
    counts = defaultdict(int)
    for h, _ in worlds:
        counts[h] += 1
    total = sum(counts.values())
    return {h: counts[h] / total for h in sorted(counts.keys())}


# ---------------------------
# Query evaluator (for UI)
# ---------------------------
def evaluate_queries(known_hands, answers, asker, strategy="score"):
    """
    Returns a list of dicts: {target, q_index, query_set, score, ig, expected_entropy}
    For 'score' strategy, ig/expected_entropy will be None; for 'shannon', score==ig.
    """
    worlds = possible_worlds(known_hands, answers)
    results = []
    if not worlds:
        return results

    # compute prior entropy once for convenience
    prior_counts = defaultdict(int)
    for h, _ in worlds:
        prior_counts[h] += 1
    prior_ent = entropy_from_counts(prior_counts)

    for target in range(PLAYERS):
        if target == asker:
            continue
        for q_idx in range(len(QUERY_CARDS)):
            if strategy == "shannon":
                ig, expected_ent = score_shannon(worlds, q_idx, asker, target)
                score = ig
                results.append({
                    "target": target,
                    "q_index": q_idx,
                    "set": sorted(list(QUERY_CARDS[q_idx])),
                    "score": score,
                    "ig": ig,
                    "expected_entropy": expected_ent,
                    "prior_entropy": prior_ent
                })
            else:
                score = score_max_split(worlds, q_idx, asker, target)
                # approximate info gain by looking at group sizes (not exact)
                results.append({
                    "target": target,
                    "q_index": q_idx,
                    "set": sorted(list(QUERY_CARDS[q_idx])),
                    "score": score,
                    "ig": None,
                    "expected_entropy": None,
                    "prior_entropy": prior_ent
                })
    # sort descending by score
    results.sort(key=lambda x: x["score"] if x["score"] is not None else 0, reverse=True)
    return results


# ---------------------------
# Suggest query (single best)
# ---------------------------
def suggest_query(known_hands, answers, asker, strategy="score"):
    worlds = possible_worlds(known_hands, answers)
    if not worlds:
        return None, None

    best = None
    best_score = -1e9

    for target in range(PLAYERS):
        if target == asker:
            continue
        for q_idx in range(len(QUERY_CARDS)):
            if strategy == "shannon":
                ig, _ = score_shannon(worlds, q_idx, asker, target)
                score = ig
            else:
                score = score_max_split(worlds, q_idx, asker, target)

            if score > best_score:
                best_score = score
                best = (target, q_idx)

    return best, best_score
