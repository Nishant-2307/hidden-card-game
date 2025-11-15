# app.py
from flask import Flask, render_template, request, jsonify
from game_logic import (
    deal_cards, possible_worlds, suggest_query, QUERY_CARDS,
    evaluate_queries, hidden_distribution_for_player
)
import copy

app = Flask(__name__, static_folder='static', template_folder='templates')

# ---------- GLOBAL GAME STATE ----------
game_state = {
    "hands": None,
    "hidden": None,
    "answers": [],
    "known_hands": None,
    "history": [],
    "game_over": False,
    "snapshots": [],
    "step_index": 0,
    "strategy": "score"   # "score" (information-split) or "shannon"
}


# ---------- HELPERS ----------
def _record_history(asker, target, q_index, response):
    q_set = sorted(list(QUERY_CARDS[q_index]))
    entry = {"who": f"P{asker}", "asked": f"Player {target} about {q_set}", "got": response}
    game_state["history"].append(entry)
    return entry


def snapshot_label():
    labels = ["Start"]
    for i, _ in enumerate(game_state["history"], start=1):
        labels.append(f"Step {i}")
    return labels


def take_snapshot():
    snap = {
        "hands": {p: sorted(list(game_state["hands"][p])) for p in range(3)} if game_state["hands"] else None,
        "hidden": game_state["hidden"],
        "answers": list(game_state["answers"]),
        "known_hands": {p: sorted(list(game_state["known_hands"][p])) for p in range(3)} if game_state["known_hands"] else None,
        "history": list(game_state["history"]),
        "game_over": game_state["game_over"]
    }
    idx = game_state["step_index"]
    if idx < len(game_state["snapshots"]) - 1:
        game_state["snapshots"] = game_state["snapshots"][:idx+1]
    game_state["snapshots"].append(snap)
    game_state["step_index"] = len(game_state["snapshots"]) - 1


def restore_snapshot(idx):
    if idx < 0 or idx >= len(game_state["snapshots"]):
        return False
    snap = game_state["snapshots"][idx]
    game_state["hands"] = {p: set(snap["hands"][p]) for p in range(3)} if snap["hands"] is not None else None
    game_state["hidden"] = snap["hidden"]
    game_state["answers"] = list(snap["answers"])
    game_state["known_hands"] = {p: set(snap["known_hands"][p]) for p in range(3)} if snap["known_hands"] is not None else None
    game_state["history"] = list(snap["history"])
    game_state["game_over"] = bool(snap["game_over"])
    game_state["step_index"] = idx
    return True


def _check_winner():
    for p in range(3):
        kh = {0: set(), 1: set(), 2: set()}
        kh[p] = set(game_state["known_hands"][p])
        worlds = possible_worlds(kh, game_state["answers"])
        possible_hidden = sorted({h for h, _ in worlds})
        if len(possible_hidden) == 1:
            return True, p, possible_hidden[0]
    return False, None, None


def beliefs_for_player(player_idx):
    kh = {0: set(), 1: set(), 2: set()}
    kh[player_idx] = set(game_state["known_hands"][player_idx])
    worlds = possible_worlds(kh, game_state["answers"])
    return sorted({h for h, _ in worlds})


def get_certain_cards_player0_view():
    # Which cards player 0 can be sure about (who has what) from P0's POV
    kh = {0: set(), 1: set(), 2: set()}
    kh[0] = set(game_state["known_hands"][0])
    worlds = possible_worlds(kh, game_state["answers"])
    certain = {0: [], 1: [], 2: []}
    for pl in range(3):
        certain[pl] = sorted({card for card in range(1, 11) if worlds and all(card in hands[pl] for _, hands in worlds)})
    return certain


def get_player_knowledge_all():
    pk = {}
    for p in range(3):
        kh = {0: set(), 1: set(), 2: set()}
        kh[p] = set(game_state["known_hands"][p])
        worlds = possible_worlds(kh, game_state["answers"])
        beliefs = sorted({h for h, _ in worlds})
        certain = {pl: sorted({card for card in range(1, 11) if worlds and all(card in hands[pl] for _, hands in worlds)}) for pl in range(3)}
        pk[p] = {"beliefs": beliefs, "certain": certain}
    return pk


def build_response_payload(bots_list, extra=None):
    """
    Build a comprehensive payload for frontend, converting sets->lists etc.
    extra: dict with optional keys: 'shannon_dist', 'query_scores', 'recommendation'
    """
    payload = {
        "you": None,
        "bots": bots_list,
        "history": list(game_state["history"]),
        "beliefs": {0: beliefs_for_player(0)},
        "certain_cards": get_certain_cards_player0_view(),
        "player_knowledge": get_player_knowledge_all(),
        "step_index": game_state["step_index"],
        "total_steps": len(game_state["snapshots"]),
        "step_labels": snapshot_label(),
        "game_over": game_state["game_over"],
        "winner": None,
        "hidden_card": None,
        "final_hands": None,
        "strategy": game_state.get("strategy", "score")
    }

    # last "you" entry — find last player move in history (the player's question is recorded earlier)
    # We expect that in a response after a round, the player's question was the first of the recent history entries.
    if game_state["history"]:
        # the most recent player move is the one that occurs (history length - (1 + bots_count))
        # here we can't easily know bots_list length; we'll simply expose the last human-like entry
        # But store a "you" as the last P0 entry in history if any:
        last_you = None
        for h in reversed(game_state["history"]):
            if h["who"] == "P0":
                last_you = h
                break
        payload["you"] = {"asked": last_you["asked"], "response": last_you["got"]} if last_you else None

    if game_state["game_over"]:
        over, who, card = _check_winner()
        payload["winner"] = who
        payload["hidden_card"] = card
        payload["final_hands"] = {p: sorted(list(game_state["hands"][p])) for p in range(3)}

    if extra:
        payload.update(extra)
    return payload


# ---------- ROUTES ----------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/set_strategy", methods=["POST"])
def set_strategy():
    data = request.get_json() or {}
    s = data.get("strategy", "score")
    if s not in ("score", "shannon", "info"):
        return jsonify({"error": "Invalid strategy"}), 400
    # map UI values to internal keys
    if s == "info":
        game_state["strategy"] = "score"
    elif s == "shannon":
        game_state["strategy"] = "shannon"
    else:
        game_state["strategy"] = s
    return jsonify({"strategy": game_state["strategy"]})


@app.route("/start", methods=["POST"])
def start():
    hands, hidden = deal_cards()
    game_state["hands"] = {p: set(hands[p]) for p in range(3)}
    game_state["hidden"] = hidden
    game_state["answers"] = []
    game_state["known_hands"] = {p: set(hands[p]) for p in range(3)}
    game_state["history"] = []
    game_state["game_over"] = False
    game_state["snapshots"] = []
    game_state["step_index"] = 0
    take_snapshot()

    extra = {
        "shannon_dist": hidden_distribution_for_player({0: set(game_state["known_hands"][0]), 1: set(), 2: set()}, game_state["answers"]),
        "query_scores": [], "recommendation": None
    }

    return jsonify({
        "your_hand": sorted(list(game_state["hands"][0])),
        "query_cards": [sorted(list(q)) for q in QUERY_CARDS],
        "step_index": game_state["step_index"],
        "total_steps": len(game_state["snapshots"]),
        "step_labels": snapshot_label(),
        "history": game_state["history"],
        "beliefs": {0: beliefs_for_player(0)},
        "certain_cards": {p: [] for p in range(3)},
        "player_knowledge": get_player_knowledge_all(),
        "game_over": False,
        "final_hands": None,
        "shannon_dist": extra["shannon_dist"],
        "query_scores": extra["query_scores"],
        "recommendation": extra["recommendation"],
        "strategy": game_state["strategy"]
    })


@app.route("/set_distribution", methods=["POST"])
def set_distribution():
    data = request.get_json() or {}
    mode = data.get("mode", "random")
    if mode == "random":
        hands, hidden = deal_cards()
    else:
        # custom mode
        hands_in = data.get("hands")
        hidden = data.get("hidden")
        if not hands_in:
            return jsonify({"error": "Missing hands for custom mode"}), 400
        try:
            seen = set()
            hands = {}
            for p in range(3):
                arr = hands_in.get(str(p)) or hands_in.get(p)
                if arr is None:
                    return jsonify({"error": f"Missing hand for player {p}"}), 400
                arr = list(map(int, arr))
                hands[p] = set(arr)
                for c in arr:
                    if c in seen:
                        return jsonify({"error": f"Duplicate card {c}"}), 400
                    seen.add(c)
            if hidden is None:
                all_cards = set(range(1, 11))
                leftover = all_cards - seen
                if len(leftover) != 1:
                    return jsonify({"error": "Cannot deduce hidden card; provide hidden"}), 400
                hidden = leftover.pop()
            else:
                hidden = int(hidden)
                if hidden in seen:
                    return jsonify({"error": "Hidden card appears in hands"}), 400
        except Exception as e:
            return jsonify({"error": f"Invalid input: {e}"}), 400

    game_state["hands"] = {p: set(hands[p]) for p in range(3)}
    game_state["hidden"] = int(hidden)
    game_state["answers"] = []
    game_state["known_hands"] = {p: set(hands[p]) for p in range(3)}
    game_state["history"] = []
    game_state["game_over"] = False
    game_state["snapshots"] = []
    game_state["step_index"] = 0
    take_snapshot()

    return jsonify({
        "message": "Distribution set",
        "your_hand": sorted(list(game_state["hands"][0])),
        "query_cards": [sorted(list(q)) for q in QUERY_CARDS],
        "step_index": game_state["step_index"],
        "total_steps": len(game_state["snapshots"]),
        "step_labels": snapshot_label(),
        "history": game_state["history"],
        "beliefs": {0: beliefs_for_player(0)},
        "certain_cards": {p: [] for p in range(3)},
        "player_knowledge": get_player_knowledge_all(),
        "game_over": False,
        "final_hands": None,
        "shannon_dist": hidden_distribution_for_player({0: set(game_state["known_hands"][0]), 1: set(), 2: set()}, game_state["answers"]),
        "query_scores": [],
        "recommendation": None,
        "strategy": game_state["strategy"]
    })


@app.route("/ask", methods=["POST"])
def ask():
    if game_state["game_over"]:
        return jsonify({"error": "Game already over"}), 400

    data = request.get_json() or {}
    asker = int(data.get("asker", 0))
    target = int(data["target"])
    q_index = int(data["query"])
    strategy = data.get("strategy", game_state.get("strategy", "score"))
    # normalize strategy name
    if strategy == "info":
        strategy = "score"

    # perform player's move
    q_set = QUERY_CARDS[q_index]
    response = len(game_state["hands"][target] & q_set)
    game_state["answers"].append((asker, target, q_index, response))
    _record_history(asker, target, q_index, response)
    take_snapshot()

    # check for winner after player's move
    over, who, card = _check_winner()
    if over:
        game_state["game_over"] = True
        payload = build_response_payload([], extra={
            "shannon_dist": hidden_distribution_for_player({0: set(game_state["known_hands"][0]), 1: set(), 2: set()}, game_state["answers"]),
            "query_scores": evaluate_queries({0: set(game_state["known_hands"][0]), 1: set(), 2: set()}, game_state["answers"], 0, strategy=strategy),
            "recommendation": evaluate_queries({0: set(game_state["known_hands"][0]), 1: set(), 2: set()}, game_state["answers"], 0, strategy=strategy)[:1][0] if evaluate_queries({0: set(game_state["known_hands"][0]), 1: set(), 2: set()}, game_state["answers"], 0, strategy=strategy) else None
        })
        return jsonify(payload)

    # BOT 1
    # choose based on bot's POV
    bot1_choice, _ = suggest_query({0: set(), 1: set(game_state["known_hands"][1]), 2: set()}, game_state["answers"], 1, strategy=strategy)
    bot1 = None
    if bot1_choice:
        t1, q1 = bot1_choice
        resp1 = len(game_state["hands"][t1] & QUERY_CARDS[q1])
        game_state["answers"].append((1, t1, q1, resp1))
        _record_history(1, t1, q1, resp1)
        take_snapshot()
        bot1 = {"bot": 1, "asked": f"Player {t1} about {sorted(list(QUERY_CARDS[q1]))}", "response": resp1}

    over, who, card = _check_winner()
    if over:
        game_state["game_over"] = True
        payload = build_response_payload([bot1] if bot1 else [], extra={
            "shannon_dist": hidden_distribution_for_player({0: set(game_state["known_hands"][0]), 1: set(), 2: set()}, game_state["answers"]),
            "query_scores": evaluate_queries({0: set(game_state["known_hands"][0]), 1: set(), 2: set()}, game_state["answers"], 0, strategy=strategy),
            "recommendation": evaluate_queries({0: set(game_state["known_hands"][0]), 1: set(), 2: set()}, game_state["answers"], 0, strategy=strategy)[:1][0] if evaluate_queries({0: set(game_state["known_hands"][0]), 1: set(), 2: set()}, game_state["answers"], 0, strategy=strategy) else None
        })
        return jsonify(payload)

    # BOT 2
    bot2_choice, _ = suggest_query({0: set(), 1: set(), 2: set(game_state["known_hands"][2])}, game_state["answers"], 2, strategy=strategy)
    bot2 = None
    if bot2_choice:
        t2, q2 = bot2_choice
        resp2 = len(game_state["hands"][t2] & QUERY_CARDS[q2])
        game_state["answers"].append((2, t2, q2, resp2))
        _record_history(2, t2, q2, resp2)
        take_snapshot()
        bot2 = {"bot": 2, "asked": f"Player {t2} about {sorted(list(QUERY_CARDS[q2]))}", "response": resp2}

    over, who, card = _check_winner()
    if over:
        game_state["game_over"] = True

    # prepare extra UI info: distribution & per-query scores & recommendation
    shannon_dist = hidden_distribution_for_player({0: set(game_state["known_hands"][0]), 1: set(), 2: set()}, game_state["answers"])
    query_scores = evaluate_queries({0: set(game_state["known_hands"][0]), 1: set(), 2: set()}, game_state["answers"], 0, strategy=strategy)
    recommendation = query_scores[0] if query_scores else None

    return jsonify(build_response_payload([b for b in (bot1, bot2) if b], extra={
        "shannon_dist": shannon_dist,
        "query_scores": query_scores,
        "recommendation": recommendation
    }))


@app.route("/rewind", methods=["POST"])
def rewind():
    data = request.get_json() or {}
    idx = int(data.get("step", 0))
    ok = restore_snapshot(idx)
    if not ok:
        return jsonify({"error": "Invalid step index"}), 400

    shannon_dist = hidden_distribution_for_player({0: set(game_state["known_hands"][0]), 1: set(), 2: set()}, game_state["answers"])
    query_scores = evaluate_queries({0: set(game_state["known_hands"][0]), 1: set(), 2: set()}, game_state["answers"], 0, strategy=game_state.get("strategy", "score"))
    recommendation = query_scores[0] if query_scores else None

    return jsonify({
        "history": game_state["history"],
        "beliefs": {0: beliefs_for_player(0)},
        "certain_cards": get_certain_cards_player0_view(),
        "player_knowledge": get_player_knowledge_all(),
        "step_index": game_state["step_index"],
        "total_steps": len(game_state["snapshots"]),
        "step_labels": snapshot_label(),
        "game_over": game_state["game_over"],
        "winner": None,
        "hidden_card": game_state["hidden"],
        "final_hands": {p: sorted(list(game_state["hands"][p])) for p in range(3)} if game_state["game_over"] else None,
        "shannon_dist": shannon_dist,
        "query_scores": query_scores,
        "recommendation": recommendation,
        "strategy": game_state.get("strategy", "score")
    })


if __name__ == "__main__":
    app.run(debug=True)
