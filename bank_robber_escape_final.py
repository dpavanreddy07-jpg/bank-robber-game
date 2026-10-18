"""Bank Robber Escape - single-file Solara game (balanced version).

Run with:   python -m solara run bank_robber_escape.py

Rules (from the case-study synopsis):
  * 3 robbers (speed 2), 2 police (speed 1), 5 money bags, 1 exit
  * Robbers win: team has >= 3 money bags AND a robber reaches the exit
  * Police win: all robbers caught (or time runs out)
  * Both sides use A* search with a Manhattan-distance heuristic

Pick a scenario in the UI: "Balanced", "Robbers win" or "Police win".

Why results now differ from game to game:
  * money bags and police start positions are random every game
  * each robber has a random "bravery" (how much it fears police)
  * robbers sometimes change their mind about which bag to chase
  * police predict a robber's route and move to INTERCEPT it, but
    sometimes hesitate (human-like delay)
"""
import heapq
import math
import random
import threading
import time
from collections import deque
from dataclasses import dataclass

import solara

# =====================================================================
# SCENARIOS  (pick one in the UI; each one changes who is favoured)
# =====================================================================
TOTAL_MONEY = 5
MAX_TURNS = 120

SCENARIOS = {
    # Fair fight: roughly 50 / 50
    "Balanced": dict(
        ROBBER_SPEED=2, POLICE_SPEED=1, MONEY_NEEDED=3, GUARD_AT_MONEY=1,
        POLICE_MIN_START=6, POLICE_HESITATE=0.0, BRAVERY_RANGE=(0.4, 1.2)),
    # Robbers usually win: slow, sleepy police far away, only 2 bags needed, no exit guard
    "Robbers win": dict(
        ROBBER_SPEED=2, POLICE_SPEED=1, MONEY_NEEDED=2, GUARD_AT_MONEY=99,
        POLICE_MIN_START=8, POLICE_HESITATE=0.3, BRAVERY_RANGE=(0.4, 1.2)),
    # Police usually win: police as fast as robbers, start close, 4 bags needed, guard the exit
    "Police win": dict(
        ROBBER_SPEED=2, POLICE_SPEED=2, MONEY_NEEDED=4, GUARD_AT_MONEY=1,
        POLICE_MIN_START=4, POLICE_HESITATE=0.0, BRAVERY_RANGE=(0.4, 1.2)),
}

# Legend: # wall | . floor | E exit | R robber start
# (money bags and police are placed randomly each game)
LAYOUT = [
    "###############",
    "#R..#.....#..E#",
    "#R..#.....#...#",
    "#R............#",
    "#...#.....#...#",
    "##.###.#.###.##",
    "#.............#",
    "#.............#",
    "#.###.###.###.#",
    "#.#.#.#.#.#...#",
    "#.#.#.#.#.#...#",
    "#.............#",
    "###############",
]
DIRS = [(0, -1), (0, 1), (-1, 0), (1, 0)]


# =====================================================================
# A* SEARCH
# =====================================================================
def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def astar(walls, w, h, start, goal, extra=None):
    """A* on a 4-connected grid. f(n) = g(n) + h(n), h = Manhattan distance.
    `extra` maps cell -> additional cost (danger near police).
    Returns (path, cost); path excludes `start`. (None, inf) if unreachable."""
    extra = extra or {}
    if start == goal:
        return [], 0
    open_heap = [(manhattan(start, goal), 0, start)]
    best_g = {start: 0}
    came_from = {}
    while open_heap:
        _, g, cur = heapq.heappop(open_heap)
        if cur == goal:
            path = [cur]
            while path[-1] in came_from:
                path.append(came_from[path[-1]])
            path.pop()
            path.reverse()
            return path, g
        if g > best_g.get(cur, float("inf")):
            continue
        for dx, dy in DIRS:
            n = (cur[0] + dx, cur[1] + dy)
            if n in walls or not (0 <= n[0] < w and 0 <= n[1] < h):
                continue
            ng = g + 1 + extra.get(n, 0)
            if ng < best_g.get(n, float("inf")):
                best_g[n] = ng
                came_from[n] = cur
                heapq.heappush(open_heap, (ng + manhattan(n, goal), ng, n))
    return None, float("inf")


# =====================================================================
# GAME LOGIC
# =====================================================================
@dataclass
class Robber:
    name: str
    pos: tuple
    bravery: float = 1.0
    target: tuple = None
    status: str = "active"      # active | caught | escaped


class Game:
    def __init__(self, layout=LAYOUT, seed=None, scenario="Balanced"):
        self.scenario = scenario
        self.cfg = SCENARIOS[scenario]
        self.money_needed = self.cfg["MONEY_NEEDED"]
        self.max_turns = MAX_TURNS
        self.rng = random.Random(seed)
        self.h, self.w = len(layout), len(layout[0])
        self.walls, self.robbers = set(), []
        self.exit = None
        for y, row in enumerate(layout):
            for x, ch in enumerate(row):
                if ch == "#":
                    self.walls.add((x, y))
                elif ch == "E":
                    self.exit = (x, y)
                elif ch == "R":
                    self.robbers.append(Robber(
                        f"R{len(self.robbers) + 1}", (x, y),
                        bravery=round(self.rng.uniform(*self.cfg["BRAVERY_RANGE"]), 2)))
        floor = [(x, y) for y in range(self.h) for x in range(self.w)
                 if (x, y) not in self.walls]
        near = lambda c: min(manhattan(c, r.pos) for r in self.robbers)
        # random money (not on exit, not right on top of the robbers)
        spots = [c for c in floor if c != self.exit and near(c) >= 3]
        self.money_left = set(self.rng.sample(spots, TOTAL_MONEY))
        # random police start (far from the robbers' hideout)
        starts = [c for c in floor if near(c) >= self.cfg["POLICE_MIN_START"] and c not in self.money_left]
        self.police = self.rng.sample(starts, 2)
        self.total_money = TOTAL_MONEY
        self.money, self.turn = 0, 0
        self.over, self.counted = False, False
        self.result = ""
        self.log = ["Game started."]

    # ---------- helpers ----------
    def active(self):
        return [r for r in self.robbers if r.status == "active"]

    def say(self, msg):
        self.log.append(f"T{self.turn}: {msg}")

    def danger_map(self, bravery):
        """Extra A* cost for cells near police (robbers avoid these)."""
        extra = {}
        for p in self.police:
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    d = abs(dx) + abs(dy)
                    if d <= 2:
                        c = (p[0] + dx, p[1] + dy)
                        extra[c] = extra.get(c, 0) + {0: 50, 1: 15, 2: 5}[d] * bravery
        return extra

    def distances_from(self, start):
        """BFS distance from `start` to every reachable cell."""
        dist, q = {start: 0}, deque([start])
        while q:
            cur = q.popleft()
            for dx, dy in DIRS:
                n = (cur[0] + dx, cur[1] + dy)
                if n not in self.walls and n not in dist:
                    dist[n] = dist[cur] + 1
                    q.append(n)
        return dist

    # ---------- robber AI ----------
    def choose_target(self, r):
        extra = self.danger_map(r.bravery)
        taken = {o.target for o in self.active() if o is not r and o.target}
        cands = [m for m in self.money_left if m not in taken] or list(self.money_left)
        scored = []
        for m in cands:
            path, cost = astar(self.walls, self.w, self.h, r.pos, m, extra)
            if path is not None:
                scored.append((cost, m))
        scored.sort()
        if not scored:
            r.target = None
        elif len(scored) > 1 and self.rng.random() < 0.2:
            r.target = scored[1][1]         # sometimes not the greediest choice
        else:
            r.target = scored[0][1]

    def plan_robber(self, r):
        """A* path for a robber toward its goal (no side effects)."""
        goal = self.exit if self.money >= self.money_needed else r.target
        if goal is None:
            return None
        path, _ = astar(self.walls, self.w, self.h, r.pos, goal,
                        self.danger_map(r.bravery))
        return path

    def move_robber(self, r):
        for _ in range(self.cfg["ROBBER_SPEED"]):
            if self.over or r.status != "active":
                return
            if self.money < self.money_needed and (r.target is None or r.target not in self.money_left):
                self.choose_target(r)
            path = self.plan_robber(r)
            if not path:
                return
            r.pos = path[0]
            if r.pos in self.money_left:
                self.money_left.remove(r.pos)
                self.money += 1
                self.say(f"{r.name} grabbed money! Team total: {self.money}/{self.total_money}")
            if r.pos in self.police:
                r.status = "caught"
                self.say(f"{r.name} ran into police and was caught!")
                return
            if r.pos == self.exit and self.money >= self.money_needed:
                r.status = "escaped"
                self.over = True
                self.result = "ROBBERS WIN"
                self.say(f"{r.name} escaped with the loot!")
                return

    # ---------- police AI ----------
    def move_police(self, i, claimed):
        if self.rng.random() < self.cfg["POLICE_HESITATE"]:
            return                                   # hesitated this turn
        dist = self.distances_from(self.police[i])
        if i == 1 and self.money >= self.cfg["GUARD_AT_MONEY"]:  # officer 2 guards the exit
            path, _ = astar(self.walls, self.w, self.h, self.police[i], self.exit)
            for _ in range(self.cfg["POLICE_SPEED"]):
                if path:
                    self.police[i] = path.pop(0)
            for rob in self.active():
                if rob.pos == self.police[i]:
                    rob.status = "caught"
                    self.say(f"Police 2 caught {rob.name} at the exit!")
            return
        options = []                                 # (arrival_turn, robber, cell)
        for r in self.active():
            path = self.plan_robber(r) or []
            for idx, cell in enumerate(path, 1):     # predicted route of the robber
                t = math.ceil(idx / self.cfg["ROBBER_SPEED"])
                if cell in dist and dist[cell] <= t:  # police can be there first
                    options.append((t, r, cell))
                    break
        pool = [o for o in options if o[1].name not in claimed] or options
        if pool:
            _, r, goal = min(pool, key=lambda o: o[0])
        else:                                        # cannot intercept: just chase nearest
            reach = [r for r in self.active() if r.pos in dist]
            if not reach:
                return
            r = min(reach, key=lambda r: dist[r.pos])
            goal = r.pos
        claimed.add(r.name)
        path, _ = astar(self.walls, self.w, self.h, self.police[i], goal)
        for _ in range(self.cfg["POLICE_SPEED"]):
            if path:
                self.police[i] = path.pop(0)
        for rob in self.active():
            if rob.pos == self.police[i]:
                rob.status = "caught"
                self.say(f"Police {i + 1} caught {rob.name}!")

    # ---------- one turn ----------
    def step(self):
        if self.over:
            return
        self.turn += 1
        for r in self.active():                      # occasional change of mind
            if r.target not in self.money_left or self.rng.random() < 0.15:
                r.target = None
        for r in self.active():
            self.move_robber(r)
            if self.over:
                return
        claimed = set()
        for i in range(len(self.police)):
            self.move_police(i, claimed)
        if not self.active():
            self.over = True
            self.result = "POLICE WIN"
            self.say("All robbers caught!")
        elif self.turn >= self.max_turns:
            self.over = True
            self.result = "POLICE WIN (time out)"
            self.say("Time ran out - robbers failed to escape.")


def simulate(n=100, scenario="Balanced"):
    """Play n complete games and count who wins."""
    res = {"robbers": 0, "police": 0}
    for _ in range(n):
        g = Game(scenario=scenario)
        while not g.over:
            g.step()
        res["robbers" if g.result.startswith("ROBBERS") else "police"] += 1
    return res


# =====================================================================
# SOLARA UI
# =====================================================================
scenario = solara.reactive("Balanced")
game = solara.reactive(Game())
tick = solara.reactive(0)          # bumped to force a re-render after each step
playing = solara.reactive(False)
delay = solara.reactive(0.4)       # seconds between auto-play turns
stats = solara.reactive({"robbers": 0, "police": 0})
sim_text = solara.reactive("")


def do_step():
    g = game.value
    g.step()
    if g.over and not g.counted:
        g.counted = True
        key = "robbers" if g.result.startswith("ROBBERS") else "police"
        stats.value = {**stats.value, key: stats.value[key] + 1}
    tick.value += 1


def do_reset():
    playing.value = False
    game.value = Game(scenario=scenario.value)
    tick.value += 1


def do_change_scenario(name):
    scenario.value = name
    stats.value = {"robbers": 0, "police": 0}
    sim_text.value = ""
    do_reset()


def do_simulate():
    playing.value = False
    r = simulate(100, scenario.value)
    sim_text.value = f"[{scenario.value}] 100 games -> Robbers won {r['robbers']}, Police won {r['police']}"


def grid_html(g: Game) -> str:
    size = 36
    robbers = {r.pos for r in g.robbers if r.status == "active"}
    cells = []
    for y in range(g.h):
        for x in range(g.w):
            p = (x, y)
            bg, ch = "#f4f1e8", ""
            if p in g.walls:
                bg = "#4a4a52"
            elif p == g.exit:
                bg, ch = "#b7e4c7", "🚪"
            if p in g.money_left:
                ch = "💰"
            if p in robbers:
                ch = "🦹"
            if p in g.police:
                ch = "👮"
            cells.append(
                f'<div style="width:{size}px;height:{size}px;background:{bg};'
                f'display:flex;align-items:center;justify-content:center;'
                f'font-size:22px">{ch}</div>'
            )
    return (
        f'<div style="display:grid;grid-template-columns:repeat({g.w},{size}px);'
        f'gap:1px;background:#222;width:fit-content;border:3px solid #222">'
        + "".join(cells) + "</div>"
    )


@solara.component
def Page():
    _ = tick.value                 # subscribe to updates
    g = game.value

    def autoplay(cancel: threading.Event):
        while playing.value and not g.over and not cancel.is_set():
            time.sleep(delay.value)
            if cancel.is_set():
                return
            do_step()
        playing.value = False

    solara.use_thread(autoplay, dependencies=[playing.value, g])

    with solara.Column(style={"padding": "16px"}):
        solara.Title("Bank Robber Escape")
        solara.Markdown("# 🏦 Bank Robber Escape")
        with solara.Row(gap="32px", style={"flex-wrap": "wrap"}):
            solara.HTML(unsafe_innerHTML=grid_html(g))

            with solara.Column(style={"min-width": "300px"}):
                with solara.Row():
                    solara.Button("Step", on_click=do_step, disabled=g.over or playing.value)
                    solara.Button(
                        "Pause" if playing.value else "Play",
                        on_click=lambda: playing.set(not playing.value),
                        disabled=g.over,
                        color="primary",
                    )
                    solara.Button("New game", on_click=do_reset)
                solara.ToggleButtonsSingle(
                    value=scenario.value, values=list(SCENARIOS), on_value=do_change_scenario)
                solara.SliderFloat("Delay (s)", value=delay, min=0.1, max=1.5, step=0.1)

                solara.Markdown(
                    f"**Turn:** {g.turn} / {g.max_turns}  \n"
                    f"**Money:** {g.money} / {g.total_money} (need {g.money_needed})  \n"
                    + "  \n".join(f"**{r.name}:** {r.status} at {r.pos} (bravery {r.bravery})"
                                  for r in g.robbers)
                )
                if g.over:
                    (solara.Success if "ROBBERS" in g.result else solara.Error)(g.result)

                solara.Markdown(
                    f"**Scoreboard (this session):** Robbers {stats.value['robbers']} - "
                    f"Police {stats.value['police']}"
                )
                solara.Button("Simulate 100 games", on_click=do_simulate, outlined=True)
                if sim_text.value:
                    solara.Info(sim_text.value)

                solara.Markdown("**Event log**")
                solara.Markdown("  \n".join(reversed(g.log[-10:])))
                solara.Markdown("🦹 robber · 👮 police · 💰 money · 🚪 exit")
