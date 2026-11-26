"""
Zustand:   State(kind, station, line)
             kind = START   -> Startkoordinate (Zustand vor der ersten Aktion)
             kind = STATION -> man befindet sich in `station` auf `line`
             kind = GOAL    -> Zielkoordinate erreicht
Aktionen:  WALK_TO_STATION  (START   -> STATION), Dauer = Fussweg / 5 km/h
           TRAVEL           (STATION -> STATION, gleiche Linie), Dauer aus CSV
           TRANSFER         (STATION -> STATION, gleicher Bahnhof, andere Linie), 5 min
           WALK_TO_GOAL     (STATION -> GOAL), Dauer = Fussweg / 5 km/h
           WALK_DIRECT      (START   -> GOAL), Dauer = Fussweg / 5 km/h
Kosten:    Minuten (g = bisherige Dauer)
Heuristik: Luftlinie zum Ziel / v_max   (zulaessig, siehe `heuristic`)
Zieltest:  state.kind == GOAL

Dateistruktur (alles in einer Datei, damit es leicht zu starten ist):
1. Konfiguration & Datentypen
2. Geo-Hilfsfunktionen
3. UBahnGraph          (Netz aus CSV)
4. RouteProblem        (Zustaende, Aktionen, Zieltest, Heuristik)
5. a_star              (Suche inkl. p_max und Zeitlimit)
6. Logging / Ausgabe
7. Experimente a) - f)
8. main
"""

from __future__ import annotations

import csv
import heapq
import itertools
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

CSV_PATH = "A1-GKI-WS26-berliner_ubahn_geokoordinaten.csv"
EARTH_RADIUS_KM = 6371.0


# 1. KONFIGURATION & DATENTYPEN
@dataclass
class Config:
    #alle einstellbaren Parameter an einer Stelle (wichtig fuer Experimente d-f)
    walk_speed_kmh: float = 5.0          # laut Aufgabe
    transfer_min: float = 5.0            # laut Aufgabe (steht auch in der CSV)

    #Einschraenkung (1): maximale Pfadlaenge (Anzahl Aktionen ab Start)
    p_max: int = 40

    #Einschraenkung (2): Zeitlimit der Suche in Sekunden (< 120)
    time_limit_s: float = 5.0

    #UPDATE-Fussnote 1: Laufen nur zu "nahen" Stationen
    max_walk_to_station_min: float = 20.0
    #Alternativ/zusaetzlich: maximale Gesamtreisedauer (schliesst z.B. Spandau->Neukoelln zu Fuss aus)
    max_total_min: Optional[float] = None


class Kind(Enum):
    START = "start"
    STATION = "station"
    GOAL = "goal"


class ActionType(Enum):
    WALK_TO_STATION = "walk_to_station"
    TRAVEL = "travel"
    TRANSFER = "transfer"
    WALK_TO_GOAL = "walk_to_goal"
    WALK_DIRECT = "walk_direct"


class Status(Enum):
    #Ergebnis-Status nach Terminierung der Suche (Faelle aus Einschraenkung 1 und 2)
    OPTIMAL = "Optimale Loesung gefunden"
    POTENTIALLY_SUBOPTIMAL = "potenziell nicht optimale Loesung gefunden"
    NO_SOLUTION_MEMORY = "Loesung mit begrenztem Speicher nicht generierbar"
    TIMEOUT_WITH_SOLUTION = "Zeitlimit erreicht - beste gefundene Loesung (evtl. nicht optimal)"
    TIMEOUT_NO_SOLUTION = "Die Suche mit Zeitlimit hatte keinen Erfolg"


@dataclass(frozen=True)
class Coord:
    lat: float
    lon: float


@dataclass(frozen=True)
class State:
    #Hashbarer Suchzustand, `station`/`line` sind None bei START und GOAL
    kind: Kind
    station: Optional[str] = None
    line: Optional[str] = None


@dataclass(frozen=True)
class Edge:
    #Kante im U-Bahn-Graph (nur TRAVEL und TRANSFER, Laufwege entstehen dynamisch)
    action: ActionType
    src: State
    dst: State
    cost_min: float


@dataclass
class SearchNode:
    #Knoten im Suchbaum
    state: State
    g: float                              # bisherige Dauer in Minuten
    f: float                              # g + h
    depth: int                            # Pfadlaenge = Anzahl Aktionen seit Start
    parent: Optional["SearchNode"] = None
    action: Optional[ActionType] = None   # Aktion, die zu diesem Knoten fuehrte


@dataclass
class SearchStats:
    #Daten fuer Logging
    max_frontier_size: int = 0            # Speicherverbrauch: max. Anzahl Pfade in sortierter Liste
    expanded_states: int = 0              # Anzahl untersuchter (expandierter) Zustaende
    pruned_by_pmax: bool = False          # wurde irgendein Pfad wegen p_max nicht eingefuegt?
    elapsed_s: float = 0.0


@dataclass
class SearchResult:
    status: Status
    goal_node: Optional[SearchNode]
    stats: SearchStats
    start: Coord
    goal: Coord


# 2. GEO-HILFSFUNKTIONEN
def haversine_km(a: Coord, b: Coord) -> float:
    #Luftlinie in km zwischen zwei Geokoordinaten (Haversine-Formel, Erdradius 6371 km)
    lat1, lon1 = math.radians(a.lat), math.radians(a.lon)
    lat2, lon2 = math.radians(b.lat), math.radians(b.lon)
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    #min(1.0, ...) faengt Rundungsfehler ab, die sqrt/asin sonst aus dem Definitionsbereich treiben koennten
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(min(1.0, h)))


def walk_minutes(a: Coord, b: Coord, speed_kmh: float) -> float:
    #Fussweg-Dauer in Minuten (Luftlinie / Gehgeschwindigkeit)
    return haversine_km(a, b) / speed_kmh * 60


#3. GRAPH: U-BAHN-NETZ
class UBahnGraph:
    """
    Knoten  : State(STATION, station, line)  -> ein Bahnhof pro Linie
    Kanten  : TRAVEL (benachbarte Bahnhoefe einer Linie) und
              TRANSFER (gleicher Bahnhof, Linienwechsel, 5 min)
    """
    def __init__(self):
        self.coords = {}
        self.nachbar = {}
        self.lines_at_station = {}
        self.v_max_kmh = 5.0

    def load_csv(self, js):
        for j in js:
            start_line, start_station = j["from"].split(":", 1)
            to_line, to_station = j["to"].split(":", 1)
            start = State(Kind.STATION, start_station, start_line)
            target = State(Kind.STATION, to_station, to_line)
            action = ActionType(j["action"])
            cost_min = float(j["cost_min"].replace(",", "."))
            self.coords[start_station] = Coord(
                float(j["from_lat"].replace(",", ".")),
                float(j["from_lon"].replace(",", "."))
            )
            self.coords[to_station] = Coord(
                float(j["to_lat"].replace(",", ".")),
                float(j["to_lon"].replace(",", "."))
            )
            for station, line in [(start_station, start_line),
                                  (to_station, to_line)]:
                if station not in self.lines_at_station:
                    self.lines_at_station[station] = []
                if line not in self.lines_at_station[station]:
                    self.lines_at_station[station].append(line)
            if start not in self.nachbar:
                self.nachbar[start] = []
            self.nachbar[start].append(Edge(action, start, target, cost_min))
        self.compute_v_max()

    def compute_v_max(self):
        for state in self.nachbar:
            for edge in self.nachbar[state]:
                if edge.action == ActionType.TRAVEL and edge.cost_min > 0:
                    a = self.coords[edge.src.station]
                    b = self.coords[edge.dst.station]
                    speed = haversine_km(a, b) / edge.cost_min * 60
                    self.v_max_kmh = max(self.v_max_kmh, speed)

    def edges_from(self, state):
        return self.nachbar.get(state, [])

    def stations_near(self, c, max_walk_min, speed_kmh):
        nearby = []
        for station in self.coords:
            minutes = walk_minutes(c, self.coords[station], speed_kmh)
            if minutes <= max_walk_min:
                nearby.append((station, minutes))
        return nearby

    def coord_of(self, state):
        return self.coords[state.station]


#4. SUCHPROBLEM: ZUSTAENDE, AKTIONEN, ZIELTEST, HEURISTIK

class RouteProblem:
    """Kapselt das Suchproblem fuer beliebige Start und Zielkoordinaten."""

    def __init__(self, graph: UBahnGraph, start: Coord, goal: Coord, cfg: Config) -> None:
        self.graph = graph
        self.start = start
        self.goal = goal
        self.cfg = cfg
        self.initial_state = State(Kind.START)
        self._goal_state = State(Kind.GOAL)

        #Bahnhoefe nahe dem Ziel einmalig vorab berechnen
        #dadurch ist WALK_TO_GOAL in successors() ein O(1)-Lookup.
        self.goal_walk_cache: Dict[str, float] = dict(
            self.graph.stations_near(
                self.goal,
                self.cfg.max_walk_to_station_min,
                self.cfg.walk_speed_kmh,
            )
        )

    def is_goal(self, state: State) -> bool:
        """Zieltest: state.kind == Kind.GOAL."""
        return state.kind == Kind.GOAL

    def successors(self, state: State) -> List[Tuple[ActionType, State, float]]:
        """Alle Nachfolger (Aktion, Folgezustand, Kosten in min)."""
        result: List[Tuple[ActionType, State, float]] = []

        if state.kind == Kind.START:
            #direkt zum Ziel laufen
            direct = walk_minutes(self.start, self.goal, self.cfg.walk_speed_kmh)
            result.append((ActionType.WALK_DIRECT, self._goal_state, direct))
            #zu einer nahen Station laufen und dort in jede Linie einsteigen
            #(kein Umsteigeaufschlag beim Einsteigen)
            near = self.graph.stations_near(
                self.start,
                self.cfg.max_walk_to_station_min,
                self.cfg.walk_speed_kmh,
            )
            for station, walk_min in near:
                for line in self.graph.lines_at_station.get(station, []):
                    result.append(
                        (ActionType.WALK_TO_STATION, State(Kind.STATION, station, line), walk_min)
                    )

        elif state.kind == Kind.STATION:
            #TRAVEL / TRANSFER
            for edge in self.graph.edges_from(state):
                result.append((edge.action, edge.dst, edge.cost_min))

            #aussteigen und zum Ziel laufen, falls der Bahnhof nah genug am Ziel liegt
            walk_min = self.goal_walk_cache.get(state.station)
            if walk_min is not None:
                result.append((ActionType.WALK_TO_GOAL, self._goal_state, walk_min))

        #Kind.GOAL - keine Nachfolger
        return result

    def heuristic(self, state: State) -> float:
        """h(n) = Luftlinie zum Ziel / v_max * 60 (Minuten)."""
        if state.kind == Kind.GOAL:
            return 0.0

        if state.kind == Kind.START:
            here = self.start
        else:
            here = self.graph.coord_of(state)

        #max(...) schuetzt vor Division durch 0, falls v_max noch nicht berechnet wurde
        v_max = max(self.graph.v_max_kmh, self.cfg.walk_speed_kmh)
        return haversine_km(here, self.goal) / v_max * 60


#5. A*-SUCHE
def a_star(problem: RouteProblem) -> SearchResult:
    """
      - frontier:    heapq mit Eintraegen (f, tie_breaker, SearchNode)  -> 'sortierte Liste'
      - best_g:      dict State -> kleinstes bisher gefundenes g (Duplikat-/Zyklenvermeidung)
      - open_states: Zustaende, die aktuell mit ihrem *besten* Pfad in der Frontier liegen.
                     Noetig, weil heapq nur "lazy deletion" kann: ueberholte Eintraege bleiben
                     im Heap liegen, zaehlen aber nicht als Pfad der sortierten Liste.
                     len(open_states) ist damit der korrekte Speicherverbrauch.
      - counter:     itertools.count() als Tie-Breaker, damit SearchNodes nie verglichen werden
    """
    cfg = problem.cfg
    stats = SearchStats()
    counter = itertools.count()
    t0 = time.monotonic()

    def finish(status: Status, goal_node: Optional[SearchNode]) -> SearchResult:
        stats.elapsed_s = time.monotonic() - t0
        return SearchResult(status, goal_node, stats, problem.start, problem.goal)

    #1. Startknoten
    start_state = problem.initial_state
    start_node = SearchNode(
        state=start_state,
        g=0.0,
        f=problem.heuristic(start_state),
        depth=0,
    )
    frontier: List[Tuple[float, int, SearchNode]] = []
    heapq.heappush(frontier, (start_node.f, next(counter), start_node))
    best_g: Dict[State, float] = {start_state: 0.0}
    open_states = {start_state}
    stats.max_frontier_size = 1

    incumbent: Optional[SearchNode] = None   #beste bisher erzeugte Zielknoten - Loesung

    #2. Hauptschleife ----------------------------------------------------------
    while frontier:
        #a) Zeitlimit
        if time.monotonic() - t0 > cfg.time_limit_s:
            if incumbent is not None:
                return finish(Status.TIMEOUT_WITH_SOLUTION, incumbent)
            return finish(Status.TIMEOUT_NO_SOLUTION, None)

        #b) Knoten mit kleinstem f; ueberholte Eintraege ueberspringen
        _, _, node = heapq.heappop(frontier)
        state = node.state
        if node.g > best_g.get(state, math.inf):
            continue
        open_states.discard(state)

        #c) Zieltest beim Poppen (nur so ist die Loesung bei zulaessiger h optimal)
        if problem.is_goal(state):
            if stats.pruned_by_pmax:
                return finish(Status.POTENTIALLY_SUBOPTIMAL, node)   # Fall c
            return finish(Status.OPTIMAL, node)                      # Fall b

        #d) Expansion
        stats.expanded_states += 1

        #e) Nachfolger
        for action, nxt, cost in problem.successors(state):
            g_new = node.g + cost
            depth_new = node.depth + 1

            #prüft Zeitlimit auch während verarbeitung der einzelnen Nachfolger
            if time.monotonic() - t0 >= cfg.time_limit_s:
                if incumbent is not None:
                    return finish(Status.TIMEOUT_WITH_SOLUTION, incumbent)
                return finish(Status.TIMEOUT_NO_SOLUTION, None)

            #Maximale Gesamtreisedauer (UPDATE-Fussnote 1)
            if cfg.max_total_min is not None and g_new > cfg.max_total_min:
                continue
            #Einschraenkung (1) - Pfad zu lang -> nicht einfuegen, Suche laeuft weiter
            if depth_new > cfg.p_max:
                stats.pruned_by_pmax = True
                continue
            #Duplikatvermeidung - Zustand schon mit <= g bekannt
            if g_new >= best_g.get(nxt, math.inf):
                continue
            child = SearchNode(
                state=nxt,
                g=g_new,
                f=g_new + problem.heuristic(nxt),
                depth=depth_new,
                parent=node,
                action=action,
            )
            best_g[nxt] = g_new
            open_states.add(nxt)
            heapq.heappush(frontier, (child.f, next(counter), child))
            if problem.is_goal(nxt) and (incumbent is None or g_new < incumbent.g):
                incumbent = child

        #f) Speicherverbrauch = maximale Zahl gleichzeitiger Pfade in der sortierten Liste
        stats.max_frontier_size = max(stats.max_frontier_size, len(open_states))

    #3. Frontier leer ohne Loesung -> Fall a
    return finish(Status.NO_SOLUTION_MEMORY, None)


def reconstruct_path(node: SearchNode) -> List[SearchNode]:
    #folgt parent-Zeigern vom Ziel zum Start und gibt die Knoten in Reihenfolge Start->Ziel zurueck
    path = []
    current = node
    while current is not None: 
        path.append(current)
        current = current.parent

    path.reverse()
    return path 

#6. AUSGABE
def print_result(result: SearchResult, cfg: Config) -> None:
    """Konsolenausgabe laut Aufgabe.
    - Start- und Zielkoordinaten
    - Status-Meldung (result.status.value), bei NO_SOLUTION_MEMORY zusätzlich
      "Mit der maximalen Pfadlänge kann eine Lösung nicht generiert werden"
    - Lösungspfad, falls vorhanden: Aktion, Bahnhof/Linie, Dauer und Gesamtdauer
    - Speicherverbrauch = stats.max_frontier_size
    - Anzahl untersuchter Zustände = stats.expanded_states
    - Suchdauer und verwendete Limits (p_max, time_limit_s)
    """
    
    print("\n" + "=" * 50)
    print("           A* ROUTENPLANUNG")
    print("=" * 50)

    #Start/zielkoordinaten angeben
    print(f"Start: {result.start.lat:.6f}, {result.start.lon:.6f}")
    print(f"Ziel:  {result.goal.lat:.6f}, {result.goal.lon:.6f}")

    #Suchstatus
    print(f"\nStatus: {result.status.value}")

    if result.status == Status.NO_SOLUTION_MEMORY:
        print("Mit der maximalen Pfadlaenge kann eine Loesung nicht generiert werden.")

    #Route ausgeben
    if result.goal_node is not None:
        path = reconstruct_path(result.goal_node)
        print("\nROUTE:")
        for i in range(1, len(path)):
            vorher = path[i - 1]
            aktuell = path[i]
            dauer = aktuell.g - vorher.g
            aktion = aktuell.action
            if aktion == ActionType.WALK_TO_STATION:
                print(
                    f"{i}. Laufen zu {aktuell.state.station} "
                    f"({dauer:.2f} min)"
                )
            elif aktion == ActionType.TRAVEL:
                print(
                    f"{i}. U-Bahn {aktuell.state.line}: "
                    f"{vorher.state.station} -> "
                    f"{aktuell.state.station} "
                    f"({dauer:.2f} min)"
                )
            elif aktion == ActionType.TRANSFER:
                print(
                    f"{i}. Umsteigen in {aktuell.state.station}: "
                    f"{vorher.state.line} -> "
                    f"{aktuell.state.line} "
                    f"({dauer:.2f} min)"
                )
            elif aktion == ActionType.WALK_TO_GOAL:
                print(
                    f"{i}. Laufen zum Ziel "
                    f"({dauer:.2f} min)"
                )
            elif aktion == ActionType.WALK_DIRECT:
                print(
                    f"{i}. Direkt zum Ziel laufen "
                    f"({dauer:.2f} min)"
                )

        print(f"\nGesamtdauer: {result.goal_node.g:.2f} Minuten")

    else:
        print("\nKeine Route gefunden.")

    #Angeben von Stats
    print("\nSTATISTIKEN:")
    print(f"Untersuchte Zustaende: {result.stats.expanded_states}")
    print(f"Maximale Frontier-Groesse: {result.stats.max_frontier_size}")
    print(f"Suchdauer: {result.stats.elapsed_s:.4f} Sekunden")

    #Angeben der Constraints
    print("\nCONSTRAINTS:")
    print(f"Maximale Pfadlaenge (p_max): {cfg.p_max}")
    print(f"Zeitlimit: {cfg.time_limit_s:.2f} Sekunden")
    print(f"Pfadbegrenzung aktiv geworden: "
          f"{'Ja' if result.stats.pruned_by_pmax else 'Nein'}")
    print("=" * 50)

#7. EXPERIMENTE (Edge-Cases d-f)
@dataclass
class Experiment:
    name: str
    description: str
    start: Coord
    goal: Coord
    cfg: Config
    expected: str
    #expected


def build_experiments() -> List[Experiment]:
    default = Config()

    #Langstrecke Rathaus Spandau -> Rudow (U7 von Ende zu Ende, ca. 24 km Luftlinie)
    spandau = Coord(52.5360, 13.2010)
    rudow = Coord(52.4150, 13.4960)

    return [
        Experiment(
            "a", "Laufen ist optimal, A* untersucht aber auch U-Bahn-Pfade",
            Coord(52.5035, 13.3290),      #Kurfuerstendamm, zwischen Uhlandstr. und Kurfuerstendamm
            Coord(52.5000, 13.3340),      #ca. 0,5 km Luftlinie (ca. 6 min zu Fuss)
            default,
            "Loesung = WALK_DIRECT (ca. 6 min); expanded_states enthaelt auch STATION-Zustaende, "
            "da h sehr optimistisch ist (Luftlinie / v_max)"),
        Experiment(
            "b", "Optimale Loesung: Fahren ohne Umsteigen",
            Coord(52.5320, 13.3830),      #nahe U Naturkundemuseum (U6)
            Coord(52.4860, 13.3860),      #nahe U Platz der Luftbruecke (U6), ca. 5 km
            default,
            "Laufen -> U6 -> Laufen, kein TRANSFER im Pfad"),
        Experiment(
            "c", "Optimale Loesung: genau 1x Umsteigen",
            Coord(52.4620, 13.3290),      #nahe U Schlossstrasse (U9)
            Coord(52.5225, 13.4130),      #nahe Alexanderplatz
            default,
            "Laufen -> U9 -> TRANSFER (Zoologischer Garten) -> U2 -> Laufen"),
        Experiment(
            "d", "p_max-Pruning aktiv, Loesung gefunden => potenziell nicht optimal",
            spandau, rudow,
            Config(p_max=30),
            "Status POTENTIALLY_SUBOPTIMAL: Die optimale U7-Fahrt braucht ca. 40 Aktionen, "
            "wird bei p_max=30 abgeschnitten; gefunden wird nur eine teurere Loesung "
            "(z.B. WALK_DIRECT oder U7-Teilstrecke + langer Fussweg)"),
        Experiment(
            "e", "p_max so klein, dass keine Loesung existiert",
            spandau, rudow,
            Config(p_max=5, max_total_min=120.0),
            "Status NO_SOLUTION_MEMORY + Zusatzhinweis. max_total_min=120 schliesst WALK_DIRECT "
            "(ca. 290 min) aus; mit hoechstens 5 Aktionen ist Rudow per U-Bahn nicht erreichbar"),
        Experiment(
            "f", "Zeitlimit bricht die Suche ab",
            spandau, rudow,
            Config(time_limit_s=0.001),
            "Status TIMEOUT_WITH_SOLUTION oder TIMEOUT_NO_SOLUTION (maschinenabhaengig). "
            "Falls die Suche trotzdem fertig wird: time_limit_s weiter senken"),
    ]


def run_experiment(graph: UBahnGraph, exp: Experiment) -> SearchResult:
    print("=" * 78)
    print(f"Experiment {exp.name}) {exp.description}")
    print(f"Erwartet: {exp.expected}")
    print("-" * 78)
    problem = RouteProblem(graph, exp.start, exp.goal, exp.cfg)
    result = a_star(problem)
    print_result(result, exp.cfg)
    return result

# 8. MAIN
def main():
    #CSV read
    with open(CSV_PATH, encoding="utf-8-sig") as file:
        reader = csv.DictReader(file, delimiter=";")
        data = list(reader)
    for row in data[:5]:
        print(row)
    actions = set()
    lines = set()
    stations = set()
    for row in data:
        actions.add(row["action"])
        if row["line"]:
            lines.add(row["line"])
        stations.add(row["from"])
        stations.add(row["to"])

    #Graph aufbauen
    graph = UBahnGraph()
    graph.load_csv(data)

    #Experimente
    for exp in build_experiments():
        run_experiment(graph, exp)


if __name__ == "__main__":
    main()