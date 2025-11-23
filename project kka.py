# coding: utf-8
"""
Sistem pencarian rute Surabaya dengan Multi-Modal Transport
Mendukung: Bus, Trotoar Pejalan Kaki, dan Ojek Online
Dengan sistem penalty dan transit untuk aksesibilitas kursi roda
"""

import math
import heapq
from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass
from enum import Enum

class TransportMode(Enum):
    BUS = "Bus"
    SIDEWALK = "Trotoar"
    OJEK = "Ojek Online"

@dataclass
class EdgeInfo:
    distance: float
    modes: Set[TransportMode]  # Available transport modes
    wheelchair_accessible: bool = True

ROWS = ["A", "B", "C", "D", "E", "F", "G"]
COLS = ["1", "2", "3", "4"]
NODES = [r + c for r in ROWS for c in COLS]

# EDGES dengan informasi moda transportasi
# Format: node_a, node_b, distance, available_modes (set), wheelchair_accessible
EDGES_WITH_MODES: List[Tuple[str, str, float, Set[TransportMode], bool]] = [

    ("A1", "A2", 1.000, {TransportMode.BUS, TransportMode.SIDEWALK}, True),
    ("A1", "F4", 0.800, {TransportMode.OJEK}, True),
    
    ("A2", "B1", 0.550, {TransportMode.SIDEWALK}, True),
    ("A2", "A3", 1.000, {TransportMode.BUS, TransportMode.SIDEWALK}, True),
    ("A2", "E1", 1.000, {TransportMode.OJEK}, True),
    
    ("A3", "A4", 1.000, {TransportMode.BUS, TransportMode.SIDEWALK}, True),
    ("A3", "D1", 0.650, {TransportMode.SIDEWALK}, True),
    
    ("A4", "C1", 0.950, {TransportMode.SIDEWALK}, True),
    
    ("B1", "B3", 0.650, {TransportMode.SIDEWALK}, True),
    ("B1", "D1", 0.950, {TransportMode.SIDEWALK}, True),
    
    ("B3", "B4", 0.750, {TransportMode.SIDEWALK}, True),
    ("B4", "G2", 1.500, {TransportMode.OJEK}, True),
    ("B4", "E1", 2.800, {TransportMode.OJEK}, True),
    
    ("C1", "C2", 0.550, {TransportMode.SIDEWALK}, True),
    ("C2", "D2", 1.500, {TransportMode.SIDEWALK}, True),
    ("C2", "D4", 0.450, {TransportMode.SIDEWALK}, True),
    
    ("D1", "D2", 0.650, {TransportMode.SIDEWALK}, True),
    ("D4", "E1", 0.650, {TransportMode.SIDEWALK}, True),
    
    ("E1", "F2", 1.700, {TransportMode.OJEK}, True), # Tidak accessible via sidewalk
    
    ("F1", "F2", 0.700, {TransportMode.BUS, TransportMode.SIDEWALK}, True),
    ("F2", "F3", 0.500, {TransportMode.BUS, TransportMode.SIDEWALK}, True),
    ("F3", "F4", 0.350, {TransportMode.BUS, TransportMode.SIDEWALK}, True),
    ("F3", "G1", 1.000, {TransportMode.SIDEWALK}, True),
    
    ("G1", "G2", 1.000, {TransportMode.SIDEWALK}, True),
]

COORDS: Dict[str, Tuple[float, float]] = {
    "A1": (0.0, 0.0), 
    "A2": (1.0, 0.0), 
    "A3": (2.0, 0.0), 
    "A4": (3.0, 0.0),
    "B1": (0.0, 0.65), 
    "B3": (2.0, 0.65), 
    "B4": (3.0, 0.65),
    "C1": (0.0, 1.30), 
    "C2": (1.0, 1.30),
    "D1": (0.0, 1.95), 
    "D2": (1.0, 1.95), 
    "D4": (3.0, 1.95),
    "E1": (0.0, 2.60),
    "F1": (0.0, 3.25), 
    "F2": (1.0, 3.25), 
    "F3": (2.0, 3.25), 
    "F4": (3.0, 3.25),
    "G1": (0.0, 3.90), 
    "G2": (1.0, 3.90),
}

# Special nodes - Anda bisa ubah ini untuk menambahkan/mengurangi lokasi
SPECIAL_NODES: Dict[str, List[str]] = {
    "ITS": ["A1", "F4"], # Simplified access
    "Galaxy Mall": ["B1", "D1"],
    "Pakuwon City Mall": ["F1"],
    "Universitas Airlangga - Kampus B": ["C1"],
    "RSUD Dr. Soetomo": ["D4", "C2"],
}

SPECIAL_ATTACH_DIST_KM = 0.100

# Penalty system (multiplier untuk cost)
PENALTIES: Dict[TransportMode, float] = {
    TransportMode.BUS: 1.0,        
    TransportMode.SIDEWALK: 1.5,   
    TransportMode.OJEK: 3.5,       
}

# Penalty tambahan untuk consecutive sidewalk usage
CONSECUTIVE_SIDEWALK_PENALTY = 0.8  

# Transit penalty (switching modes)
TRANSIT_PENALTY = 0.5  # Biaya setara km untuk berpindah moda

def empty_graph(nodes: List[str]) -> Dict[str, Dict[str, EdgeInfo]]:
    return {n: {} for n in nodes}

def add_edge(graph: Dict[str, Dict[str, EdgeInfo]], a: str, b: str, edge_info: EdgeInfo) -> None:
    if a not in graph: graph[a] = {}
    if b not in graph: graph[b] = {}
    
    if b not in graph[a]:
        graph[a][b] = edge_info
        graph[b][a] = edge_info

def build_graph() -> Tuple[Dict[str, Dict[str, EdgeInfo]], Dict[str, float]]:
    all_current_nodes = set(NODES)
    
    for special_name in SPECIAL_NODES.keys():
        all_current_nodes.add(special_name)
    
    graph_map = empty_graph(list(all_current_nodes))
    
    for a, b, d, modes, wheelchair in EDGES_WITH_MODES:
        edge_info = EdgeInfo(distance=d, modes=modes, wheelchair_accessible=wheelchair)
        add_edge(graph_map, a, b, edge_info)
    
    for special_name, connect_nodes in SPECIAL_NODES.items():
        for connect_node in connect_nodes:
            if connect_node in NODES:
                edge_info = EdgeInfo(
                    distance=SPECIAL_ATTACH_DIST_KM,
                    modes={TransportMode.BUS, TransportMode.SIDEWALK, TransportMode.OJEK},
                    wheelchair_accessible=True
                )
                add_edge(graph_map, special_name, connect_node, edge_info)
    
    for special_name, connect_nodes in SPECIAL_NODES.items():
        if special_name not in COORDS:
            valid_coords = [COORDS[node] for node in connect_nodes if node in COORDS]
            if valid_coords:
                avg_x = sum(c[0] for c in valid_coords) / len(valid_coords)
                avg_y = sum(c[1] for c in valid_coords) / len(valid_coords)
                COORDS[special_name] = (avg_x, avg_y)
    
    straight_line_dist: Dict[str, float] = {n: 0.0 for n in graph_map}
    
    return graph_map, straight_line_dist

def compute_euclid_heuristic(coords: Dict[str, Tuple[float,float]], goal: str) -> Dict[str, float]:
    heur = {}
    if goal not in coords:
        return {n: 0.0 for n in coords}
    
    gx, gy = coords[goal]
    for n in coords:
        if n in coords:
            x, y = coords[n]
            heur[n] = math.hypot(x - gx, y - gy)
        else:
            heur[n] = 0.0
    
    return heur

@dataclass
class PathNode:
    node: str
    mode: Optional[TransportMode]
    cumulative_cost: float
    actual_distance: float
    
    def __lt__(self, other):
        return self.cumulative_cost < other.cumulative_cost

def multimodal_a_star(
    graph_map: Dict[str, Dict[str, EdgeInfo]],
    start: str,
    goal: str,
    wheelchair_mode: bool = True
) -> Tuple[Optional[List[PathNode]], Optional[float], Optional[float]]:
    """
    A* search dengan multi-modal transport dan transit support
    Returns: (path, total_cost, actual_distance)
    """
    if start not in graph_map or goal not in graph_map:
        return None, None, None
    
    heur = compute_euclid_heuristic(COORDS, goal) if COORDS else {}
    
    counter = 0
    
    # State: (f_cost, counter, g_cost, actual_distance, path, current_mode, consecutive_sidewalk_count)
    start_node = PathNode(node=start, mode=None, cumulative_cost=0.0, actual_distance=0.0)
    priority_queue: List[Tuple[float, int, float, float, List[PathNode], Optional[TransportMode], int]] = [
        (heur.get(start, 0.0), counter, 0.0, 0.0, [start_node], None, 0)
    ]
    counter += 1
    
    # visited: (node, mode, consecutive_sidewalk) -> cost
    visited_costs: Dict[Tuple[str, Optional[TransportMode], int], float] = {}
    
    while priority_queue:
        f_cost, _, g_cost, actual_dist, path, current_mode, consec_sidewalk = heapq.heappop(priority_queue)
        current_node = path[-1].node
        
        state = (current_node, current_mode, consec_sidewalk)
        if state in visited_costs and g_cost > visited_costs[state]:
            continue
        
        if current_node == goal:
            return path, g_cost, actual_dist
        
        for neighbor, edge_info in graph_map.get(current_node, {}).items():
            
            if neighbor in SPECIAL_NODES and current_node in SPECIAL_NODES.get(neighbor, []):
                continue
            
            for mode in edge_info.modes:
                if wheelchair_mode and not edge_info.wheelchair_accessible:
                    continue
                
                edge_cost = edge_info.distance * PENALTIES[mode]
                
                new_consec_sidewalk = 0
                if mode == TransportMode.SIDEWALK:
                    if current_mode == TransportMode.SIDEWALK:
                        new_consec_sidewalk = consec_sidewalk + 1
                    else:
                        new_consec_sidewalk = 1 # Start new sidewalk segment

                consecutive_penalty = 0.0
                if new_consec_sidewalk > 1:
                    consecutive_penalty = CONSECUTIVE_SIDEWALK_PENALTY * (new_consec_sidewalk - 1)
                
                transit_cost = 0.0
                if current_mode is not None and current_mode != mode:
                    transit_cost = TRANSIT_PENALTY
                
                new_gCost = g_cost + edge_cost + transit_cost + consecutive_penalty
                new_actual_dist = actual_dist + edge_info.distance
                
                new_state = (neighbor, mode, new_consec_sidewalk)
                if new_state not in visited_costs or new_gCost < visited_costs[new_state]:
                    visited_costs[new_state] = new_gCost
                    f_cost_new = new_gCost + heur.get(neighbor, 0.0)
                    
                    new_path_node = PathNode(
                        node=neighbor,
                        mode=mode,
                        cumulative_cost=new_gCost,
                        actual_distance=new_actual_dist
                    )
                    new_path = path + [new_path_node]
                    heapq.heappush(priority_queue, (f_cost_new, counter, new_gCost, new_actual_dist, new_path, mode, new_consec_sidewalk))
                    counter += 1
    
    return None, None, None

def get_mode_icon(mode: Optional[TransportMode]) -> str:
    """Returns the text abbreviation for the transport mode."""
    if mode is None:
        return ""
    icons = {
        TransportMode.BUS: "BUS",
        TransportMode.SIDEWALK: "TROT",
        TransportMode.OJEK: "OJEK"
    }
    return icons.get(mode, "N/A")

def analyze_route(path: List[PathNode]) -> Dict[str, Any]:
    """Analyze route for statistics"""
    if not path or len(path) < 2:
        return {
            "mode_distances": {mode: 0.0 for mode in TransportMode},
            "mode_segments": {mode: 0 for mode in TransportMode},
            "transits": 0,
            "max_consecutive_sidewalk": 0
        }
    
    mode_distances = {mode: 0.0 for mode in TransportMode}
    mode_segments = {mode: 0 for mode in TransportMode}
    transits = 0
    consecutive_sidewalk_segments = 0
    max_consecutive_sidewalk = 0
    
    for i in range(1, len(path)):
        prev_mode = path[i-1].mode
        curr_mode = path[i].mode
        
        if curr_mode:
            segment_dist = path[i].actual_distance - path[i-1].actual_distance
            mode_distances[curr_mode] += segment_dist
            mode_segments[curr_mode] += 1
            
            if curr_mode == TransportMode.SIDEWALK:
                if prev_mode == TransportMode.SIDEWALK:
                    consecutive_sidewalk_segments += 1
                else:
                    consecutive_sidewalk_segments = 1
                max_consecutive_sidewalk = max(max_consecutive_sidewalk, consecutive_sidewalk_segments)
            else:
                consecutive_sidewalk_segments = 0
        
        if prev_mode is not None and curr_mode is not None and prev_mode != curr_mode:
            transits += 1
    
    return {
        "mode_distances": mode_distances,
        "mode_segments": mode_segments,
        "transits": transits,
        "max_consecutive_sidewalk": max_consecutive_sidewalk
    }

def print_result(start: str, goal: str, best_path: List[PathNode], best_actual_dist: float, best_cost: float, wheelchair_mode: bool):
    """Prints the final result in a structured format."""
    print("\n" + "=" * 80)
    print("HASIL RUTE OPTIMAL DITEMUKAN" + (" (Mode Kursi Roda)" if wheelchair_mode else " (Mode Umum)"))
    print("Start: " + start + " | Goal: " + goal)
    print("=" * 80)
    
    stats = analyze_route(best_path)
    
    if start in SPECIAL_NODES and len(best_path) > 1:
        entry_node = best_path[1].node
        print(f"-> '{start}' (Pintu masuk: {entry_node})")
    
    if goal in SPECIAL_NODES and len(best_path) > 1:
        exit_node = best_path[-2].node
        print(f"-> '{goal}' (Pintu keluar: {exit_node})")

    print("\n" + "-" * 80)
    print("DETAIL LANGKAH:")
    print("-" * 80)
    
    for i, path_node in enumerate(best_path):
        current_node_name = path_node.node
        mode_name = get_mode_icon(path_node.mode)
        
        if i == 0:
            print(f" {i+1}. START di {current_node_name}")
        elif i == len(best_path) - 1:
            if start in SPECIAL_NODES and i > 1:
                dist_increment = path_node.actual_distance - best_path[i-1].actual_distance
                print(f" {i}. {best_path[i-1].node} -> {current_node_name} | {mode_name} | +{dist_increment:.3f}km [FINAL SEGMENT]")
            print(f" {i+1}. GOAL di {current_node_name} (Selesai)")
            print(f"      [Total Jarak: {best_actual_dist:.3f} km]")
        else:
            prev_node_name = best_path[i-1].node
            dist_increment = path_node.actual_distance - best_path[i-1].actual_distance
            
            transit_marker = ""
            if i > 1 and best_path[i-1].mode != path_node.mode:
                prev_mode_name = get_mode_icon(best_path[i-1].mode)
                transit_marker = f" [TRANSIT dari {prev_mode_name} ke {mode_name}]"
            
            print(f" {i+1}. {prev_node_name} -> {current_node_name} | {mode_name} | +{dist_increment:.3f}km{transit_marker}")
            print(f"      [Cost: {path_node.cumulative_cost:.3f}]")

    print("\n" + "=" * 80)
    print("STATISTIK PERJALANAN:")
    print("-" * 80)
    print(f"Total jarak aktual: {best_actual_dist:.3f} km")
    print(f"Total cost (dengan penalty): {best_cost:.3f} (Angka yang lebih rendah lebih baik)")
    print(f"Jumlah transit (ganti moda): {stats['transits']}")
    
    print("\nPer Moda Transportasi:")
    for mode in TransportMode:
        if stats['mode_distances'][mode] > 0:
            dist = stats['mode_distances'][mode]
            segments = stats['mode_segments'][mode]
            penalty = PENALTIES[mode]
            print(f"  - {mode.value:15s}: {dist:.3f}km ({segments} segmen, Penalty: {penalty:.1f}x)")
    
    print("\nREKOMENDASI:")
    if wheelchair_mode:
        if stats['max_consecutive_sidewalk'] > 1:
            print(f"  Peringatan: Rute menggunakan trotoar selama {stats['max_consecutive_sidewalk']} segmen berturut-turut.")
            print("  Saran: Pertimbangkan Ojek Online untuk kenyamanan dan menghemat tenaga.")
        
        if stats['mode_distances'][TransportMode.OJEK] > 0:
            print("  Peringatan: Rute menggunakan Ojek Online.")
            print("  Saran: Pertimbangkan transit Bus/Trotoar jika ingin biaya lebih rendah.")


def handle_route_search(graph_map: Dict[str, Dict[str, EdgeInfo]], all_nodes: List[str]):
    """Handles the interactive route search logic."""
    print("\nCARI RUTE MULTI-MODAL")
    print("-" * 80)
    
    print("\nMode pengguna:")
    print("  1. Pengguna kursi roda (hanya jalur accessible)")
    print("  2. Pengguna umum (semua jalur)")
    wheelchair_choice = input("Pilih (1/2, default=1): ").strip()
    wheelchair_mode = wheelchair_choice != "2"
    
    print("\nDaftar lokasi:")
    for i, n in enumerate(all_nodes, start=1):
        marker = "[L]" if n in SPECIAL_NODES else "[N]" # L for Location, N for Node
        print(f"  {i:2d}. {marker} {n}")
    
    print("\nMasukkan nomor atau nama lokasi:")
    s = input("  START (dari): ").strip()
    t = input("  GOAL (tujuan): ").strip()
    
    def parse_choice(x: str) -> Optional[str]:
        if x.isdigit():
            idx = int(x) - 1
            if 0 <= idx < len(all_nodes):
                return all_nodes[idx]
        for node in all_nodes:
            if x.lower() == node.lower() or (len(x) > 1 and x.lower() in node.lower()):
                return node
        return x
    
    start = parse_choice(s)
    goal = parse_choice(t)
    
    if start not in graph_map or goal not in graph_map:
        print(f"\nError: Lokasi '{start}' atau '{goal}' tidak valid!")
        return
    
    start_nodes = SPECIAL_NODES[start] if start in SPECIAL_NODES else [start]
    goal_nodes = SPECIAL_NODES[goal] if goal in SPECIAL_NODES else [goal]
    
    print(f"\nMencari rute optimal dengan transit support...")
    if wheelchair_mode:
        print("-> Mode: Kursi roda (hanya jalur accessible)")
    
    best_path: Optional[List[PathNode]] = None
    best_cost = float('inf')
    best_actual_dist = None
    
    for s_node in start_nodes:
        for g_node in goal_nodes:
            path, cost, actual_dist = multimodal_a_star(graph_map, s_node, g_node, wheelchair_mode)
            
            if path and cost is not None and actual_dist is not None:
                total_cost = cost
                total_actual = actual_dist
                
                is_start_special = start in SPECIAL_NODES and s_node != start
                is_goal_special = goal in SPECIAL_NODES and g_node != goal

                if is_start_special:
                    total_cost += SPECIAL_ATTACH_DIST_KM * PENALTIES[TransportMode.BUS]
                    total_actual += SPECIAL_ATTACH_DIST_KM
                if is_goal_special:
                    total_cost += SPECIAL_ATTACH_DIST_KM * PENALTIES[TransportMode.BUS]
                    total_actual += SPECIAL_ATTACH_DIST_KM

                if total_cost < best_cost:
                    best_cost = total_cost
                    best_actual_dist = total_actual
                    
                    full_path = []
                    if is_start_special:
                        full_path.append(PathNode(start, None, 0, 0))
                    full_path.extend(path)
                    if is_goal_special:
                        last_mode = path[-1].mode if path else None
                        full_path.append(PathNode(goal, last_mode, total_cost, total_actual))
                    best_path = full_path
    
    if not best_path:
        print("\nRute tidak ditemukan!")
        print("Kemungkinan: Tidak ada jalur yang accessible atau terhubung.")
        return
    
    print_result(start, goal, best_path, best_actual_dist, best_cost, wheelchair_mode)



def multimodal_greedy_bfs(
    graph_map: Dict[str, Dict[str, EdgeInfo]],
    start: str,
    goal: str,
    wheelchair_mode: bool = True
) -> Tuple[Optional[List[PathNode]], Optional[float], Optional[float]]:
    """Greedy Best-First Search dengan multi-modal transport"""
    if start not in graph_map or goal not in graph_map:
        return None, None, None

    heur = compute_euclid_heuristic(COORDS, goal)
    visited: Set[str] = set()
    queue: List[Tuple[float, List[PathNode], Optional[TransportMode]]] = []

    start_node = PathNode(node=start, mode=None, cumulative_cost=0.0, actual_distance=0.0)
    queue.append((heur.get(start, 0.0), [start_node], None))

    while queue:
        queue.sort(key=lambda x: x[0])  # Sort by heuristic only
        _, path, current_mode = queue.pop(0)
        current_node = path[-1].node

        if current_node == goal:
            total_dist = path[-1].actual_distance
            return path, None, total_dist

        if current_node in visited:
            continue
        visited.add(current_node)

        for neighbor, edge_info in graph_map.get(current_node, {}).items():
            if wheelchair_mode and not edge_info.wheelchair_accessible:
                continue

            for mode in edge_info.modes:
                dist = edge_info.distance
                new_actual_dist = path[-1].actual_distance + dist
                new_node = PathNode(node=neighbor, mode=mode, cumulative_cost=0.0, actual_distance=new_actual_dist)
                new_path = path + [new_node]
                queue.append((heur.get(neighbor, 0.0), new_path, mode))

    return None, None, None


def handle_route_search_greedy(graph_map: Dict[str, Dict[str, EdgeInfo]], all_nodes: List[str]):
    print("CARI RUTE CEPAT (Greedy BFS)")
    print("-" * 80)

    print("Mode pengguna:")
    print("  1. Pengguna kursi roda (hanya jalur accessible)")
    print("  2. Pengguna umum (semua jalur)")
    wheelchair_choice = input("Pilih (1/2, default=1): ").strip()
    wheelchair_mode = wheelchair_choice != "2"

    print("Daftar lokasi:")
    for i, n in enumerate(all_nodes, start=1):
        marker = "[L]" if n in SPECIAL_NODES else "[N]"
        print(f"  {i:2d}. {marker} {n}")

    print("Masukkan nomor atau nama lokasi:")
    s = input("  START (dari): ").strip()
    t = input("  GOAL (tujuan): ").strip()

    def parse_choice(x: str) -> Optional[str]:
        if x.isdigit():
            idx = int(x) - 1
            if 0 <= idx < len(all_nodes):
                return all_nodes[idx]
        for node in all_nodes:
            if x.lower() == node.lower() or (len(x) > 1 and x.lower() in node.lower()):
                return node
        return x

    start = parse_choice(s)
    goal = parse_choice(t)

    if start not in graph_map or goal not in graph_map:
        print(f"Error: Lokasi '{start}' atau '{goal}' tidak valid!")
        return

    start_nodes = SPECIAL_NODES[start] if start in SPECIAL_NODES else [start]
    goal_nodes = SPECIAL_NODES[goal] if goal in SPECIAL_NODES else [goal]

    print(f"Mencari rute cepat dengan Greedy BFS...")
    if wheelchair_mode:
        print("-> Mode: Kursi roda (hanya jalur accessible)")

    best_path: Optional[List[PathNode]] = None
    best_actual_dist = float('inf')

    for s_node in start_nodes:
        for g_node in goal_nodes:
            path, _, actual_dist = multimodal_greedy_bfs(graph_map, s_node, g_node, wheelchair_mode)
            if path and actual_dist is not None and actual_dist < best_actual_dist:
                full_path = []
                if start in SPECIAL_NODES and s_node != start:
                    full_path.append(PathNode(start, None, 0, 0))
                full_path.extend(path)
                if goal in SPECIAL_NODES and g_node != goal:
                    last_mode = path[-1].mode if path else None
                    full_path.append(PathNode(goal, last_mode, 0, actual_dist + SPECIAL_ATTACH_DIST_KM))
                best_path = full_path
                best_actual_dist = actual_dist

    if not best_path:
        print("Rute tidak ditemukan!")
        return
        
    print_result(start, goal, best_path, best_actual_dist, 0.0, wheelchair_mode)

def greedy_best_first_search(
    graph_map: Dict[str, Dict[str, EdgeInfo]],
    start: str,
    goal: str,
    wheelchair_mode: bool = True
) -> Tuple[Optional[List[PathNode]], Optional[float], Optional[float]]:
    """
    Greedy Best-First Search (heuristic-only priority) for multi-modal graph.
    Returns (path, total_cost, actual_distance) similar to multimodal_a_star.
    Note: uses Euclidean heuristic from COORDS to prioritize expansion.
    """
    if start not in graph_map or goal not in graph_map:
        return None, None, None

    heur = compute_euclid_heuristic(COORDS, goal) if COORDS else {}
    counter = 0

    start_node = PathNode(node=start, mode=None, cumulative_cost=0.0, actual_distance=0.0)
    pq: List[Tuple[float, int, float, float, List[PathNode], Optional[TransportMode], int]] = [
        (heur.get(start, 0.0), counter, 0.0, 0.0, [start_node], None, 0)
    ]
    counter += 1

    visited_costs: Dict[Tuple[str, Optional[TransportMode], int], float] = {}
    expanded: Set[Tuple[str, Optional[TransportMode], int]] = set()

    visited_costs[(start, None, 0)] = 0.0

    MAX_EXPANSIONS = 20000
    expansions = 0

    while pq:
        h_priority, _, g_cost, actual_dist, path, current_mode, consec_sidewalk = heapq.heappop(pq)
        current_node = path[-1].node

        state = (current_node, current_mode, consec_sidewalk)

        if state in expanded:
            continue

        expansions += 1
        if expansions > MAX_EXPANSIONS:
            return None, None, None

        expanded.add(state)

        if state in visited_costs and g_cost > visited_costs[state]:
            continue

        if current_node == goal:
            return path, g_cost, actual_dist

        for neighbor, edge_info in graph_map.get(current_node, {}).items():
            if neighbor in SPECIAL_NODES and current_node in SPECIAL_NODES.get(neighbor, []):
                continue

            for mode in edge_info.modes:
                if wheelchair_mode and not edge_info.wheelchair_accessible:
                    continue

                edge_cost = edge_info.distance * PENALTIES[mode]

                new_consec = 0
                if mode == TransportMode.SIDEWALK:
                    new_consec = consec_sidewalk + 1 if current_mode == TransportMode.SIDEWALK else 1

                consecutive_penalty = 0.0
                if new_consec > 1:
                    consecutive_penalty = CONSECUTIVE_SIDEWALK_PENALTY * (new_consec - 1)

                transit_cost = 0.0
                if current_mode is not None and current_mode != mode:
                    transit_cost = TRANSIT_PENALTY

                new_g = g_cost + edge_cost + transit_cost + consecutive_penalty
                new_actual = actual_dist + edge_info.distance

                new_state = (neighbor, mode, new_consec)

                if new_state in expanded:
                    continue

                if new_state not in visited_costs or new_g < visited_costs[new_state]:
                    visited_costs[new_state] = new_g
                    h_val = heur.get(neighbor, 0.0)
                    new_path_node = PathNode(node=neighbor, mode=mode, cumulative_cost=new_g, actual_distance=new_actual)
                    new_path = path + [new_path_node]
                    heapq.heappush(pq, (h_val, counter, new_g, new_actual, new_path, mode, new_consec))
                    counter += 1
                    
        return None,None,None



def main():
    graph_map, _ = build_graph()
    all_nodes = sorted(list(graph_map.keys()))

    print("\n" + "=" * 80)
    print("SISTEM PENCARIAN RUTE SURABAYA - MULTI-MODAL TRANSPORT")
    print("=" * 80)
    
    while True:
        print("\nMENU UTAMA:")
        print("  1. Lihat daftar lokasi")
        print("  2. Lihat koneksi moda transportasi")
        print("  3. Cari rute multi-modal (A* dengan penalty)")
        print("  4. Cari rute cepat (Greedy BFS)")
        print("  5. Keluar")
        
        choice = input("\nPilih menu (1-5): ").strip()
        
        if choice == "1":
            print("\nDAFTAR LOKASI:")
            print("-" * 80)
            
            special_locations = [n for n in all_nodes if n in SPECIAL_NODES]
            grid_nodes = [n for n in all_nodes if n not in SPECIAL_NODES]
            
            print("\nLokasi Khusus:")
            for i, n in enumerate(special_locations, start=1):
                connections = SPECIAL_NODES[n]
                print(f"  {i:2d}. {n} (Akses via: {', '.join(connections)})")
                
            print(f"\nNode Grid ({len(grid_nodes)}):")
            print(", ".join(grid_nodes))
        
        elif choice == "2":
            print("\nKONEKSI MODA TRANSPORTASI:")
            print("-" * 80)
            print("Edge\t\t\tJarak\tModa Tersedia\t\tAccessible")
            print("-" * 80)
            
            edges_shown = set()
            for node_a in sorted(graph_map.keys()):
                for node_b, edge_info in sorted(graph_map[node_a].items()):
                    edge_key = tuple(sorted([node_a, node_b]))
                    if edge_key not in edges_shown:
                        edges_shown.add(edge_key)
                        modes_str = ", ".join([get_mode_icon(m) for m in edge_info.modes])
                        wheelchair = "YA" if edge_info.wheelchair_accessible else "TIDAK"
                        print(f"{node_a:4s} <-> {node_b:4s}\t{edge_info.distance:.3f}km\t{modes_str:15s}\t{wheelchair}")
        
        elif choice == "3":
            handle_route_search(graph_map, all_nodes)
        
        elif choice == "4":
            print("CARI RUTE CEPAT (Greedy BFS)")
            handle_route_search_greedy(graph_map, all_nodes)
        elif choice == "5":
            print("\nTerima kasih!")
            break
                
        else:
            print("\nPilihan tidak valid.")

if __name__ == "__main__":
    main()
