
from __future__ import annotations

import argparse
import heapq
import math
import random
import time
from dataclasses import dataclass
from typing import Callable, Dict, Generic, Hashable, Iterable, List, Optional, Sequence, Tuple, TypeVar


StateT = TypeVar("StateT", bound=Hashable)
ActionT = TypeVar("ActionT")


@dataclass(frozen=True)
class Node(Generic[StateT, ActionT]):
	state: StateT
	parent: Optional["Node[StateT, ActionT]"] = None
	action: Optional[ActionT] = None
	g: int = 0
	h: int = 0

	@property
	def f(self) -> int:
		return self.g + self.h

	def iter_actions(self) -> Iterable[ActionT]:
		node: Optional[Node[StateT, ActionT]] = self
		rev: List[ActionT] = []
		while node is not None and node.action is not None:
			rev.append(node.action)
			node = node.parent
		return reversed(rev)


@dataclass(frozen=True)
class AStarResult(Generic[StateT, ActionT]):
	solution_node: Optional[Node[StateT, ActionT]]
	expanded: int
	generated: int
	max_frontier: int
	max_explored: int
	frontier_final: int
	explored_final: int
	elapsed_sec: float

	@property
	def found(self) -> bool:
		return self.solution_node is not None

	def solution_actions(self) -> List[ActionT]:
		if self.solution_node is None:
			return []
		return list(self.solution_node.iter_actions())

	@property
	def solution_cost(self) -> Optional[int]:
		if self.solution_node is None:
			return None
		return self.solution_node.g

	@property
	def branching_factor_proxy(self) -> Optional[float]:
		# Not the true branching factor in a graph with duplicate elimination;
		# useful as a cheap proxy.
		if self.expanded == 0:
			return None
		return self.generated / self.expanded


class Problem(Generic[StateT, ActionT]):
	def __init__(
		self,
		initial_state: StateT,
		goal_test: Callable[[StateT], bool],
		actions: Callable[[StateT], Iterable[ActionT]],
		result: Callable[[StateT, ActionT], StateT],
		step_cost: Callable[[StateT, ActionT, StateT], int],
		heuristic: Callable[[StateT], int],
	) -> None:
		self.initial_state = initial_state
		self.goal_test = goal_test
		self.actions = actions
		self.result = result
		self.step_cost = step_cost
		self.heuristic = heuristic


def astar_duplicate_elimination_no_reopening(
	problem: Problem[StateT, ActionT],
) -> AStarResult[StateT, ActionT]:
	"""A* as in 'Chapter 4 - Classical Search - Part II' slide 32:

	- frontier ordered by ascending f=g+h
	- explored is a set of states
	- duplicates handled as in uniform-cost search:
	  if a better g is found for a state already in frontier -> replace
	- no reopening: if a state is in explored, we never put it back in frontier
	"""

	start_time = time.time()

	start = Node(state=problem.initial_state, parent=None, action=None, g=0, h=problem.heuristic(problem.initial_state))

	# heap items: (f, tie, state)
	frontier_heap: List[Tuple[int, int, StateT]] = []
	tie_counter = 0
	heapq.heappush(frontier_heap, (start.f, tie_counter, start.state))

	# best node currently stored for each state in frontier
	frontier_nodes: Dict[StateT, Node[StateT, ActionT]] = {start.state: start}

	explored: set[StateT] = set()

	expanded = 0
	generated = 1
	max_frontier = 1
	max_explored = 0

	while True:
		if not frontier_heap:
			return AStarResult(
				solution_node=None,
				expanded=expanded,
				generated=generated,
				max_frontier=max_frontier,
				max_explored=max_explored,
				frontier_final=len(frontier_nodes),
				explored_final=len(explored),
				elapsed_sec=time.time() - start_time,
			)

		# Pop; skip stale heap entries (because we do 'replace' by updating dict and pushing again)
		while frontier_heap:
			_, _, state = heapq.heappop(frontier_heap)
			node = frontier_nodes.get(state)
			if node is not None:
				break
		else:
			# heap exhausted due to stale entries
			return AStarResult(
				solution_node=None,
				expanded=expanded,
				generated=generated,
				max_frontier=max_frontier,
				max_explored=max_explored,
				frontier_final=len(frontier_nodes),
				explored_final=len(explored),
				elapsed_sec=time.time() - start_time,
			)

		# Remove from frontier
		del frontier_nodes[node.state]

		if problem.goal_test(node.state):
			return AStarResult(
				solution_node=node,
				expanded=expanded,
				generated=generated,
				max_frontier=max_frontier,
				max_explored=max_explored,
				frontier_final=len(frontier_nodes),
				explored_final=len(explored),
				elapsed_sec=time.time() - start_time,
			)

		explored.add(node.state)
		max_explored = max(max_explored, len(explored))

		expanded += 1
		for action in problem.actions(node.state):
			child_state = problem.result(node.state, action)
			if child_state in explored:
				# no reopening
				continue

			step = problem.step_cost(node.state, action, child_state)
			child_g = node.g + step
			child_h = problem.heuristic(child_state)
			child = Node(state=child_state, parent=node, action=action, g=child_g, h=child_h)
			generated += 1

			existing = frontier_nodes.get(child_state)
			if existing is None:
				tie_counter += 1
				frontier_nodes[child_state] = child
				heapq.heappush(frontier_heap, (child.f, tie_counter, child_state))
			else:
				# duplicate elimination in frontier: replace if strictly better g
				if child.g < existing.g:
					tie_counter += 1
					frontier_nodes[child_state] = child
					heapq.heappush(frontier_heap, (child.f, tie_counter, child_state))

		max_frontier = max(max_frontier, len(frontier_nodes))


PuzzleState = Tuple[int, ...]
PuzzleAction = str  # 'U', 'D', 'L', 'R'


def _goal_state_for_size(size: int) -> PuzzleState:
	# Goal: 1..N^2-1, 0 as blank at end
	return tuple(list(range(1, size * size)) + [0])


def _format_board(state: PuzzleState, size: int) -> str:
	width = len(str(size * size - 1))
	rows = []
	for r in range(size):
		row = state[r * size : (r + 1) * size]
		rows.append(" ".join(".".rjust(width) if v == 0 else str(v).rjust(width) for v in row))
	return "\n".join(rows)


def _parse_state(flat: Sequence[int], size: int) -> PuzzleState:
	if len(flat) != size * size:
		raise ValueError(f"Expected {size*size} integers for a {size}x{size} puzzle, got {len(flat)}")
	if sorted(flat) != list(range(size * size)):
		raise ValueError(f"State must be a permutation of 0..{size*size-1}")
	return tuple(flat)


def _puzzle_actions(state: PuzzleState, size: int) -> Iterable[PuzzleAction]:
	z = state.index(0)
	r, c = divmod(z, size)
	if r > 0:
		yield "U"
	if r < size - 1:
		yield "D"
	if c > 0:
		yield "L"
	if c < size - 1:
		yield "R"


def _puzzle_result(state: PuzzleState, action: PuzzleAction, size: int) -> PuzzleState:
	z = state.index(0)
	r, c = divmod(z, size)
	if action == "U":
		nr, nc = r - 1, c
	elif action == "D":
		nr, nc = r + 1, c
	elif action == "L":
		nr, nc = r, c - 1
	elif action == "R":
		nr, nc = r, c + 1
	else:
		raise ValueError(f"Unknown action: {action}")

	nz = nr * size + nc
	lst = list(state)
	lst[z], lst[nz] = lst[nz], lst[z]
	return tuple(lst)


def _puzzle_step_cost(_: PuzzleState, __: PuzzleAction, ___: PuzzleState) -> int:
	return 1


def _heuristic_zero(_: PuzzleState) -> int:
	return 0


def make_misplaced_tiles(goal: PuzzleState) -> Callable[[PuzzleState], int]:
	def h(state: PuzzleState) -> int:
		# ignore blank (0)
		return sum(1 for i, v in enumerate(state) if v != 0 and v != goal[i])

	return h


def make_manhattan(goal: PuzzleState, size: int) -> Callable[[PuzzleState], int]:
	goal_pos: Dict[int, Tuple[int, int]] = {}
	for idx, tile in enumerate(goal):
		goal_pos[tile] = divmod(idx, size)

	def h(state: PuzzleState) -> int:
		dist = 0
		for idx, tile in enumerate(state):
			if tile == 0:
				continue
			r, c = divmod(idx, size)
			gr, gc = goal_pos[tile]
			dist += abs(r - gr) + abs(c - gc)
		return dist

	return h


def is_solvable(state: PuzzleState, size: int, goal: Optional[PuzzleState] = None) -> bool:
	"""Standard sliding puzzle solvability check.

	Assumes goal is the canonical goal unless provided.
	"""
	if goal is None:
		goal = _goal_state_for_size(size)
	# Map tiles to goal order so parity is relative to the chosen goal.
	# For even-sized boards, the parity condition depends on both inversions and
	# the blank row (counted from bottom). To support arbitrary goals, compare the
	# parity "signature" of state vs goal.
	goal_index = {tile: i for i, tile in enumerate(goal) if tile != 0}

	def inversion_count_in_goal_order(s: PuzzleState) -> int:
		seq = [goal_index[t] for t in s if t != 0]
		inv = 0
		for i in range(len(seq)):
			for j in range(i + 1, len(seq)):
				if seq[i] > seq[j]:
					inv += 1
		return inv

	inv_state = inversion_count_in_goal_order(state)
	if size % 2 == 1:
		# odd grid: parity must match the goal (goal parity is 0 in this ordering)
		return inv_state % 2 == 0

	def blank_row_from_bottom(s: PuzzleState) -> int:
		blank_idx = s.index(0)
		return size - (blank_idx // size)

	inv_goal = inversion_count_in_goal_order(goal)
	sig_state = (inv_state + blank_row_from_bottom(state)) % 2
	sig_goal = (inv_goal + blank_row_from_bottom(goal)) % 2
	return sig_state == sig_goal


def make_puzzle_problem(
	initial: PuzzleState,
	size: int,
	heuristic_name: str,
	goal: Optional[PuzzleState] = None,
) -> Problem[PuzzleState, PuzzleAction]:
	if goal is None:
		goal = _goal_state_for_size(size)

	if heuristic_name == "zero":
		heuristic = _heuristic_zero
	elif heuristic_name == "misplaced":
		heuristic = make_misplaced_tiles(goal)
	elif heuristic_name == "manhattan":
		heuristic = make_manhattan(goal, size)
	else:
		raise ValueError("Heuristic must be one of: zero, misplaced, manhattan")

	return Problem(
		initial_state=initial,
		goal_test=lambda s: s == goal,
		actions=lambda s: _puzzle_actions(s, size),
		result=lambda s, a: _puzzle_result(s, a, size),
		step_cost=_puzzle_step_cost,
		heuristic=heuristic,
	)


def _default_demo_initial(size: int) -> PuzzleState:
	# A small 3x3 demo instance (few moves), otherwise a trivial near-goal for larger sizes
	if size == 3:
		return _parse_state([1, 2, 3, 4, 5, 6, 0, 7, 8], 3)
	goal = _goal_state_for_size(size)
	# swap blank with left neighbor (one move away)
	lst = list(goal)
	lst[-1], lst[-2] = lst[-2], lst[-1]
	return tuple(lst)

# I use this one in experiments
def generate_scrambled_instance(size: int, moves: int, seed: Optional[int] = None) -> PuzzleState:
	"""Generate a random *solvable* instance by applying random moves from the goal.

	This guarantees solvability (reachable from the goal) and is usually preferable to
	generating a random permutation and retrying until solvable.
	"""
	if moves < 0:
		raise ValueError("moves must be >= 0")
	goal = _goal_state_for_size(size)
	state = goal
	rng = random.Random(seed)
	prev_action: Optional[PuzzleAction] = None
	inverse = {"U": "D", "D": "U", "L": "R", "R": "L"}
	for _ in range(moves):
		acts = list(_puzzle_actions(state, size))
		# avoid immediately undoing the previous move if possible
		if prev_action is not None:
			inv = inverse[prev_action]
			if len(acts) > 1 and inv in acts:
				acts.remove(inv)
		action = rng.choice(acts)
		state = _puzzle_result(state, action, size)
		prev_action = action
	return state


def generate_random_permutation_instance(size: int, seed: Optional[int] = None, max_tries: int = 10_000) -> PuzzleState:
	"""Generate a random solvable instance by sampling permutations and checking solvability."""
	if max_tries <= 0:
		raise ValueError("max_tries must be > 0")
	rng = random.Random(seed)
	n = size * size
	arr = list(range(n))
	goal = _goal_state_for_size(size)
	for _ in range(max_tries):
		rng.shuffle(arr)
		state = tuple(arr)
		if is_solvable(state, size, goal=goal):
			return state
	raise RuntimeError(f"Failed to sample a solvable instance within max_tries={max_tries}")


def main(argv: Optional[Sequence[str]] = None) -> int:
	parser = argparse.ArgumentParser(description="A* (duplicate elimination, no reopening) on generalized sliding puzzle")
	parser.add_argument("--size", type=int, default=4, help="Puzzle size N for an N×N board (default: 4)")
	parser.add_argument(
		"--heuristic",
		choices=["zero", "misplaced", "manhattan"],
		default="manhattan",
		help="Heuristic to use (default: manhattan)",
	)
	parser.add_argument(
		"--initial",
		nargs="*",
		type=int,
		help="Initial state as N*N integers (row-major), using 0 as blank. If omitted, uses a small demo.",
	)
	parser.add_argument(
		"--random-moves",
		type=int,
		default=None,
		help="Generate a random solvable instance by scrambling the goal with this many random moves.",
	)
	parser.add_argument(
		"--runs",
		type=int,
		default=1,
		help="Number of runs (for experiments). If >1, a new random instance is generated each run.",
	)
	parser.add_argument(
		"--seed",
		type=int,
		default=None,
		help="Random seed (used with --random-moves).",
	)
	parser.add_argument(
		"--show-boards",
		action="store_true",
		help="Print initial/goal boards for each run (useful for debugging; noisy for experiments).",
	)
	parser.add_argument(
		"--no-solvability-check",
		action="store_true",
		help="Skip solvability check (not recommended).",
	)
	args = parser.parse_args(list(argv) if argv is not None else None)

	size = args.size
	if size < 2:
		raise ValueError("size must be >= 2")
	if args.runs < 1:
		raise ValueError("--runs must be >= 1")

	goal = _goal_state_for_size(size)

	def build_initial_for_run(run_idx: int) -> PuzzleState:
		# For experiments (runs>1) we always regenerate a new random instance.
		# If --random-moves is provided, we scramble from the goal (recommended).
		# Otherwise, we sample random permutations and filter by solvability.
		if args.runs > 1:
			seed = None if args.seed is None else args.seed + run_idx
			if args.random_moves is not None:
				return generate_scrambled_instance(size=size, moves=args.random_moves, seed=seed)
			return generate_random_permutation_instance(size=size, seed=seed)

		# Single run: respect user intent/precedence
		if args.random_moves is not None:
			return generate_scrambled_instance(size=size, moves=args.random_moves, seed=args.seed)
		if args.initial is None or len(args.initial) == 0:
			return _default_demo_initial(size)
		return _parse_state(args.initial, size)

	results: List[AStarResult[PuzzleState, PuzzleAction]] = []

	for run_idx in range(args.runs):
		initial = build_initial_for_run(run_idx)
		if not args.no_solvability_check and not is_solvable(initial, size, goal=goal):
			print(f"[run {run_idx+1}] Unsolvable instance for this goal (parity mismatch).")
			return 2

		problem = make_puzzle_problem(initial=initial, size=size, heuristic_name=args.heuristic, goal=goal)
		result = astar_duplicate_elimination_no_reopening(problem)
		results.append(result)

		print(f"\n=== Run {run_idx+1}/{args.runs} ===")
		if args.show_boards or args.runs == 1:
			print("Initial:")
			print(_format_board(initial, size))
			print("\nGoal:")
			print(_format_board(goal, size))
			print()

		if not result.found:
			print("No solution found.")
		else:
			moves = "".join(result.solution_actions())
			print(f"Solution found: cost={result.solution_cost} moves={moves}")

		bf = result.branching_factor_proxy
		bf_str = "n/a" if bf is None else f"{bf:.3f}"
		print(
			" ".join(
				[
					f"expanded={result.expanded}",
					f"generated={result.generated}",
					f"branching_proxy={bf_str}",
					f"max_frontier={result.max_frontier}",
					f"max_explored={result.max_explored}",
					f"frontier_final={result.frontier_final}",
					f"explored_final={result.explored_final}",
					f"elapsed_sec={result.elapsed_sec:.6f}",
				]
			)
		)

	if args.runs == 1:
		return 0 if results[0].found else 1

	# Summary (mean over runs)
	def mean_int(values: List[int]) -> float:
		return sum(values) / len(values) if values else math.nan

	solved = [r for r in results if r.found]
	print("\n=== Summary (mean over runs) ===")
	print(f"runs={len(results)} solved={len(solved)}")
	print(
		" ".join(
			[
				f"expanded_mean={mean_int([r.expanded for r in results]):.3f}",
				f"generated_mean={mean_int([r.generated for r in results]):.3f}",
				f"max_frontier_mean={mean_int([r.max_frontier for r in results]):.3f}",
				f"max_explored_mean={mean_int([r.max_explored for r in results]):.3f}",
				f"frontier_final_mean={mean_int([r.frontier_final for r in results]):.3f}",
				f"explored_final_mean={mean_int([r.explored_final for r in results]):.3f}",
				f"elapsed_sec_mean={(sum(r.elapsed_sec for r in results) / len(results)):.6f}",
			]
		)
	)

	if solved:
		print(f"solution_cost_mean={(sum(r.solution_cost or 0 for r in solved) / len(solved)):.3f}")

	return 0


if __name__ == "__main__":
	raise SystemExit(main())

