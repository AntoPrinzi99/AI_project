from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

try:
	import z3  # type: ignore
except Exception as e:  # pragma: no cover
	raise SystemExit(
		"Missing dependency 'z3-solver'.\n"
		"You are running: " + sys.executable + "\n"
		"Install into the SAME interpreter with:\n"
		"  " + sys.executable + " -m pip install z3-solver\n"
		f"Import error: {e}"
	)

# Reuse domain utilities from implementation_A.py
from implementation_A import (  
	PuzzleAction,
	PuzzleState,
	_format_board,
	_goal_state_for_size,
	_parse_state,
	_puzzle_result,
	generate_scrambled_instance,
	is_solvable,
)


@dataclass(frozen=True)
class CSPSolveResult:
	found: bool
	actions: List[PuzzleAction]
	elapsed_sec: float
	horizon: int
	solver_calls: int
	horizons_tried: int

	@property
	def cost(self) -> int:
		return len(self.actions)


_ACTIONS: Tuple[PuzzleAction, ...] = ("U", "D", "L", "R")
_ACTION_TO_INT = {a: i for i, a in enumerate(_ACTIONS)}
_INT_TO_ACTION = {i: a for i, a in enumerate(_ACTIONS)}


def _select_cell(row: List[z3.ArithRef], idx: z3.ArithRef) -> z3.ArithRef:
	"""Select row[idx] when idx is a Z3 expression.

	Z3 Int expressions cannot be used as Python list indices, so we encode a
	"selection" as a chain of Ifs.
	"""
	return z3.simplify(z3.Sum([z3.If(idx == i, row[i], z3.IntVal(0)) for i in range(len(row))]))


def _neighbor_index(idx: z3.ArithRef, action: z3.ArithRef, size: int) -> z3.ArithRef:
	"""Return neighbor cell index for (idx, action) assuming action is valid at idx."""
	return z3.If(
		action == _ACTION_TO_INT["U"],
		idx - size,
		z3.If(
			action == _ACTION_TO_INT["D"],
			idx + size,
			z3.If(
				action == _ACTION_TO_INT["L"],
				idx - 1,
				idx + 1,
			),
		),
	)


def solve_puzzle_csp_z3(
	initial: PuzzleState,
	size: int,
	max_steps: int,
	goal: Optional[PuzzleState] = None,
	*,
	progress: bool = False,
	progress_prefix: str = "",
) -> CSPSolveResult:
	"""Solve sliding puzzle by reduction to CSP using Z3.

	We build a time-indexed encoding with horizon T and search for the smallest T
	(from 0..max_steps) such that there exists a sequence of valid blank swaps from
	initial to goal.
	"""
	if goal is None:
		goal = _goal_state_for_size(size)
	if max_steps < 0:
		raise ValueError("max_steps must be >= 0")

	start_time = time.time()

	def progress_print(msg: str) -> None:
		if not progress:
			return
		try:
			print(msg, flush=True)
		except BrokenPipeError:
			# If stdout is closed (e.g. piping to `head`), exit immediately.
			# Using os._exit avoids further writes/flushes that can trigger
			# "Exception ignored ... Broken pipe" during interpreter shutdown.
			os._exit(0)

	cells = size * size
	all_tiles = list(range(cells))

	def solve_with_horizon(T: int) -> Optional[List[PuzzleAction]]:
		solver = z3.Solver()
		solver.set("timeout", 0)  # no timeout; caller can ctrl-c

		# board[t][i] is the tile value at cell i, time t
		board: List[List[z3.IntNumRef | z3.ArithRef]] = []
		for t in range(T + 1):
			row = [z3.Int(f"b_{t}_{i}") for i in range(cells)]
			board.append(row)

		# action[t] is one of {U,D,L,R} encoded as 0..3
		action_vars = [z3.Int(f"a_{t}") for t in range(T)]

		# blank_pos[t] is index 0..cells-1 where tile 0 sits
		blank_pos = [z3.Int(f"z_{t}") for t in range(T + 1)]

		# Domains + all-different per time
		for t in range(T + 1):
			for i in range(cells):
				solver.add(board[t][i] >= 0, board[t][i] < cells)
			solver.add(z3.Distinct(*board[t]))

			# blank position domain
			solver.add(blank_pos[t] >= 0, blank_pos[t] < cells)

			# link blank_pos[t] to board[t]
			solver.add(
				z3.Or(*[z3.And(blank_pos[t] == i, board[t][i] == 0) for i in range(cells)])
			)

		# Initial state
		for i, v in enumerate(initial):
			solver.add(board[0][i] == v)

		# Goal state at time T
		for i, v in enumerate(goal):
			solver.add(board[T][i] == v)

		# Transition constraints
		for t in range(T):
			# action domain
			solver.add(action_vars[t] >= 0, action_vars[t] <= 3)

			# preconditions: action must be valid for the blank position
			z = blank_pos[t]
			r = z / size
			c = z % size
			solver.add(
				z3.Or(
					z3.And(action_vars[t] == _ACTION_TO_INT["U"], r > 0),
					z3.And(action_vars[t] == _ACTION_TO_INT["D"], r < size - 1),
					z3.And(action_vars[t] == _ACTION_TO_INT["L"], c > 0),
					z3.And(action_vars[t] == _ACTION_TO_INT["R"], c < size - 1),
				)
			)

			nb = _neighbor_index(z, action_vars[t], size)

			# blank position update
			solver.add(blank_pos[t + 1] == nb)

			# board update: swap 0 at z with tile at nb
			nb_tile = _select_cell(board[t], nb)
			for i in range(cells):
				solver.add(
					board[t + 1][i]
					== z3.If(
						i == z,
						nb_tile,
						z3.If(i == nb, z3.IntVal(0), board[t][i]),
					)
				)

		if solver.check() != z3.sat:
			return None
		model = solver.model()

		acts: List[PuzzleAction] = []
		for t in range(T):
			av = model.evaluate(action_vars[t], model_completion=True)
			acts.append(_INT_TO_ACTION[av.as_long()])
		return acts

	# Iterative deepening on horizon
	solver_calls = 0
	for T in range(max_steps + 1):
		progress_print(f"{progress_prefix}Trying horizon T={T}/{max_steps}...")
		solver_calls += 1
		acts = solve_with_horizon(T)
		if acts is None:
			progress_print(f"{progress_prefix}  -> UNSAT")
		if acts is not None:
			progress_print(f"{progress_prefix}  -> SAT")
			elapsed = time.time() - start_time
			return CSPSolveResult(
				found=True,
				actions=acts,
				elapsed_sec=elapsed,
				horizon=T,
				solver_calls=solver_calls,
				horizons_tried=T + 1,
			)

	elapsed = time.time() - start_time
	return CSPSolveResult(
		found=False,
		actions=[],
		elapsed_sec=elapsed,
		horizon=max_steps,
		solver_calls=solver_calls,
		horizons_tried=max_steps + 1,
	)


def _apply_actions(initial: PuzzleState, size: int, actions: List[PuzzleAction]) -> PuzzleState:
	state = initial
	for a in actions:
		state = _puzzle_result(state, a, size)
	return state


def main(argv: Optional[Sequence[str]] = None) -> int:
	parser = argparse.ArgumentParser(description="CSP(Z3) solver for generalized sliding puzzle (4x4 / 5x5)")
	parser.add_argument("--size", type=int, default=4, help="Puzzle size N for an N×N board (default: 4)")
	parser.add_argument(
		"--initial",
		nargs="*",
		type=int,
		help="Initial state as N*N integers (row-major), using 0 as blank.",
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
		help="Number of runs (for experiments). If >1 and --random-moves is used, a new scrambled instance is generated each run.",
	)
	parser.add_argument("--seed", type=int, default=None, help="Random seed (used with --random-moves).")
	parser.add_argument(
		"--max-steps",
		type=int,
		default=40,
		help="Max horizon to try (iterative deepening up to this many moves).",
	)
	parser.add_argument(
		"--progress",
		action="store_true",
		help="Print per-horizon progress (useful for long CSP runs).",
	)
	parser.add_argument(
		"--show-boards",
		action="store_true",
		help="Print initial/goal boards.",
	)
	args = parser.parse_args(list(argv) if argv is not None else None)

	size = args.size
	if size not in (4, 5):
		raise ValueError("This script is intended for size 4 or 5")
	if args.max_steps < 0:
		raise ValueError("--max-steps must be >= 0")
	if args.runs < 1:
		raise ValueError("--runs must be >= 1")

	goal = _goal_state_for_size(size)

	# Parse base initial (for single run or when user provides --initial)
	base_initial: Optional[PuzzleState]
	if args.random_moves is not None:
		base_initial = None
	else:
		if args.initial is None or len(args.initial) == 0:
			raise ValueError("Provide --initial or --random-moves")
		base_initial = _parse_state(args.initial, size)

	def build_initial_for_run(run_idx: int) -> PuzzleState:
		# If --random-moves is provided, we generate a scrambled instance.
		# For experiments (runs>1), we vary the seed by run index to get different instances.
		if args.random_moves is not None:
			seed = None if args.seed is None else args.seed + run_idx
			return generate_scrambled_instance(size=size, moves=args.random_moves, seed=seed)
		assert base_initial is not None
		return base_initial

	results: List[CSPSolveResult] = []

	for run_idx in range(args.runs):
		initial = build_initial_for_run(run_idx)
		if not is_solvable(initial, size, goal=goal):
			print(f"[run {run_idx+1}] Unsolvable instance for this goal (parity mismatch).")
			return 2

		if args.show_boards or args.runs == 1:
			print(f"\n=== Run {run_idx+1}/{args.runs} ===")
			print("Initial:")
			print(_format_board(initial, size))
			print("\nGoal:")
			print(_format_board(goal, size))
			print()
		else:
			print(f"\n=== Run {run_idx+1}/{args.runs} ===")

		result = solve_puzzle_csp_z3(
			initial=initial,
			size=size,
			max_steps=args.max_steps,
			goal=goal,
			progress=args.progress,
			progress_prefix=f"[run {run_idx+1}/{args.runs}] ",
		)
		results.append(result)

		if not result.found:
			print(f"No solution found up to max_steps={args.max_steps} (elapsed_sec={result.elapsed_sec:.3f}).")
			continue

		moves = "".join(result.actions)
		print(f"Solution found (CSP/Z3): cost={result.cost} moves={moves}")
		print(
			" ".join(
				[
					f"horizon={result.horizon}",
					f"solver_calls={result.solver_calls}",
					f"horizons_tried={result.horizons_tried}",
					f"elapsed_sec={result.elapsed_sec:.6f}",
				]
			)
		)

		# Verify reconstruction by simulation
		final_state = _apply_actions(initial, size, result.actions)
		if final_state != goal:
			print("WARNING: reconstructed plan does not reach goal (bug in extraction/encoding).")
			return 3

	if args.runs == 1:
		return 0 if results and results[0].found else 1

	# Summary (mean over runs)
	solved = [r for r in results if r.found]
	print("\n=== Summary (mean over runs) ===")
	print(f"runs={len(results)} solved={len(solved)}")
	if solved:
		cost_mean = sum(r.cost for r in solved) / len(solved)
		horizon_mean = sum(r.horizon for r in solved) / len(solved)
		elapsed_mean = sum(r.elapsed_sec for r in solved) / len(solved)
		solver_calls_mean = sum(r.solver_calls for r in solved) / len(solved)
		horizons_tried_mean = sum(r.horizons_tried for r in solved) / len(solved)
		print(
			" ".join(
				[
					f"solution_cost_mean={cost_mean:.3f}",
					f"horizon_mean={horizon_mean:.3f}",
					f"solver_calls_mean={solver_calls_mean:.3f}",
					f"horizons_tried_mean={horizons_tried_mean:.3f}",
					f"elapsed_sec_mean={elapsed_mean:.6f}",
				]
			)
		)
	else:
		elapsed_mean = sum(r.elapsed_sec for r in results) / len(results)
		print(f"elapsed_sec_mean={elapsed_mean:.6f}")

	return 0


if __name__ == "__main__":
	raise SystemExit(main())
