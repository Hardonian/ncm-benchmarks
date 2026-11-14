"""Advanced decoding verification, grammar masking, symbolic constraint solving & adaptive entropy exit."""
from __future__ import annotations

import math
from typing import Any

import torch

from ncm_bench.inference import repair_sudoku_board
from ncm_bench.metrics import is_valid_sudoku_board


def speculative_coarse_to_fine_verify(
    draft_codes: torch.Tensor,
    target_logits_per_level: list[torch.Tensor],
    *,
    temperature: float = 1.0,
    seed: int = 42,
) -> dict[str, Any]:
    """Verifies draft hierarchical concept codes level-by-level against target logits."""
    torch.manual_seed(seed)
    verified = draft_codes.clone()
    total_levels = len(target_logits_per_level)
    accepted_levels = 0
    offset = 0

    for logits_lvl in target_logits_per_level:
        width = logits_lvl.shape[1]
        lvl_slice = verified[:, offset : offset + width]
        probs = torch.softmax(logits_lvl.float() / max(temperature, 1e-4), dim=-1)
        draft_probs = probs.gather(-1, lvl_slice.unsqueeze(-1)).squeeze(-1)
        top_probs = probs.max(dim=-1).values.clamp_min(1e-6)
        accept_ratio = (draft_probs / top_probs).clamp(max=1.0)
        rand_u = torch.rand_like(accept_ratio)

        if bool((rand_u <= accept_ratio).all().item()):
            accepted_levels += 1
        else:
            verified[:, offset : offset + width] = probs.argmax(dim=-1)
            offset += width
            break
        offset += width

    # Refill remaining fine levels greedily from target logits if draft diverged early
    for idx in range(accepted_levels + (0 if accepted_levels == total_levels else 1), total_levels):
        logits_lvl = target_logits_per_level[idx]
        width = logits_lvl.shape[1]
        if offset + width <= verified.shape[1]:
            verified[:, offset : offset + width] = logits_lvl.argmax(dim=-1)
        offset += width

    return {
        "verified_codes": verified,
        "accepted_levels": accepted_levels,
        "total_levels": total_levels,
        "acceptance_rate": round(accepted_levels / max(1, total_levels), 4),
    }


def apply_json_grammar_mask(
    logits: torch.Tensor,
    allowed_token_ids: list[int] | set[int],
) -> torch.Tensor:
    """Masks disallowed vocabulary logits to -1e9 for strict structured/JSON decoding."""
    vocab_size = logits.shape[-1]
    masked = torch.full_like(logits, -1e9)
    valid_ids = [idx for idx in allowed_token_ids if 0 <= idx < vocab_size]
    if not valid_ids:
        return logits.clone()
    idx_tensor = torch.tensor(valid_ids, device=logits.device, dtype=torch.long)
    masked[..., idx_tensor] = logits[..., idx_tensor]
    return masked


def _solve_sat_dpll(
    clauses: list[tuple[int, ...]],
    assignment: dict[int, bool],
) -> dict[int, bool] | None:
    """Pure-Python DPLL boolean satisfiability solver with unit propagation."""
    assignment = dict(assignment)
    changed = True
    while changed:
        changed = False
        unresolved: list[list[int]] = []
        for clause in clauses:
            satisfied = False
            open_lits: list[int] = []
            for lit in clause:
                var = abs(lit)
                val = assignment.get(var)
                lit_true = (lit > 0) if val is None else (val if lit > 0 else not val)
                if val is not None and lit_true:
                    satisfied = True
                    break
                if val is None:
                    open_lits.append(lit)
            if not satisfied:
                if not open_lits:
                    return None
                if len(open_lits) == 1:
                    unit = open_lits[0]
                    assignment[abs(unit)] = unit > 0
                    changed = True
                else:
                    unresolved.append(open_lits)
        if not changed:
            if not unresolved:
                return assignment
            branch_var = abs(unresolved[0][0])
            for candidate_val in (True, False):
                attempt = dict(assignment)
                attempt[branch_var] = candidate_val
                solved = _solve_sat_dpll(clauses, attempt)
                if solved is not None:
                    return solved
            return None
    return assignment


def solve_symbolic_constraints(
    problem: str | list[int] | list[tuple[int, ...]],
    *,
    task: str = "sudoku",
) -> dict[str, Any]:
    """Symbolic verification and repair engine for Sudoku grids and CNF SAT clauses."""
    if task == "sudoku":
        raw_board = (
            problem.strip()
            if isinstance(problem, str)
            else "".join(str(int(v)) for v in problem)  # type: ignore[arg-type]
        )
        repaired = repair_sudoku_board(raw_board, "0" * len(raw_board))
        return {
            "task": "sudoku",
            "satisfiable": is_valid_sudoku_board(repaired),
            "solution": repaired,
        }
    if task == "sat":
        clauses = [tuple(int(x) for x in cl) for cl in problem]  # type: ignore[union-attr]
        sol = _solve_sat_dpll(clauses, {})
        return {
            "task": "sat",
            "satisfiable": sol is not None,
            "assignment": sol or {},
        }
    raise ValueError(f"Unsupported symbolic task: {task}")


def adaptive_entropy_early_exit(
    level_logits: list[torch.Tensor],
    *,
    entropy_threshold: float = 0.55,
    min_levels: int = 2,
) -> dict[str, Any]:
    """Stops coarse-to-fine decoding dynamically when normalized Shannon entropy falls below threshold."""
    if not level_logits:
        raise ValueError("level_logits must not be empty")

    entropies: list[float] = []
    selected_codes: list[torch.Tensor] = []
    exit_level = len(level_logits) - 1

    for idx, logits_lvl in enumerate(level_logits):
        probs = torch.softmax(logits_lvl.float(), dim=-1).clamp_min(1e-9)
        vocab = probs.shape[-1]
        norm_denom = math.log(max(2, vocab))
        entropy = float((-(probs * probs.log()).sum(dim=-1) / norm_denom).mean().item())
        entropies.append(round(entropy, 4))
        selected_codes.append(logits_lvl.argmax(dim=-1))
        if (idx + 1) >= min_levels and entropy <= entropy_threshold:
            exit_level = idx
            break

    return {
        "exit_level": exit_level,
        "passes_executed": exit_level + 1,
        "total_levels": len(level_logits),
        "passes_saved": len(level_logits) - (exit_level + 1),
        "entropies": entropies,
        "selected_codes": selected_codes,
    }
