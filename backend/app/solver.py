"""Exact three-mask assignment with stitch minimization.

Every layout fragment is assigned to one of three masks so that each
conflict edge is bichromatic, while the total weight of stitch edges whose
endpoints land on different masks ("cut" stitches) is minimized.

All reported colorings are canonical: scanning fragments in ascending id
order, the first mask encountered is 0, the next new mask is 1, then 2.
Canonical form quotients out the six mask permutations, which makes
uniqueness of the optimum well defined.

Two exact engines share the work:

* The CBC MILP is used whenever every stitch weight survives the pipeline
  exactly — PuLP serializes the MPS with thirteen significant digits and
  CBC computes in IEEE-754 doubles, so weights up to ``10**9`` (with the
  total below ``2**53``) keep exact integer semantics in the objective, the
  optimal face and the uniqueness probe.  If CBC still fails to certify a
  result, the solve falls back to the second engine.
* Heavier weights are handled by an exact search that never leaves Python's
  arbitrary-precision integers.  It combines arc-consistency propagation
  (singleton domains prune every neighbor, cascading), MRV dynamic variable
  ordering and an exact per-edge stitch lower bound; cost-zero instances are
  solved as equality/inequality constraint systems (zero-cost stitches
  merge their endpoints).  Any positive integer weight keeps its exact
  order in the objective, the optimal face, the canonical primary solution
  and the uniqueness determination.
"""

from __future__ import annotations

import time

import pulp

MASKS = (0, 1, 2)
TIME_LIMIT_SECONDS = 30

# Largest weight the MILP engine handles exactly: PuLP serializes the MPS
# with thirteen significant digits and CBC computes in IEEE-754 doubles, so
# on this domain every coefficient and every reachable objective value is an
# exactly represented integer, far inside CBC's tolerances.
# Heavier weights go to the exact integer search below.
_MILP_MAX_SAFE_WEIGHT = 10**9
_MILP_MAX_SAFE_TOTAL = 2**53 - 1

# Resource budget for the exact integer search.  Exhausting it is reported
# like a solver timeout instead of returning an uncertified answer.
_EXACT_NODE_BUDGET = 20_000_000
_EXACT_TIME_LIMIT_SECONDS = 25.0


class SolverError(Exception):
    """The solver could not certify an optimal solution."""


class _SearchExhausted(Exception):
    """The exact search exceeded its node or time budget."""


def _build_problem(order, conflict_edges, stitch_edges):
    """Build the canonical-form MILP for one instance."""
    n = len(order)
    pos = {v: i for i, v in enumerate(order)}
    prob = pulp.LpProblem("mask_assignment", pulp.LpMinimize)
    x = pulp.LpVariable.dicts("x", (range(n), MASKS), cat=pulp.LpBinary)
    y = pulp.LpVariable.dicts("y", range(len(stitch_edges)), lowBound=0, upBound=1)

    for i in range(n):
        prob += pulp.lpSum(x[i][k] for k in MASKS) == 1, f"assign_{i}"

    # Canonical first-occurrence order: fragment i may take mask k > 0 only
    # if some earlier fragment (ascending id) already took mask k - 1.
    for i in range(n):
        for k in (1, 2):
            prob += (
                x[i][k] <= pulp.lpSum(x[j][k - 1] for j in range(i)),
                f"canon_{i}_{k}",
            )

    for a, b in conflict_edges:
        ia, ib = pos[a], pos[b]
        for k in MASKS:
            prob += x[ia][k] + x[ib][k] <= 1, f"conflict_{ia}_{ib}_{k}"

    for ei, (a, b, _w) in enumerate(stitch_edges):
        ia, ib = pos[a], pos[b]
        for k in MASKS:
            prob += y[ei] >= x[ia][k] - x[ib][k], f"cut_{ei}_{k}"

    objective = pulp.lpSum(w * y[ei] for ei, (_a, _b, w) in enumerate(stitch_edges))
    prob += objective
    return prob, x, objective


def _cbc():
    return pulp.PULP_CBC_CMD(msg=False, timeLimit=TIME_LIMIT_SECONDS)


def _status(prob):
    return pulp.LpStatus[prob.status]


def _solve_proven(prob):
    """Solve ``prob`` and return the certificate: ``"optimal"`` when CBC
    proved optimality, ``"infeasible"`` when it proved infeasibility, or
    ``None`` otherwise.

    PuLP maps CBC's "Stopped on time - objective ..." to ``LpStatusOptimal``
    when an incumbent exists, so ``LpStatus`` alone cannot distinguish a
    proven optimum from an unproven incumbent; the solution status keeps
    them apart.
    """
    prob.solve(_cbc())
    if _status(prob) == "Infeasible":
        return "infeasible"
    if (
        _status(prob) == "Optimal"
        and prob.sol_status == pulp.constants.LpSolutionOptimal
    ):
        return "optimal"
    return None


def _color_of(x, i):
    return max(MASKS, key=lambda k: pulp.value(x[i][k]) or 0.0)


def _cut_stitches(stitch_edges, pos, colors):
    return [
        {"pair": [a, b], "weight": w}
        for a, b, w in stitch_edges
        if colors[pos[a]] != colors[pos[b]]
    ]


def _solve_milp(order, conflict_edges, stitches):
    """Solve one validated instance with the CBC MILP engine.

    Callers must guarantee every weight is within ``_MILP_MAX_SAFE_WEIGHT``
    and the weight total within ``_MILP_MAX_SAFE_TOTAL`` so the solve is
    exact.  Returns the same response shape as ``solve_mask_assignment``.
    """
    n = len(order)
    pos = {v: i for i, v in enumerate(order)}

    prob, x, objective = _build_problem(order, conflict_edges, stitches)
    cert = _solve_proven(prob)
    if cert == "infeasible":
        return {"status": "infeasible"}
    if cert != "optimal":
        raise SolverError("求解器未能在限定时间内证明最优解")

    # Read the optimum off the optimal coloring itself: stitch weights are
    # positive integers, so the cut total is an exact integer.
    first_colors = [_color_of(x, i) for i in range(n)]
    best = sum(w for a, b, w in stitches if first_colors[pos[a]] != first_colors[pos[b]])
    # Pin the optimal face.  Objectives are integers, so ``<= best`` selects
    # exactly the optimal face; the integer bound serializes exactly (unlike
    # a fractional slack, which CBC's tolerances and the MPS rounding can
    # corrupt — a tiny 1e-6 slack is known to break CBC's cut generation).
    prob += objective <= best, "optimal_bound"

    # Lexicographically smallest canonical optimum: minimize the mask at
    # each position in turn (ids ascending), then pin it before moving on.
    colors = [0] * n
    for i in range(n):
        prob.setObjective(pulp.lpSum(k * x[i][k] for k in MASKS))
        if _solve_proven(prob) != "optimal":
            raise SolverError("求解器在构造字典序最小方案时失败")
        colors[i] = _color_of(x, i)
        prob += x[i][colors[i]] == 1, f"fix_{i}"

    # Uniqueness probe: any canonical optimum different from the one above?
    for i in range(n):
        prob.constraints.pop(f"fix_{i}", None)
    prob += pulp.lpSum(x[i][colors[i]] for i in range(n)) <= n - 1, "exclude_assignment"
    cert = _solve_proven(prob)
    if cert is None:
        raise SolverError("求解器未能在限定时间内判定唯一性")
    unique = cert != "optimal"

    witness = None
    if not unique:
        witness_colors = [_color_of(x, i) for i in range(n)]
        witness = {
            "assignment": {str(order[i]): witness_colors[i] for i in range(n)},
            "cut_stitches": _cut_stitches(stitches, pos, witness_colors),
        }

    return {
        "status": "optimal",
        "objective": best,
        "unique": unique,
        "assignment": {str(order[i]): colors[i] for i in range(n)},
        "cut_stitches": _cut_stitches(stitches, pos, colors),
        "witness": witness,
    }


class _ConstraintGraph:
    """A disequality (conflict) graph with bitmask domains, searched with
    singleton propagation (a domain pinned to one color forbids that color
    on every neighbor, cascading) and MRV dynamic variable ordering.

    This is the workhorse for plain 3-colorability probes as well as for
    cost-zero instances, where zero-cut stitches act as equalities and the
    variables are union-find components rather than fragments.
    """

    def __init__(self, nv, edges, deadline=None):
        self.n = nv
        self.adj = [[] for _ in range(nv)]
        for a, b in edges:
            self.adj[a].append(b)
            self.adj[b].append(a)
        self.nodes = 0
        self.budget = _EXACT_NODE_BUDGET
        self.deadline = (
            time.monotonic() + _EXACT_TIME_LIMIT_SECONDS if deadline is None else deadline
        )

    def _tick(self):
        self.nodes += 1
        if self.nodes > self.budget:
            raise _SearchExhausted
        if (self.nodes & 8191) == 0 and time.monotonic() > self.deadline:
            raise _SearchExhausted

    def assign(self, domains, v, k):
        """Pin v to color k and propagate singleton domains over the
        disequality graph.  Returns an undo list, or ``None`` on wipeout
        (in which case domains are fully restored)."""
        undo = [(v, domains[v])]
        domains[v] = 1 << k
        queue = [(u, k) for u in self.adj[v]]
        while queue:
            x, forb = queue.pop()
            d = domains[x]
            if not (d & (1 << forb)):
                continue
            nd = d & ~(1 << forb)
            if nd == 0:
                for var, old in reversed(undo):
                    domains[var] = old
                return None
            undo.append((x, d))
            domains[x] = nd
            if nd & (nd - 1) == 0:  # singleton: propagate in turn
                c = nd.bit_length() - 1
                for u in self.adj[x]:
                    if domains[u] & (1 << c):
                        queue.append((u, c))
        return undo

    @staticmethod
    def restore(domains, undo):
        for var, old in reversed(undo):
            domains[var] = old

    def colorable(self, domains=None, pinned=()):
        """Whether a proper 3-coloring exists under ``domains`` plus the
        (v, k) pins.  MAC/MRV search; any domains changed here are restored
        before returning, so the caller's state is untouched.

        Canonicity is deliberately *not* enforced: a prefix extends to some
        proper coloring iff it extends to a canonical one (renaming the
        already-used colors onto themselves and any new colors onto the
        smallest unused masks never breaks a disequality).
        """
        if domains is None:
            domains = [7] * self.n
        marks = []
        for v, k in pinned:
            undo = self.assign(domains, v, k)
            if undo is None:
                self.restore(domains, marks)
                return False
            marks.extend(undo)
        try:
            return self._colorable_rec(domains)
        finally:
            self.restore(domains, marks)

    def _colorable_rec(self, domains):
        self._tick()
        # MRV: smallest non-singleton domain; a zero domain means dead end.
        v = -1
        best = 4
        for x in range(self.n):
            c = domains[x].bit_count()
            if c == 0:
                return False
            if 1 < c < best:
                best, v = c, x
        if v < 0:
            return True
        for k in MASKS:
            if domains[v] & (1 << k):
                undo = self.assign(domains, v, k)
                if undo is None:
                    continue
                ok = self._colorable_rec(domains)
                self.restore(domains, undo)
                if ok:
                    return True
        return False


class _ExactSearch:
    """Exact branch-and-bound in arbitrary-precision integer arithmetic.

    Fragment positions (ascending id) carry bitmask domains pruned by
    singleton conflict propagation (pinning a mask forbids it on every
    neighbor, cascading); feasibility probes pick the next fragment with MRV
    dynamic variable ordering, which is what lets a sparse 48-fragment graph
    be certified instead of enumerated in a fixed id order.

    Stitch weights contribute an exact cut total plus a valid per-edge lower
    bound: a colored/uncolored stitch is forced cut once the colored mask has
    been pruned from the uncolored endpoint's domain, and an
    uncolored/uncolored stitch is forced cut once the two domains are
    disjoint.  Every quantity is a Python int, so stitch weights of any size
    compare exactly.

    The canonical lexicographic primary and the distinct witness are built
    in ascending id order, each tentative placement certified by a bounded
    suffix-feasibility probe (a prefix extends to some proper 3-coloring iff
    it extends to a canonical one, so the probe need not track first
    occurrences).
    """

    def __init__(self, order, conflict_edges, stitches, deadline=None):
        n = len(order)
        self.n = n
        pos = {v: i for i, v in enumerate(order)}
        self.order = order
        self.pos = pos
        self.conf = [[] for _ in range(n)]
        for a, b in conflict_edges:
            ia, ib = pos[a], pos[b]
            self.conf[ia].append(ib)
            self.conf[ib].append(ia)
        self.stitches = [(pos[a], pos[b], w) for a, b, w in stitches]
        self.st = [[] for _ in range(n)]
        for ia, ib, w in self.stitches:
            self.st[ia].append((ib, w))
            self.st[ib].append((ia, w))
        self.total = sum(w for _a, _b, w in self.stitches)
        self.graph = _ConstraintGraph(
            n,
            [(i, j) for i in range(n) for j in self.conf[i] if i < j],
            deadline,
        )

    # -- exact lower bound -------------------------------------------------

    def _lower_bound(self, domains, colors, cost):
        """Lower bound on the final cut total.

        Edges with both endpoints colored already sit in ``cost``.  A
        colored/uncolored edge is forced cut exactly when the colored mask
        has been pruned from the uncolored endpoint's domain; an
        uncolored/uncolored edge is forced cut when its endpoint domains are
        disjoint.  Every edge is counted at most once.
        """
        lb = cost
        for a, b, w in self.stitches:
            ca, cb = colors[a], colors[b]
            if ca >= 0 and cb >= 0:
                continue
            if ca < 0 and cb < 0:
                if domains[a] & domains[b] == 0:
                    lb += w
            else:
                u, c = (a, cb) if ca < 0 else (b, ca)
                if not (domains[u] & (1 << c)):
                    lb += w
        return lb

    def _added(self, colors, v, k):
        return sum(w for u, w in self.st[v] if colors[u] >= 0 and colors[u] != k)

    # -- optimum -----------------------------------------------------------

    def optimize(self):
        """Minimum cut total, or ``None`` if no proper 3-coloring exists."""
        n = self.n
        domains = [7] * n
        colors = [-1] * n
        uncolored = set(range(n))
        best = [None]
        limit = [self.total + 1]

        def branch(cost):
            self.graph._tick()
            if self._lower_bound(domains, colors, cost) >= limit[0]:
                return
            if not uncolored:
                best[0] = cost
                limit[0] = cost
                return
            # MRV on domain width; ties favor the fragment with more
            # uncolored conflict neighbors, then the smaller id.
            v = min(
                uncolored,
                key=lambda x: (
                    domains[x].bit_count(),
                    -sum(1 for u in self.conf[x] if colors[u] < 0),
                    x,
                ),
            )
            dv = domains[v]
            uncolored.discard(v)
            cands = []
            for k in MASKS:
                if dv & (1 << k):
                    cands.append((self._added(colors, v, k), k))
            # Cheapest forced cut first finds tight incumbents early; ties
            # break by mask id for determinism.
            cands.sort()
            for add, k in cands:
                nc = cost + add
                if nc >= limit[0]:
                    continue
                undo = self.graph.assign(domains, v, k)
                if undo is None:
                    continue
                colors[v] = k
                branch(nc)
                colors[v] = -1
                self.graph.restore(domains, undo)
                if limit[0] == 0:
                    break
            uncolored.add(v)

        branch(0)
        return best[0]

    # -- suffix feasibility under a cut budget -----------------------------

    def _suffix_feasible(self, domains, colors, start, cost, budget):
        """Can the id-order prefix (positions < ``start`` colored) extend to
        a coloring with cut total at most ``budget``?  Restores every bit of
        state it creates.  MRV + singleton propagation + exact bound."""
        n = self.n

        def rec(cur_cost):
            self.graph._tick()
            if self._lower_bound(domains, colors, cur_cost) > budget:
                return False
            v = -1
            best_width = 4
            for x in range(start, n):
                c = domains[x].bit_count()
                if c == 0:
                    return False
                if 1 < c < best_width:
                    best_width, v = c, x
            if v < 0:
                return cur_cost <= budget
            dv = domains[v]
            cands = []
            for k in MASKS:
                if dv & (1 << k):
                    cands.append((self._added(colors, v, k), k))
            cands.sort()
            for add, k in cands:
                nc = cur_cost + add
                if nc > budget:
                    continue
                undo = self.graph.assign(domains, v, k)
                if undo is None:
                    continue
                colors[v] = k
                ok = rec(nc)
                colors[v] = -1
                self.graph.restore(domains, undo)
                if ok:
                    return True
            return False

        return rec(cost)

    # -- canonical primary and witness -------------------------------------

    def lexmin(self, best_cost):
        """Lexicographically smallest canonical coloring with cut total
        ``best_cost``; positions are taken in ascending id order and each
        choice is certified by a suffix feasibility probe."""
        n = self.n
        domains = [7] * n
        colors = [-1] * n
        result = []
        cost = 0
        max_used = -1
        pinned = []
        try:
            for i in range(n):
                chosen = None
                hi = min(max_used + 1, 2)
                for k in MASKS:
                    if k > hi or not (domains[i] & (1 << k)):
                        continue
                    add = self._added(colors, i, k)
                    if cost + add > best_cost:
                        continue
                    undo = self.graph.assign(domains, i, k)
                    if undo is None:
                        continue
                    colors[i] = k
                    feasible = self._suffix_feasible(
                        domains, colors, i + 1, cost + add, best_cost
                    )
                    colors[i] = -1
                    self.graph.restore(domains, undo)
                    if feasible:
                        chosen = (k, add)
                        break
                if chosen is None:  # pragma: no cover - best_cost is achievable
                    raise SolverError("求解器在构造字典序最小方案时失败")
                k, add = chosen
                # Commit the certified choice and keep it pinned for the
                # remaining positions.
                undo = self.graph.assign(domains, i, k)
                assert undo is not None
                colors[i] = k
                pinned.append(undo)
                cost += add
                max_used = max(max_used, k)
                result.append(k)
        except BaseException:
            for undo in reversed(pinned):
                self.graph.restore(domains, undo)
            raise
        return result

    def find_other(self, best_cost, exclude):
        """A canonical optimum different from ``exclude`` (tuple of colors
        in id order), or ``None`` if the optimum is unique."""
        n = self.n
        domains = [7] * n
        colors = [-1] * n
        found = [None]

        def rec(i, max_used, cost):
            self.graph._tick()
            if found[0] is not None:
                return
            hi = min(max_used + 1, 2)
            for k in MASKS:
                if k > hi or not (domains[i] & (1 << k)):
                    continue
                add = self._added(colors, i, k)
                if cost + add > best_cost:
                    continue
                undo = self.graph.assign(domains, i, k)
                if undo is None:
                    continue
                colors[i] = k
                if i + 1 == n:
                    t = tuple(colors)
                    if t != exclude:
                        found[0] = t
                elif self._suffix_feasible(
                    domains, colors, i + 1, cost + add, best_cost
                ):
                    rec(i + 1, max(max_used, k), cost + add)
                colors[i] = -1
                self.graph.restore(domains, undo)
                if found[0] is not None:
                    return

        rec(0, -1, 0)
        return found[0]


def _solve_zero_cost_equality(order, conflict_edges, stitches, deadline=None):
    """Solve the cost-zero case by treating every stitch as an equality.

    A cut total of zero forces both endpoints of every stitch onto the same
    mask, so union-find merges them into components; the instance has a
    zero-cost solution iff the quotient graph (conflict edges between
    components, a conflict inside a component is an immediate contradiction)
    is 3-colorable.

    Returns the same response shape as ``solve_mask_assignment`` when the
    optimum is zero (including the lexicographically smallest canonical
    fragment coloring and a distinct witness when one exists), or ``None``
    when the optimum cannot be zero.
    """
    n = len(order)
    pos = {v: i for i, v in enumerate(order)}

    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for a, b, _w in stitches:
        union(pos[a], pos[b])

    comp_of = [find(i) for i in range(n)]
    roots = sorted(set(comp_of))
    root_idx = {r: c for c, r in enumerate(roots)}
    nc = len(roots)
    members = [[] for _ in range(nc)]
    for i in range(n):
        members[root_idx[comp_of[i]]].append(i)
    for m in members:
        m.sort()

    qedges = set()
    for a, b in conflict_edges:
        ca, cb = root_idx[comp_of[pos[a]]], root_idx[comp_of[pos[b]]]
        if ca == cb:
            return None  # a stitch-merged pair must disagree: zero is impossible
        qedges.add((ca, cb) if ca < cb else (cb, ca))

    qgraph = _ConstraintGraph(nc, sorted(qedges), deadline)
    if not qgraph.colorable():
        # The merged constraints admit no 3-coloring at all: zero is
        # impossible (and the instance may be outright infeasible; the
        # caller's general search certifies that).
        return None

    # Components are named (for the canonical scan) by their smallest member
    # id: every fragment before that minimum belongs to a component named
    # still earlier, so assigning components in this order with the usual
    # first-occurrence rule reproduces the fragment-id canonical order.
    corder = sorted(range(nc), key=lambda c: members[c][0])

    def expand(ccolors):
        colors = [0] * n
        for c in range(nc):
            for i in members[c]:
                colors[i] = ccolors[c]
        return colors

    def lexmin_components():
        domains = [7] * nc
        ccolors = [-1] * nc
        result = [-1] * nc
        max_used = -1
        pinned = []
        try:
            for c in corder:
                chosen = None
                hi = min(max_used + 1, 2)
                for k in MASKS:
                    if k > hi or not (domains[c] & (1 << k)):
                        continue
                    undo = qgraph.assign(domains, c, k)
                    if undo is None:
                        continue
                    # colorable restores only its own probe marks, leaving
                    # the candidate pin in place on success.
                    if qgraph.colorable(domains):
                        chosen = (k, undo)
                        break
                    qgraph.restore(domains, undo)
                if chosen is None:  # pragma: no cover - zero solution exists
                    raise SolverError("求解器在构造字典序最小方案时失败")
                k, undo = chosen
                pinned.append(undo)
                ccolors[c] = k
                result[c] = k
                max_used = max(max_used, k)
        except BaseException:
            for undo in reversed(pinned):
                qgraph.restore(domains, undo)
            raise
        return result

    primary_cc = lexmin_components()
    primary = expand(primary_cc)

    # Uniqueness / witness: another canonical zero-cost fragment coloring.
    # Search at component level in naming order; a differing component
    # coloring is exactly a differing fragment coloring.
    domains = [7] * nc
    ccolors = [-1] * nc
    other = [None]
    primary_comp_t = tuple(primary_cc)

    def rec(ci, max_used):
        if other[0] is not None:
            return
        qgraph._tick()
        c = corder[ci]
        hi = min(max_used + 1, 2)
        for k in MASKS:
            if k > hi or not (domains[c] & (1 << k)):
                continue
            undo = qgraph.assign(domains, c, k)
            if undo is None:
                continue
            ccolors[c] = k
            if ci + 1 == nc:
                t = tuple(ccolors)
                if t != primary_comp_t:
                    other[0] = expand(ccolors)
            elif qgraph.colorable(domains):
                rec(ci + 1, max(max_used, k))
            ccolors[c] = -1
            qgraph.restore(domains, undo)
            if other[0] is not None:
                return

    if nc > 1:
        rec(0, -1)
    unique = other[0] is None

    witness = None
    if not unique:
        witness = {
            "assignment": {str(order[i]): other[0][i] for i in range(n)},
            "cut_stitches": [],
        }

    return {
        "status": "optimal",
        "objective": 0,
        "unique": unique,
        "assignment": {str(order[i]): primary[i] for i in range(n)},
        "cut_stitches": [],
        "witness": witness,
    }


def _solve_exact(order, conflict_edges, stitches):
    """Solve one validated instance in exact integer arithmetic.

    Used when stitch weights exceed the domain where the CBC MILP is
    exact (or as a fallback when CBC cannot certify); returns the same
    response shape as ``solve_mask_assignment``.
    """
    n = len(order)
    pos = {v: i for i, v in enumerate(order)}
    deadline = time.monotonic() + _EXACT_TIME_LIMIT_SECONDS

    # Fast exact path for a zero optimum: with no cut budget every stitch
    # must stay uncut, so merge equalities and test the quotient graph for
    # 3-colorability.  With no stitches this is plain conflict feasibility
    # and also avoids enumerating the unconstrained colorings.
    zero = _solve_zero_cost_equality(order, conflict_edges, stitches, deadline)
    if zero is not None:
        return zero

    search = _ExactSearch(order, conflict_edges, stitches, deadline)
    try:
        best = search.optimize()
        if best is None:
            return {"status": "infeasible"}
        colors = search.lexmin(best)
        other = search.find_other(best, tuple(colors))
    except _SearchExhausted:
        raise SolverError("求解器未能在限定时间内求得最优解") from None

    unique = other is None
    witness = None
    if not unique:
        witness = {
            "assignment": {str(order[i]): other[i] for i in range(n)},
            "cut_stitches": _cut_stitches(stitches, pos, other),
        }

    return {
        "status": "optimal",
        "objective": best,
        "unique": unique,
        "assignment": {str(order[i]): colors[i] for i in range(n)},
        "cut_stitches": _cut_stitches(stitches, pos, colors),
        "witness": witness,
    }


def solve_mask_assignment(fragments, conflict_edges, stitch_edges):
    """Solve one validated instance.

    Returns ``{"status": "infeasible"}`` when the conflict graph admits no
    three-mask coloring.  Otherwise returns the lexicographically smallest
    canonical optimum (fragment ids ascending), whether that optimum is the
    unique canonical optimum, and — when it is not — a second, different
    canonical optimum as a witness.
    """
    order = sorted(fragments)
    stitches = [tuple(edge) for edge in stitch_edges]
    weights = [w for _a, _b, w in stitches]
    milp_exact = not weights or (
        max(weights) <= _MILP_MAX_SAFE_WEIGHT
        and sum(weights) <= _MILP_MAX_SAFE_TOTAL
    )
    if milp_exact:
        try:
            return _solve_milp(order, conflict_edges, stitches)
        except SolverError:
            # CBC could not certify a result; the exact integer search
            # always can (within its resource budget).
            return _solve_exact(order, conflict_edges, stitches)
    return _solve_exact(order, conflict_edges, stitches)
