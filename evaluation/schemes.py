"""
schemes.py -- per-payment routing/rebalancing logic for every scheme, run on
the shared Sim from common.py.

Every scheme's work() consumes the *same* graph, balances, and payment
sequence (for a fixed seed) and returns a uniform RunResult.  Crucially, each
run is **instrumented**: for every successful payment we record whether it
required rebalancing and, if so, its rebalancing *depth* k (number of shift /
draw operations).  This per-payment log is what the Coordination-Cost Model
(coordination.py) needs to turn the baselines' idealized zero-latency success
ratio into the latency-aware effective success ratio of the paper
(Prop.~\\ref{prop:coord-cost}, Fig.~\\ref{fig:latency}).

Schemes
-------
LN       : no rebalancing.  A short hop fails outright.  (paper baseline i)
Shaduf   : node-local coin shift among *bound* channels.  Port of
           github-shaduf/evaluation/shaduf.py.  depth k = #bound channels
           touched.  (paper baseline iii)
Horcrux  : along-path coin shift.  Port of
           github-horcrux/simulation_in_testnet/horcrux.py.  depth k =
           #path hops that had to shift.
Ballast  : node-level logical draw against a single bond that backs all of a
           node's channels.  No coin movement, no counterparty cooperation;
           depth k = #draws (each a single node-local round trip).
Ballast-Q: v8 epoch-quota variant. One checkpoint partitions each node bond
           into per-channel exposure caps; bilateral fast-path draws stay
           within those caps. Checkpoints adapt caps from the previous epoch's
           observed shortfall demand and never reduce a cap below outstanding
           exposure.

Fairness (equal locked capital).  Ballast does not add capital: it *skims* a
fraction phi of every channel's balance into a node-level bond and runs its
channels at (1-phi) of the baseline.  Per-node locked capital is therefore
identical to the baselines'; the only difference is that the skimmed capital is
pooled into one fungible bond instead of being stranded per channel.  phi is
exposed (default 0.30) and the service-level study (sizing.py) sweeps the bond
size directly.
"""

import collections
from functools import cmp_to_key

import networkx as nx

from common import tuple_sort, tuple_trans


def _route(sim, source, target):
    """Use configurable routing while preserving minimal test simulators."""
    if hasattr(sim, "route"):
        return sim.route(source, target)
    return nx.shortest_path(sim.G, source, target)


class RunResult:
    def __init__(self):
        self.n_total = 0
        self.n_success = 0
        self.success_volume = 0
        self.total_volume = 0
        # per successful payment that needed rebalancing: depth k (int).
        # payments that succeeded with NO rebalancing are not listed (k=0).
        self.rebal_depths = []
        # Ballast-only diagnostics for the capital-efficiency study (Q3):
        self.draw_events = []        # (node, draw_amount, node_outstanding_after)
        # Ballast-only timestamped directional shortfalls.  This is kept
        # separately so the sizing API above remains backward compatible.
        # Entries are (payment_index, operator, counterparty, amount).
        self.channel_demand_events = []
        self.channel_peak_short = {} # directed (operator, counterparty) -> peak exposure
        self.node_bond = {}          # node -> bond posted
        self.refusals = 0            # honest draws refused for lack of bond headroom
        self.checkpoint_events = 0   # Ballast-Q: epoch checkpoints executed
        self.slow_path_events = 0    # Ballast-Q+: payments using overflow receipt

    @property
    def success_ratio(self):
        return self.n_success / self.n_total if self.n_total else 0.0


# --------------------------------------------------------------------------
# LN -- vanilla Lightning, no rebalancing
# --------------------------------------------------------------------------
def work_ln(sim, tx_load, mode="uniform", skew_param=None):
    res = RunResult()
    for i in range(tx_load):
        t1, t2 = sim.sample_pair(mode, skew_param)
        path = _route(sim, t1, t2)
        amt = sim.tx[i]
        ok = all(sim.get_within(path[j], path[j + 1])[0] >= amt
                 for j in range(len(path) - 1))
        if ok:
            for j in range(len(path) - 1):
                z0, z1 = sim.get_within(path[j], path[j + 1])
                sim.update_within(path[j], path[j + 1], z0 - amt, z1 + amt)
            res.n_success += 1
            res.success_volume += amt
            # k = 0 : LN never rebalances, so nothing is appended.
        res.n_total += 1
        res.total_volume += amt
    return res


# --------------------------------------------------------------------------
# Shaduf -- node-local shift among bound channels (port of shaduf.py)
# --------------------------------------------------------------------------
def _shaduf_bind(sim, inter, all_inter, mode):
    """Set up bindings exactly as shaduf.py's bind()."""
    def seg(x):
        return int(0.5 * x)

    for node in list(sim.G.nodes()):
        neighbors = list(sim.G[node])
        degree = len(neighbors)
        if degree == 1:
            continue
        bal = {nb: sim.get_within(node, nb)[0] for nb in neighbors}
        h2l = dict(sorted(bal.items(), key=lambda x: x[1], reverse=True))
        l2h = collections.OrderedDict(reversed(list(h2l.items())))
        pool = []
        if mode == "high-to-low":
            for i in range(seg(degree)):
                pool.append((list(h2l.items())[i][0], list(l2h.items())[i][0]))
        else:  # all-bind / random-bind both enumerate pairs; we use all-bind
            for i in range(degree):
                for j in range(i + 1, degree):
                    pool.append((neighbors[i], neighbors[j]))
            if mode == "random-bind":
                import random
                pool = random.sample(pool, seg(degree))

        cnt = collections.Counter()
        for t1, t2 in pool:
            cnt[t1] += 1
            cnt[t2] += 1
        share = {}
        for t, c in cnt.items():
            share[t] = sim.get_within(node, t)[0] // c
        for t1, t2 in pool:
            key = tuple_trans((t1, node), (node, t2))
            a1, a2 = share[t1], share[t2]
            if key[1] == t1:
                inter[key] = (a1, a2)
            else:
                inter[key] = (a2, a1)
            all_inter.setdefault((node, t1), []).append(t2)
            all_inter.setdefault((node, t2), []).append(t1)


def _get_inter(inter, a, b, c):
    key = tuple_trans((a, b), (b, c))
    if key not in inter:
        return None
    v = inter[key]
    return v if key[1] == a else v[::-1]


def _set_inter(inter, a, b, c, amt1, amt2):
    key = tuple_trans((a, b), (b, c))
    inter[key] = (amt1, amt2) if key[1] == a else (amt2, amt1)


def work_shaduf(sim, tx_load, bind_mode="high-to-low", mode="uniform", skew_param=None):
    inter, all_inter = {}, {}
    _shaduf_bind(sim, inter, all_inter, bind_mode)
    res = RunResult()

    def max_amt_channel(a, b):
        total = 0
        for obj in all_inter.get((a, b), []):
            i = _get_inter(inter, obj, a, b)[0]
            w = sim.get_within(obj, a)[1]
            total += min(i, w)
        return total

    def do_shift(a, b, amt, journal):
        """shift `amt` of a's coins from bound channels into (a,b); return #moves."""
        rl = {}
        for obj in all_inter.get((a, b), []):
            i = _get_inter(inter, obj, a, b)[0]
            w = sim.get_within(obj, a)[1]
            rl[obj] = min(i, w)
        rl = dict(sorted(rl.items(), key=lambda x: x[1], reverse=True))
        moves = 0
        for obj, avail in rl.items():
            cur = min(avail, amt)
            if cur <= 0:
                continue
            iv = _get_inter(inter, obj, a, b)
            donor_before = sim.get_within(obj, a)
            target_before = sim.get_within(a, b)
            journal.append((obj, a, b, iv, donor_before, target_before))
            _set_inter(inter, obj, a, b, iv[0] - cur, iv[1] + cur)
            w0, w1 = donor_before
            sim.update_within(obj, a, w0, w1 - cur)
            z0, z1 = target_before
            sim.update_within(a, b, z0 + cur, z1)
            amt -= cur
            moves += 1
            if amt == 0:
                break
        return moves

    for i in range(tx_load):
        t1, t2 = sim.sample_pair(mode, skew_param)
        path = _route(sim, t1, t2)
        amt = sim.tx[i]

        flag, depth, journal = True, 0, []
        for j in range(len(path) - 1):
            z0 = sim.get_within(path[j], path[j + 1])[0]
            if z0 >= amt:
                continue
            need = amt - z0
            if max_amt_channel(path[j], path[j + 1]) >= need:
                depth += do_shift(path[j], path[j + 1], need, journal)
            else:
                flag = False
                break
        if not flag:
            # A payment is atomic: speculative shifts used to test earlier
            # hops must not survive when a later hop proves infeasible.
            for obj, a, b, iv, donor_before, target_before in reversed(journal):
                _set_inter(inter, obj, a, b, iv[0], iv[1])
                sim.update_within(obj, a, *donor_before)
                sim.update_within(a, b, *target_before)
        if flag:
            for j in range(len(path) - 1):
                z0, z1 = sim.get_within(path[j], path[j + 1])
                sim.update_within(path[j], path[j + 1], z0 - amt, z1 + amt)
            res.n_success += 1
            res.success_volume += amt
            if depth > 0:
                res.rebal_depths.append(depth)
        res.n_total += 1
        res.total_volume += amt
    return res


# --------------------------------------------------------------------------
# Horcrux -- along-path coin shift (port of horcrux.py)
# --------------------------------------------------------------------------
def work_horcrux(sim, tx_load, mode="uniform", skew_param=None):
    res = RunResult()
    for i in range(tx_load):
        t1, t2 = sim.sample_pair(mode, skew_param)
        path = _route(sim, t1, t2)
        amt = sim.tx[i]

        flag, depth = True, 0
        for j in range(len(path) - 1):
            z_r = sim.get_within(path[j], path[j + 1])[0]
            if z_r >= amt:
                continue
            elif j > 0:
                z_l = sim.get_within(path[j - 1], path[j])[1]
                if z_l + z_r >= amt:
                    depth += 1   # this hop must pull from the upstream channel
                    continue
                else:
                    flag = False
                    break
            else:
                flag = False
                break

        if flag:
            for j in range(len(path) - 1):
                z0, z1 = sim.get_within(path[j], path[j + 1])
                if j == 0 and z0 >= amt:
                    sim.update_within(path[j], path[j + 1], z0 - amt, z1 + amt)
                    continue
                z0_l, z1_l = sim.get_within(path[j - 1], path[j])
                if z1_l - amt >= amt:
                    sim.update_within(path[j - 1], path[j], z0_l, z1_l - amt)
                    sim.update_within(path[j], path[j + 1], z0, z1 + amt)
                else:
                    sim.update_within(path[j - 1], path[j], z0_l, amt)
                    sim.update_within(path[j], path[j + 1],
                                      z1_l + z0 - 2 * amt, z1 + amt)
            res.n_success += 1
            res.success_volume += amt
            if depth > 0:
                res.rebal_depths.append(depth)
        res.n_total += 1
        res.total_volume += amt
    return res


# --------------------------------------------------------------------------
# Ballast -- node-level logical draw against a single bond
# --------------------------------------------------------------------------
def work_ballast(sim, tx_load, bond_fraction=0.30, mode="uniform", skew_param=None):
    """Skim `bond_fraction` of every channel into a per-node bond (conserving
    each node's locked capital), then serve depletion by logically drawing
    against the bond -- no coin movement, no counterparty cooperation.

    Invariant enforced every draw: sum of a node's outstanding draws never
    exceeds its bond (the paper's sum_i b_i <= bond).  The bond is a revolving
    line: inbound liquidity on any of a node's channels retires outstanding
    draw, freeing headroom.
    """
    phi = bond_fraction
    bond = {}
    # Exposure is pooled for admission but remains attributable to the served
    # counterparty.  A reverse payment on (v,u) may repay v's exposure to u; an
    # unrelated inbound payment must not silently erase another counterparty's
    # claim.  The previous aggregate-only implementation made exactly that
    # unsafe cross-channel reduction and therefore overstated revolving
    # headroom.
    outstanding = collections.defaultdict(int)       # directed (u,v) -> exposure
    total_outstanding = collections.defaultdict(int) # u -> sum_v exposure(u,v)

    # skim phi into the bond, conserving each node's total locked capital
    for ch in list(sim.within.keys()):
        a, b = ch
        ba, bb = sim.within[ch]
        sa, sb = int(ba * phi), int(bb * phi)
        sim.within[ch] = (ba - sa, bb - sb)
        bond[a] = bond.get(a, 0) + sa
        bond[b] = bond.get(b, 0) + sb
    for n in sim.nodes:
        bond.setdefault(n, 0)

    res = RunResult()
    res.node_bond = dict(bond)

    def headroom(u):
        return bond[u] - total_outstanding[u]

    for i in range(tx_load):
        t1, t2 = sim.sample_pair(mode, skew_param)
        path = _route(sim, t1, t2)
        amt = sim.tx[i]

        # feasibility: each short hop's sender must have bond headroom.
        flag, depth, refused = True, 0, False
        planned = collections.defaultdict(int)  # node -> draw planned this payment
        draws = []                               # (u, v, zmax)
        for j in range(len(path) - 1):
            u, v = path[j], path[j + 1]
            z0 = sim.get_within(u, v)[0]
            if z0 >= amt:
                continue
            zmax = amt - z0
            if headroom(u) - planned[u] >= zmax:
                planned[u] += zmax
                draws.append((u, v, zmax))
            else:
                flag = False
                refused = True
                break

        if flag:
            # apply draws: logical top-up backed by the bond
            for (u, v, zmax) in draws:
                outstanding[(u, v)] += zmax
                total_outstanding[u] += zmax
                res.draw_events.append((u, zmax, total_outstanding[u]))
                res.channel_demand_events.append((i, u, v, zmax))
                key = (u, v)
                res.channel_peak_short[key] = max(
                    res.channel_peak_short.get(key, 0), outstanding[key]
                )
                z0, z1 = sim.get_within(u, v)
                sim.update_within(u, v, z0 + zmax, z1)  # now z0+zmax == amt
            # Perform the payment debits.  Reverse flow retires exposure only
            # on the same directed channel, matching the coordinate-specific
            # repay relation and counterparty authorization in the paper.
            for j in range(len(path) - 1):
                u, v = path[j], path[j + 1]
                z0, z1 = sim.get_within(u, v)
                sim.update_within(u, v, z0 - amt, z1 + amt)
                reverse = (v, u)
                repaid = min(amt, outstanding[reverse])
                if repaid:
                    outstanding[reverse] -= repaid
                    total_outstanding[v] -= repaid
            res.n_success += 1
            res.success_volume += amt
            if depth or draws:
                res.rebal_depths.append(len(draws))  # k = #node-local draws
        else:
            if refused:
                res.refusals += 1
        res.n_total += 1
        res.total_volume += amt
    return res


# --------------------------------------------------------------------------
# Ballast-Q -- v8 epoch quotas with a bilateral fast path
# --------------------------------------------------------------------------
def work_ballast_epoch(sim, tx_load, bond_fraction=0.30, epoch_size=1000,
                       overflow_fraction=0.0, mode="uniform", skew_param=None):
    """Safe amortized variant used by the v8 paper.

    A node-level bond is checkpointed into directional channel quotas whose sum
    equals that node's bond. During an epoch, a channel can draw only up to its
    quota. At a boundary, outstanding exposure is carried forward (so an old
    obligation is never reset) and only free headroom is reassigned using the
    previous epoch's observed shortfall demand. This models the on-chain
    checkpoint plus affected-counterparty authorization: a quota is never
    reduced below the jointly signed outstanding channel state.
    """
    if epoch_size <= 0:
        raise ValueError("epoch_size must be positive")
    if not 0.0 <= overflow_fraction <= 1.0:
        raise ValueError("overflow_fraction must be in [0,1]")

    phi = bond_fraction
    bond = {}
    outstanding = collections.defaultdict(int)  # directed (u,v) exposure
    overflow = collections.defaultdict(int)     # directed slow-path exposure
    overflow_total = collections.defaultdict(int)
    quota = collections.defaultdict(int)        # directed (u,v) epoch cap
    demand = collections.defaultdict(int)       # observed shortfall this epoch

    # Equal-capital skim, identical to work_ballast.
    for ch in list(sim.within.keys()):
        a, b = ch
        ba, bb = sim.within[ch]
        sa, sb = int(ba * phi), int(bb * phi)
        sim.within[ch] = (ba - sa, bb - sb)
        bond[a] = bond.get(a, 0) + sa
        bond[b] = bond.get(b, 0) + sb
    for n in sim.nodes:
        bond.setdefault(n, 0)
    overflow_cap = {n: int(bond[n] * overflow_fraction) for n in sim.nodes}
    base_cap = {n: bond[n] - overflow_cap[n] for n in sim.nodes}

    res = RunResult()
    res.node_bond = dict(bond)

    def allocate_node(u):
        neighbors = list(sim.G[u])
        if not neighbors:
            return
        carried = sum(outstanding[(u, v)] for v in neighbors)
        if carried > base_cap[u]:
            raise ValueError("outstanding exposure exceeds node bond")
        free = base_cap[u] - carried
        weights = [demand[(u, v)] for v in neighbors]
        total_weight = sum(weights)
        if total_weight == 0:
            weights = [1] * len(neighbors)
            total_weight = len(neighbors)

        assigned = 0
        for pos, v in enumerate(neighbors):
            if pos + 1 == len(neighbors):
                share = free - assigned
            else:
                share = (free * weights[pos]) // total_weight
                assigned += share
            quota[(u, v)] = outstanding[(u, v)] + share

    # Cold-start checkpoint uses equal quotas; later checkpoints are demand-aware.
    for u in sim.nodes:
        allocate_node(u)
    res.checkpoint_events = 1

    for i in range(tx_load):
        if i > 0 and i % epoch_size == 0:
            for u in sim.nodes:
                allocate_node(u)
            demand.clear()
            res.checkpoint_events += 1

        t1, t2 = sim.sample_pair(mode, skew_param)
        path = _route(sim, t1, t2)
        amt = sim.tx[i]

        flag, refused = True, False
        planned = collections.defaultdict(int)
        planned_overflow = collections.defaultdict(int)
        draws = []
        for j in range(len(path) - 1):
            u, v = path[j], path[j + 1]
            z0 = sim.get_within(u, v)[0]
            if z0 >= amt:
                continue
            need = amt - z0
            demand[(u, v)] += need
            base_room = quota[(u, v)] - outstanding[(u, v)] - planned[(u, v)]
            base_take = min(need, max(0, base_room))
            spill = need - base_take
            over_room = overflow_cap[u] - overflow_total[u] - planned_overflow[u]
            if spill <= over_room:
                planned[(u, v)] += base_take
                planned_overflow[u] += spill
                draws.append((u, v, base_take, spill))
            else:
                flag = False
                refused = True
                break

        if flag:
            used_slow = False
            for u, v, base_take, spill in draws:
                need = base_take + spill
                outstanding[(u, v)] += base_take
                overflow[(u, v)] += spill
                overflow_total[u] += spill
                used_slow = used_slow or spill > 0
                res.draw_events.append(
                    (u, need, outstanding[(u, v)] + overflow[(u, v)])
                )
                ch = tuple_sort((u, v))
                res.channel_peak_short[ch] = max(
                    res.channel_peak_short.get(ch, 0), need
                )
                z0, z1 = sim.get_within(u, v)
                sim.update_within(u, v, z0 + need, z1)

            for j in range(len(path) - 1):
                u, v = path[j], path[j + 1]
                z0, z1 = sim.get_within(u, v)
                sim.update_within(u, v, z0 - amt, z1 + amt)
                # Reverse flow on this same channel retires v's exposure to u.
                key = (v, u)
                restored = amt
                if overflow[key] > 0:
                    paid = min(restored, overflow[key])
                    overflow[key] -= paid
                    overflow_total[v] -= paid
                    restored -= paid
                if restored > 0 and outstanding[key] > 0:
                    outstanding[key] = max(0, outstanding[key] - restored)

            res.n_success += 1
            res.success_volume += amt
            if draws:
                res.rebal_depths.append(len(draws))
            if used_slow:
                res.slow_path_events += 1
        elif refused:
            res.refusals += 1

        res.n_total += 1
        res.total_volume += amt

    return res


# --------------------------------------------------------------------------
# Ballast ablation -- logical draw WITHOUT pooling (per-channel reserve)
# --------------------------------------------------------------------------
def work_ballast_perchannel(sim, tx_load, bond_fraction=0.30,
                            mode="uniform", skew_param=None):
    """Ablation of Ballast that removes ONLY the pooling.

    It skims the *same* fraction phi from every channel and serves depletion by
    the *same* logical-draw / revolving-repayment mechanics as work_ballast --
    no coin movement, no counterparty cooperation, identical latency.  The one
    difference: a channel's skim stays a *per-channel* reserve.  A depletion on
    hop (u,v) may draw only against the reserve skimmed from THAT channel's
    u-side, never against reserve sitting on u's other channels.

    Therefore (work_ballast - work_ballast_perchannel) at L=0 isolates the pure
    statistical-multiplexing gain: same capital, same mechanics, the only delta
    is whether the reserve is pooled across a node's channels or stranded per
    channel.
    """
    phi = bond_fraction
    reserve = {}                                 # directed (u,v) -> u's reserve to top up u->v
    outstanding = collections.defaultdict(int)   # directed (u,v) -> outstanding draw

    for ch in list(sim.within.keys()):
        a, b = ch
        ba, bb = sim.within[ch]
        sa, sb = int(ba * phi), int(bb * phi)
        sim.within[ch] = (ba - sa, bb - sb)
        reserve[(a, b)] = sa
        reserve[(b, a)] = sb

    res = RunResult()
    nb = collections.defaultdict(int)            # per-node bond = sum of its channel reserves
    for (u, _v), r in reserve.items():
        nb[u] += r
    res.node_bond = dict(nb)

    def headroom(u, v):
        return reserve.get((u, v), 0) - outstanding[(u, v)]

    for i in range(tx_load):
        t1, t2 = sim.sample_pair(mode, skew_param)
        path = _route(sim, t1, t2)
        amt = sim.tx[i]

        flag, refused = True, False
        draws = []
        for j in range(len(path) - 1):
            u, v = path[j], path[j + 1]
            z0 = sim.get_within(u, v)[0]
            if z0 >= amt:
                continue
            zmax = amt - z0
            if headroom(u, v) >= zmax:          # only THIS channel's reserve
                draws.append((u, v, zmax))
            else:
                flag = False
                refused = True
                break

        if flag:
            for (u, v, zmax) in draws:
                outstanding[(u, v)] += zmax
                res.draw_events.append((u, zmax, outstanding[(u, v)]))
                ch = tuple_sort((u, v))
                res.channel_peak_short[ch] = max(res.channel_peak_short.get(ch, 0), zmax)
                z0, z1 = sim.get_within(u, v)
                sim.update_within(u, v, z0 + zmax, z1)
            for j in range(len(path) - 1):
                u, v = path[j], path[j + 1]
                z0, z1 = sim.get_within(u, v)
                sim.update_within(u, v, z0 - amt, z1 + amt)
                # v received amt on this channel -> retire v's draw on (v,u)
                if outstanding[(v, u)] > 0:
                    outstanding[(v, u)] = max(0, outstanding[(v, u)] - amt)
            res.n_success += 1
            res.success_volume += amt
            if draws:
                res.rebal_depths.append(len(draws))
        else:
            if refused:
                res.refusals += 1
        res.n_total += 1
        res.total_volume += amt
    return res


# --------------------------------------------------------------------------
# Revive -- batched cycle-based rebalancing via LP (port of revive.py)
# --------------------------------------------------------------------------
def work_revive(sim, tx_load, mode="uniform", skew_param=None,
                node_threshold=400, channel_threshold=1500):
    """Faithful port of github-shaduf/evaluation/revive.py onto the shared Sim.

    Revive batches failed payments and, when enough demand accumulates, solves
    a linear program (revive_linear.linear_proj) for a multi-party cyclic
    rebalancing that it then applies, retrying the suspended payments.

    Coordination depth (for the latency model): a payment rescued by a Revive
    round needed that round's multi-party negotiation, all participants online;
    we attribute it depth = its path length (a documented proxy for the number
    of cooperating parties), so Revive carries the p^k liveness penalty and,
    via its on-chain settlement (coordination.py chain_term), degrades steeply.
    """
    from revive_linear import linear_proj

    # directed balance view over sim.within: bal[(a,b)] = a's balance toward b
    bal = {}
    for (lo, hi), (blo, bhi) in sim.within.items():
        bal[(lo, hi)] = blo
        bal[(hi, lo)] = bhi
    G = sim.G

    req_node = set()
    req_passage = set()
    req_dict = {}

    def get_max_amount(path):
        return min(bal[(path[i], path[i + 1])] for i in range(len(path) - 1))

    def gather_demand(path, amt):
        for i in range(len(path) - 1):
            if bal[(path[i], path[i + 1])] < amt:
                req_node.add(path[i]); req_node.add(path[i + 1])
                req_passage.add((path[i], path[i + 1]))
                req_passage.add((path[i + 1], path[i]))

    def update_amount(path, amt):
        for i in range(len(path) - 1):
            bal[(path[i], path[i + 1])] -= amt
            bal[(path[i + 1], path[i])] += amt

    def transaction(src, dst, amt):
        path = _route(sim, src, dst)
        if get_max_amount(path) < amt:
            gather_demand(path, amt)
            return False, path
        update_amount(path, amt)
        return True, path

    def richness(a, b):
        x, y = bal[a[::-1]], bal[b[::-1]]
        return -1 if x < y else (1 if x > y else 0)

    def set_objective(path, amt):
        for i in range(len(path) - 1):
            tmp = bal[(path[i], path[i + 1])]
            if tmp >= amt:
                continue
            require = amt - tmp
            if bal[(path[i + 1], path[i])] < require:
                continue
            if (path[i], path[i + 1]) in req_dict or (path[i + 1], path[i]) in req_dict:
                continue
            req_dict[(path[i], path[i + 1])] = require
            cand = [(o, path[i]) for o in G[path[i]] if o != path[i + 1]]
            cand = [o for o in cand if o not in req_dict and o[::-1] not in req_dict]
            cand = sorted(cand, key=cmp_to_key(richness), reverse=True)
            if sum(bal[o[::-1]] for o in cand) < require:
                del req_dict[(path[i], path[i + 1])]
                continue
            for o in cand:
                req_dict[o] = min(bal[o[::-1]], require)
                require -= req_dict[o]
                if require <= 0:
                    break

    def confirm_demand(suspended):
        for src, dst, amt in suspended:
            set_objective(_route(sim, src, dst), amt)
        return [o + (req_dict[o],) for o in req_dict if req_dict[o] != 0]

    def adjust(project):
        for src, dst, amt in project:
            amt = int(amt + 0.5)
            bal[(src, dst)] += amt
            bal[(dst, src)] -= amt

    res = RunResult()
    suspended = []
    for i in range(tx_load):
        t1, t2 = sim.sample_pair(mode, skew_param)
        amt = sim.tx[i]
        ok, path = transaction(t1, t2, amt)
        if ok:
            res.n_success += 1
            res.success_volume += amt
        else:
            suspended.append((t1, t2, amt))
        res.n_total += 1
        res.total_volume += amt

        if (len(req_node) >= node_threshold or len(req_passage) >= 2 * channel_threshold
                or i == tx_load - 1):
            requirement = confirm_demand(suspended)
            project = linear_proj(requirement) if requirement else []
            if project:
                adjust(project)
            for (s, d, a) in suspended:
                ok, path = transaction(s, d, a)
                if ok:
                    res.n_success += 1
                    res.success_volume += a
                    res.rebal_depths.append(max(1, len(path) - 1))  # multi-party depth
            suspended = []
            req_node, req_passage, req_dict = set(), set(), {}

    return res
