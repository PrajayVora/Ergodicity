"""
Toy flare ergodicity study (steps 2 & 3) -- VECTORIZED version.

Same model, same outputs as flare_ergodicity.py, but with no per-site or
per-repeat Python loops. All flares in all repeats are simulated at once as
NumPy arrays (rows = flares, columns = state periods). The only remaining loop
is over the ~50 experiment settings, which costs almost nothing.

QUESTION
    Is watching ONE flare for a long time (CMS) as good as glancing at MANY
    flares once (aircraft)?  If yes, the system is ergodic (MAES assumes this).

HOW THE SCRIPT ANSWERS IT
    1. Model flares (many at once) -> simulate_periods, rates_at
    2. Check the model is right    -> analytic_*, check_step2, integral_autocorr_time
    3. Build a basin of flares     -> make_population  (true average is known)
    4. Pretend to measure it       -> aircraft_estimates, cms_estimates
    5. Score each method vs truth  -> score, run_experiments, plot

FLARE MODEL (mirrors MAES Flare.json)
    States: OPERATING / MALFUNCTIONING / UNLIT, uniform durations.
    After each OPERATING period the next state is drawn per transition:
        MALFUNCTIONING w.p. pMalfunction, UNLIT w.p. pUnlit, else OPERATING.
    MALFUNCTIONING / UNLIT always return to OPERATING.
    Emission rate  E(t) = Q * (1 - DE[state(t)]).

KEY VECTORIZATION IDEA
    Every flare history is a chain of "cycles": one OPERATING period followed by
    an optional MALFUNCTIONING or UNLIT period (length 0 if none). Because each
    cycle's random draws don't depend on the previous cycle, we can draw ALL
    cycles for ALL flares in one shot as (n_flares x n_cycles) arrays, then use
    cumsum to get start times. No while-loop needed.

Run:  python flare_ergodicity_vec.py       -> writes CSVs + PNGs to ./out
"""
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OP, MALF, UNLIT = 0, 1, 2
STATE_NAMES = ["OPERATING", "MALFUNCTIONING", "UNLIT"]


# ----------------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------------
@dataclass
class FlareParams:
    """Inputs for ONE typical flare (one row of the MAES 'Flares' sheet).

    Durations in days, uniform[min, max]. Defaults: flare_7 durations from
    MEET2/Flare.xlsx with probabilities lowered from that file's stress-test
    values (0.75 / 0.25). Used as the 'base' site in make_population.
    """
    op_min: float = 10.0
    op_max: float = 20.0
    malf_min: float = 1.0
    malf_max: float = 3.0
    unlit_min: float = 2.0
    unlit_max: float = 6.0
    p_malf: float = 0.10
    p_unlit: float = 0.05
    de_op: float = 0.98
    de_malf: float = 0.50
    de_unlit: float = 0.0
    Q: float = 10.0          # inlet methane flow [kg/h]


def make_population(n_sites, het, rng, base=FlareParams()):
    """Build a basin of n_sites flares as a dict of NumPy arrays (one entry per site).

    Storing the basin as arrays ("struct of arrays") instead of a list of
    objects is what lets every later function work on many sites at once.

    het = 0  -> all sites identical to `base` (ergodic)
    het > 0  -> per-site p_unlit, p_malf ~ Beta (mean = base, concentration
                1/het^2) and Q ~ mean-preserving lognormal(sigma = het)
                (non-ergodic: some sites much worse than others)

    Returns
    -------
    dict: field name -> array of length n_sites
    """
    pop = {k: np.full(n_sites, float(v)) for k, v in asdict(base).items()}
    if het > 0:
        k = 1.0 / het**2
        pop["p_unlit"] = rng.beta(base.p_unlit * k, (1 - base.p_unlit) * k, n_sites)
        pm = rng.beta(base.p_malf * k, (1 - base.p_malf) * k, n_sites)
        pop["p_malf"] = np.minimum(pm, 0.95 - pop["p_unlit"])          # keep p_malf + p_unlit <= 0.95
        pop["Q"] = base.Q * rng.lognormal(-het**2 / 2, het, n_sites)
    return pop


def subset(pop, idx):
    """Pick rows `idx` (any shape, flattened) from every array in the population.

    Lets one site appear many times, e.g. the same basin sampled in 200 repeats.
    """
    idx = np.asarray(idx).ravel()
    return {k: v[idx] for k, v in pop.items()}


def de_matrix(pop):
    """Destruction efficiencies as an (n_sites x 3) matrix, columns = state code.

    de_matrix(pop)[i, UNLIT] is site i's unlit DE. Used to turn a matrix of
    states into a matrix of emission rates with one fancy-indexing step.
    """
    return np.column_stack([pop["de_op"], pop["de_malf"], pop["de_unlit"]])


# ----------------------------------------------------------------------------
# Analytic (exact) answers -- vectorized over sites
# ----------------------------------------------------------------------------
def analytic_time_fractions(pop):
    """Long-run fraction of time in each state, for every site at once.

    Semi-Markov result: the jump chain's stationary vector is proportional to
    [1, p_malf, p_unlit] (visits per OPERATING period); multiply by mean
    durations and normalise.

    Returns
    -------
    (n_sites x 3) array of time fractions
    """
    visits = np.column_stack([np.ones_like(pop["p_malf"]), pop["p_malf"], pop["p_unlit"]])
    mean_dur = np.column_stack([(pop["op_min"] + pop["op_max"]) / 2,
                                (pop["malf_min"] + pop["malf_max"]) / 2,
                                (pop["unlit_min"] + pop["unlit_max"]) / 2])
    w = visits * mean_dur
    return w / w.sum(axis=1, keepdims=True)


def analytic_mean_rate(pop):
    """Exact long-run mean emission rate [kg/h] of every site (array).

    mean_i = Q_i * sum_s frac_is * (1 - DE_is). The basin truth is the mean of this.
    """
    return pop["Q"] * np.sum(analytic_time_fractions(pop) * (1 - de_matrix(pop)), axis=1)


# ----------------------------------------------------------------------------
# Simulation -- all flares at once
# ----------------------------------------------------------------------------
def simulate_periods(pop, t_end_days, rng, burn_in_days=200.0):
    """Simulate state histories for every flare in `pop` simultaneously.

    Each flare = n_cyc cycles of [OPERATING period, extra period], where the
    extra period is MALFUNCTIONING (p_malf), UNLIT (p_unlit) or a zero-length
    OPERATING filler (no event). All draws are (n_flares x n_cyc) arrays.

    n_cyc is chosen so even the shortest possible cycles reach t_end_days, so
    every history is long enough (checked with an assert).

    The clock starts at -burn_in_days so that at t=0 each flare is in a
    random, typical state rather than always OPERATING.

    Returns
    -------
    starts : (n_flares x 2*n_cyc) start time [days] of each period
    states : (n_flares x 2*n_cyc) state code of each period
    """
    K = len(pop["Q"])
    n_cyc = int(np.ceil((t_end_days + burn_in_days) / pop["op_min"].min())) + 2
    col = lambda a: a[:, None]                       # broadcast a per-site value across cycles

    op_dur = rng.uniform(col(pop["op_min"]), col(pop["op_max"]), (K, n_cyc))
    u = rng.random((K, n_cyc))
    kind = np.where(u < col(pop["p_malf"]), MALF,
                    np.where(u < col(pop["p_malf"] + pop["p_unlit"]), UNLIT, OP))
    extra_dur = np.where(kind == MALF, rng.uniform(col(pop["malf_min"]), col(pop["malf_max"]), (K, n_cyc)),
                np.where(kind == UNLIT, rng.uniform(col(pop["unlit_min"]), col(pop["unlit_max"]), (K, n_cyc)), 0.0))

    # Interleave: [OP, extra, OP, extra, ...]
    durs = np.empty((K, 2 * n_cyc)); durs[:, 0::2] = op_dur; durs[:, 1::2] = extra_dur
    states = np.empty((K, 2 * n_cyc), dtype=np.int8); states[:, 0::2] = OP; states[:, 1::2] = kind

    ends = -burn_in_days + np.cumsum(durs, axis=1)
    starts = ends - durs
    assert ends[:, -1].min() >= t_end_days, "history too short; increase n_cyc"
    return starts, states


def rates_at(pop, starts, states, query_days):
    """Emission rate of every flare at its own query times, in one call.

    query_days is (n_flares x n_queries): row i holds the times to look at
    flare i (1 column = aircraft snapshot; hourly grid = CMS).

    Vectorization trick: np.searchsorted works on one sorted 1-D array, so we
    shift each row by a large offset (row_number * span) so rows don't
    overlap, flatten everything, and do a single searchsorted for all flares.

    Returns
    -------
    (n_flares x n_queries) emission rates [kg/h]
    """
    K, P = starts.shape
    t0 = starts[:, :1]
    span = (starts - t0).max() + np.ptp(query_days) + abs(query_days).max() + 1.0
    off = np.arange(K)[:, None] * span
    flat_starts = (starts - t0 + off).ravel()
    flat_q = (query_days - t0 + off).ravel()
    pos = np.searchsorted(flat_starts, flat_q, side="right") - 1
    rows = np.repeat(np.arange(K), query_days.shape[1])
    st = states.ravel()[pos].reshape(query_days.shape)        # state at each query
    de = de_matrix(pop)[rows.reshape(query_days.shape), st]   # DE at each query
    return pop["Q"][:, None] * (1 - de)


def check_step2(base: FlareParams, rng, years=200):
    """Sanity check: simulated vs analytic time fractions and mean rate for one flare.

    Also reports tau_days (memory time, see integral_autocorr_time) and the
    std of the hourly rate.

    Returns
    -------
    pd.DataFrame with columns: quantity, analytic, simulated
    """
    pop = make_population(1, 0.0, rng, base)
    T = 365 * years
    starts, states = simulate_periods(pop, T, rng)
    s = np.clip(starts[0], 0, T)
    dur = np.diff(np.append(s, T))
    dur = np.clip(dur, 0, None)
    sim_frac = np.bincount(states[0], weights=dur, minlength=3) / T
    hourly = rates_at(pop, starts, states, np.arange(0, T, 1 / 24)[None, :])[0]
    return pd.DataFrame({
        "quantity": [f"frac_{n}" for n in STATE_NAMES] + ["mean_rate_kg_h", "tau_days", "std_rate_kg_h"],
        "analytic": list(analytic_time_fractions(pop)[0]) + [analytic_mean_rate(pop)[0], np.nan, np.nan],
        "simulated": list(sim_frac) + [hourly.mean(), integral_autocorr_time(hourly, 1 / 24), hourly.std()],
    })


def integral_autocorr_time(x, dt_days, max_lag_days=200):
    """Memory time tau [days]: how long the series stays correlated with itself.

    A T-day continuous record is worth ~T/(2*tau) independent samples:
        Var(time-average over T) ~ sigma^2 * 2*tau / T
    Computed by summing the FFT autocorrelation up to its first zero crossing.
    """
    x = x - x.mean()
    n = len(x)
    f = np.fft.rfft(x, 2 * n)
    acf = np.fft.irfft(f * np.conj(f))[:n]
    acf /= acf[0]
    acf = acf[:int(max_lag_days / dt_days)]
    cut = np.argmax(acf < 0) if np.any(acf < 0) else len(acf)
    return dt_days * (0.5 + acf[1:cut].sum())


# ----------------------------------------------------------------------------
# Measurement estimators -- all repeats at once
# ----------------------------------------------------------------------------
def sample_sites(n_pop, n_pick, reps, rng):
    """Pick n_pick distinct sites for each of `reps` repeats in one shot.

    Sorting a (reps x n_pop) matrix of random numbers and keeping the first
    n_pick columns = sampling without replacement, independently per row.

    Returns
    -------
    (reps x n_pick) array of site indices
    """
    return rng.random((reps, n_pop)).argsort(axis=1)[:, :n_pick]


def aircraft_estimates(pop, N, reps, rng, horizon_days=365, mdl_kg_h=0.0):
    """All `reps` aircraft campaigns at once: N sites x 1 snapshot each.

    1. Pick N sites per repeat -> reps*N flares, simulated together.
    2. One random look per flare within the year.
    3. Below detection limit (mdl_kg_h) -> 0.
    4. Average over the N sites of each repeat.

    Returns
    -------
    (reps,) array of basin-average estimates [kg/h]
    """
    idx = sample_sites(len(pop["Q"]), N, reps, rng)
    sub = subset(pop, idx)
    starts, states = simulate_periods(sub, horizon_days, rng)
    q = rng.uniform(0, horizon_days, (len(sub["Q"]), 1))
    r = rates_at(sub, starts, states, q)[:, 0]
    r = np.where(r >= mdl_kg_h, r, 0.0)
    return r.reshape(reps, N).mean(axis=1)


def cms_estimates(pop, m, T_days, reps, rng, p_detect=0.6):
    """All `reps` CMS campaigns at once: m sites x hourly for T_days.

    1. Pick m sites per repeat -> reps*m flares, simulated together.
    2. Hourly rates for every flare: (reps*m x hours) matrix.
    3. Random mask keeps each hour w.p. p_detect (wind / non-detects).
    4. Site mean = mean of kept hours; basin estimate = mean over the m sites.

    Returns
    -------
    (reps,) array of basin-average estimates [kg/h]
    """
    idx = sample_sites(len(pop["Q"]), m, reps, rng)
    sub = subset(pop, idx)
    starts, states = simulate_periods(sub, T_days, rng)
    hrs = np.broadcast_to(np.arange(0, T_days, 1 / 24), (len(sub["Q"]), int(T_days * 24)))
    r = rates_at(sub, starts, states, hrs)
    obs = rng.random(r.shape) < p_detect
    n_obs = obs.sum(axis=1)
    site_mean = np.where(n_obs > 0, (r * obs).sum(axis=1) / np.maximum(n_obs, 1), 0.0)
    return site_mean.reshape(reps, m).mean(axis=1)


def score(estimates, truth):
    """Bias (% systematic offset) and RMSE (% typical single-campaign error)."""
    e = np.asarray(estimates)
    return {"bias_pct": 100 * (e.mean() - truth) / truth,
            "rmse_pct": 100 * np.sqrt(np.mean((e - truth) ** 2)) / truth}


# ----------------------------------------------------------------------------
# Experiment grid (pandas) and plotting
# ----------------------------------------------------------------------------
def build_grid():
    """All experiment settings as one pandas DataFrame (one row per setting).

    Replaces the old nested loops: het x N for aircraft, het x m x T for CMS.
    """
    hets = [0.0, 0.5, 1.0]
    air = pd.MultiIndex.from_product([hets, [10, 30, 100, 300, 1000]], names=["het", "sites"]).to_frame(index=False)
    air["method"], air["T_days"] = "aircraft", 0
    cms = pd.MultiIndex.from_product([hets, [1, 2, 5], [7, 30, 90, 365]], names=["het", "sites", "T_days"]).to_frame(index=False)
    cms["method"] = "cms"
    return pd.concat([air, cms], ignore_index=True)[["method", "het", "sites", "T_days"]]


def run_experiments(out, reps=200, n_pop=2000, seed=1):
    """Run every setting in the grid and score it against the basin truth.

    One basin per het level is built and cached; each grid row then runs all
    `reps` campaigns in a single vectorized call.

    Returns
    -------
    pd.DataFrame: method, het, sites, T_days, bias_pct, rmse_pct
    """
    rng = np.random.default_rng(seed)
    grid = build_grid()
    basins = {h: make_population(n_pop, h, rng) for h in grid.het.unique()}
    truths = {h: analytic_mean_rate(p).mean() for h, p in basins.items()}

    def run_row(row):
        pop = basins[row.het]
        if row.method == "aircraft":
            est = aircraft_estimates(pop, int(row.sites), reps, rng)
        else:
            est = cms_estimates(pop, int(row.sites), int(row.T_days), reps, rng)
        return pd.Series(score(est, truths[row.het]))

    df = pd.concat([grid, grid.apply(run_row, axis=1)], axis=1)
    df["truth_kg_h"] = df.het.map(truths)
    df.to_csv(out / "step3_results.csv", index=False)
    return df


def plot(df, out):
    """Save out/step3_rmse.png: aircraft error vs N (left), CMS error vs T (right).

    A CMS line that flattens means longer monitoring no longer helps: the
    signature of non-ergodic (different) sites.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for het, ls in zip(sorted(df.het.unique()), ["-", "--", ":"]):
        a = df[(df.method == "aircraft") & (df.het == het)]
        axes[0].plot(a.sites, a.rmse_pct, ls, marker="o", color="k", label=f"het={het}")
        for m, c in zip([1, 2, 5], ["tab:blue", "tab:orange", "tab:green"]):
            cm = df[(df.method == "cms") & (df.het == het) & (df.sites == m)]
            axes[1].plot(cm.T_days, cm.rmse_pct, ls, marker="o", color=c,
                         label=f"m={m}, het={het}" if het == 0 else None)
    axes[0].set(xscale="log", yscale="log", xlabel="Sites snapshotted (N)", ylabel="RMSE of basin mean (%)",
                title="Aircraft: N sites x 1 snapshot")
    axes[1].set(xscale="log", xlabel="Monitoring duration T (days)",
                title="CMS: m sites x T days\n(solid het=0, dashed 0.5, dotted 1.0)")
    axes[0].legend(); axes[1].legend(fontsize=8)
    for ax in axes: ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout(); fig.savefig(out / "step3_rmse.png", dpi=150)


if __name__ == "__main__":
    out = Path("out"); out.mkdir(exist_ok=True)
    chk = check_step2(FlareParams(), np.random.default_rng(0))
    chk.to_csv(out / "step2_check.csv", index=False)
    print(chk.to_string(index=False))
    df = run_experiments(out)
    plot(df, out)
    print(df.to_string(index=False))
