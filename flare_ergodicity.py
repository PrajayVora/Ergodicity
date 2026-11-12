"""
Toy flare ergodicity study (steps 2 & 3).

QUESTION
    Is watching ONE flare for a long time (continuous monitoring, CMS) as good
    as glancing at MANY flares once (aircraft)?  If yes, the system is ergodic
    (MAES assumes this).  If no, some methods stay wrong no matter how long
    they measure.

HOW THE SCRIPT ANSWERS IT
    1. Model one flare             -> FlareParams, simulate_states, rate_at
    2. Check the model is right    -> analytic_*, check_step2, integral_autocorr_time
    3. Build a basin of flares     -> make_population  (true average is known)
    4. Pretend to measure it       -> aircraft_estimate, cms_estimate
    5. Score each method vs truth  -> score, run_experiments, plot

FLARE MODEL (mirrors MAES Flare.json)
    States: OPERATING / MALFUNCTIONING / UNLIT, each with a uniform duration.
    At the end of each OPERATING period the next state is drawn per transition:
        MALFUNCTIONING w.p. pMalfunction, UNLIT w.p. pUnlit, else OPERATING again.
    MALFUNCTIONING and UNLIT always return to OPERATING.
    Emission rate  E(t) = Q * (1 - DE[state(t)])   (constant inlet Q per site).

Run:  python flare_ergodicity.py            -> writes CSVs + PNGs to ./out
"""
from dataclasses import dataclass, replace
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Integer codes for the three flare states (used as array indices).
OP, MALF, UNLIT = 0, 1, 2
STATE_NAMES = ["OPERATING", "MALFUNCTIONING", "UNLIT"]


# ----------------------------------------------------------------------------
# Step 2: single-flare model
# ----------------------------------------------------------------------------
@dataclass
class FlareParams:
    """All inputs that define ONE flare (one row of the MAES 'Flares' sheet).

    Durations are in days and drawn uniformly between min and max each time
    the flare enters a state. Defaults use the flare_7 duration ranges from
    MEET2/Flare.xlsx; the probabilities are lowered from that file's
    stress-test values (0.75 / 0.25) to something more typical.

    Attributes
    ----------
    op_min, op_max       : OPERATING duration range [days]
    malf_min, malf_max   : MALFUNCTIONING duration range [days]
    unlit_min, unlit_max : UNLIT duration range [days]
    p_malf               : chance the next state after OPERATING is MALFUNCTIONING
    p_unlit              : chance the next state after OPERATING is UNLIT
    de_op, de_malf, de_unlit : destruction efficiency in each state (0-1)
    Q                    : methane flow entering the flare [kg/h]
    """
    # Durations (days), uniform[min, max] -- flare_7 in MEET2/Flare.xlsx
    op_min: float = 10.0
    op_max: float = 20.0
    malf_min: float = 1.0
    malf_max: float = 3.0
    unlit_min: float = 2.0
    unlit_max: float = 6.0
    # Per-transition probabilities (MEET2 test file uses 0.75/0.25 = stress test)
    p_malf: float = 0.10
    p_unlit: float = 0.05
    # Destruction efficiencies
    de_op: float = 0.98
    de_malf: float = 0.50
    de_unlit: float = 0.0
    # Inlet methane flow to flare (kg/h)
    Q: float = 10.0

    def de(self):
        """Return destruction efficiencies as an array indexed by state code.

        Example: de()[UNLIT] -> de_unlit. Lets us turn an array of states into
        an array of emission rates in one vectorised step (see rate_at).
        """
        return np.array([self.de_op, self.de_malf, self.de_unlit])

    def mean_dur(self):
        """Return the mean duration [days] of each state, indexed by state code.

        For a uniform[min, max] draw the mean is simply (min + max) / 2.
        Used only by the analytic (hand-calculation) check.
        """
        return np.array([(self.op_min + self.op_max) / 2,
                         (self.malf_min + self.malf_max) / 2,
                         (self.unlit_min + self.unlit_max) / 2])


def analytic_time_fractions(p: FlareParams):
    """Long-run fraction of time the flare spends in each state (exact formula).

    Logic: for every 1 OPERATING period there are, on average, p_malf
    MALFUNCTIONING periods and p_unlit UNLIT periods. Time in a state =
    (number of visits) x (mean duration). Divide by the total to get fractions.

        frac_s = visits_s * meanDur_s / sum_over_states(visits * meanDur)

    Returns
    -------
    np.ndarray of length 3: [frac_OPERATING, frac_MALFUNCTIONING, frac_UNLIT]
    """
    visits = np.array([1.0, p.p_malf, p.p_unlit])
    w = visits * p.mean_dur()
    return w / w.sum()


def analytic_mean_rate(p: FlareParams):
    """Exact long-run average emission rate of one flare [kg/h].

    mean = Q * sum_over_states( time_fraction_s * (1 - DE_s) )

    This is the "true answer" for a single site. Averaging it over all sites
    in a basin gives the truth that every measurement method is scored against.
    """
    return p.Q * np.sum(analytic_time_fractions(p) * (1 - p.de()))


def simulate_states(p: FlareParams, t_end_days, rng, burn_in_days=200.0):
    """Simulate one flare's state history from (−burn_in) to t_end_days.

    Instead of stepping hour by hour, we jump from one state change to the
    next (event-based), which is fast and exact:
        - OPERATING: stay for U(op_min, op_max) days, then roll a die to pick
          MALFUNCTIONING (p_malf), UNLIT (p_unlit) or OPERATING again.
        - MALFUNCTIONING / UNLIT: stay for their duration, then go back to
          OPERATING.

    The burn-in starts the clock before t=0 so that at t=0 the flare is in a
    random, "typical" state rather than always starting in OPERATING.

    Returns
    -------
    times  : array of state-change times [days]; times[i] is when state i begins
    states : array of state codes; state[i] holds on [times[i], times[i+1])
    """
    t, s = -burn_in_days, OP
    times, states = [], []
    while t < t_end_days:
        times.append(t); states.append(s)
        if s == OP:
            t += rng.uniform(p.op_min, p.op_max)
            u = rng.random()
            s = MALF if u < p.p_malf else (UNLIT if u < p.p_malf + p.p_unlit else OP)
        elif s == MALF:
            t += rng.uniform(p.malf_min, p.malf_max); s = OP
        else:
            t += rng.uniform(p.unlit_min, p.unlit_max); s = OP
    return np.array(times), np.array(states)


def rate_at(p: FlareParams, times, states, query_days):
    """Emission rate [kg/h] of the flare at any requested moment(s).

    For each query time, searchsorted finds which state period it falls in,
    then rate = Q * (1 - DE of that state).

    Parameters
    ----------
    times, states : output of simulate_states
    query_days    : array of times [days] to look at (one value = a snapshot,
                    an hourly grid = a continuous monitor)

    Returns
    -------
    np.ndarray of emission rates, same length as query_days
    """
    idx = np.searchsorted(times, query_days, side="right") - 1
    return p.Q * (1 - p.de()[states[idx]])


def check_step2(p: FlareParams, rng, years=200):
    """Sanity check: does the simulator agree with the hand calculation?

    Simulates one flare for a very long time (default 200 years) and compares
    the simulated time-in-state fractions and mean rate with the analytic ones.
    If they match, the simulator is behaving as designed.

    Also reports:
        tau_days      : how long the flare "remembers" its state (see
                        integral_autocorr_time). Sets how fast a CMS time
                        average settles down.
        std_rate_kg_h : how much the hourly rate swings around its mean.

    Returns
    -------
    pd.DataFrame with columns: quantity, analytic, simulated
    """
    T = 365 * years
    times, states = simulate_states(p, T, rng)
    # Clip state periods to [0, T] and add up time spent in each state.
    t_clip = np.clip(np.append(times, T), 0, T)
    dur = np.diff(t_clip)
    sim_frac = np.array([dur[states == k].sum() for k in range(3)]) / T
    hourly = rate_at(p, times, states, np.arange(0, T, 1 / 24))
    tau_days = integral_autocorr_time(hourly, dt_days=1 / 24)
    return pd.DataFrame({
        "quantity": [f"frac_{n}" for n in STATE_NAMES] + ["mean_rate_kg_h", "tau_days", "std_rate_kg_h"],
        "analytic": list(analytic_time_fractions(p)) + [analytic_mean_rate(p), np.nan, np.nan],
        "simulated": list(sim_frac) + [hourly.mean(), tau_days, hourly.std()],
    })


def integral_autocorr_time(x, dt_days, max_lag_days=200):
    """Estimate the 'memory' time tau [days] of a time series.

    Idea: if the flare is UNLIT now, it is likely still UNLIT an hour from now,
    so neighbouring hours are not independent. tau measures how long that
    dependence lasts. A T-day continuous record is worth roughly T / (2*tau)
    independent samples, so:

        Var(time-average over T) ~ sigma^2 * 2*tau / T

    Method: compute the autocorrelation function (via FFT for speed) and sum
    it up to the first lag where it drops below zero.

    Parameters
    ----------
    x            : evenly spaced series (here: hourly emission rates)
    dt_days      : spacing of x in days (1/24 for hourly)
    max_lag_days : longest lag to consider

    Returns
    -------
    tau in days
    """
    x = x - x.mean()
    n = len(x)
    f = np.fft.rfft(x, 2 * n)
    acf = np.fft.irfft(f * np.conj(f))[:n]
    acf /= acf[0]
    max_lag = int(max_lag_days / dt_days)
    acf = acf[:max_lag]
    cut = np.argmax(acf < 0) if np.any(acf < 0) else len(acf)
    return dt_days * (0.5 + acf[1:cut].sum())   # one-sided; Var(mean_T) ~ sigma^2 * 2*tau / T


# ----------------------------------------------------------------------------
# Population of sites (heterogeneity = the ergodicity-breaking knob)
# ----------------------------------------------------------------------------
def make_population(n_sites, het, rng, base=FlareParams()):
    """Build a basin of n_sites flares that differ from each other by 'het'.

    This is the key knob of the study:
        het = 0    -> every flare is a copy of `base` (identical sites, ergodic:
                      one site watched long enough looks like the whole basin)
        het > 0    -> each site gets its own p_unlit, p_malf and Q, so a few
                      sites can be much worse than others (non-ergodic)

    How sites are varied:
        p_unlit, p_malf ~ Beta distribution with mean = base value; larger het
                          -> wider spread (Beta concentration k = 1 / het^2)
        Q               ~ lognormal with sigma = het, mean kept equal to base.Q
    p_malf is capped so p_malf + p_unlit <= 0.95.

    Returns
    -------
    list of FlareParams, one per site
    """
    sites = []
    for _ in range(n_sites):
        if het == 0:
            sites.append(base); continue
        k = 1.0 / het**2  # Beta concentration
        pu = rng.beta(base.p_unlit * k, (1 - base.p_unlit) * k)
        pm = rng.beta(base.p_malf * k, (1 - base.p_malf) * k)
        if pu + pm > 0.95:
            pm = 0.95 - pu
        Q = base.Q * rng.lognormal(-het**2 / 2, het)   # mean-preserving
        sites.append(replace(base, p_unlit=pu, p_malf=pm, Q=Q))
    return sites


# ----------------------------------------------------------------------------
# Step 3: measurement estimators
# ----------------------------------------------------------------------------
def aircraft_estimate(pop, N, rng, horizon_days=365, mdl_kg_h=0.0):
    """One fake aircraft campaign: N sites, one instantaneous look at each.

    Steps:
        1. Pick N different sites at random from the basin.
        2. For each, simulate its history and read the emission rate at one
           random moment within the year.
        3. Readings below the detection limit (mdl_kg_h) are recorded as 0.
        4. Basin estimate = average of the N readings.

    Strength: sees many sites, so it samples site-to-site differences.
    Weakness: one instant per site, so it learns nothing about how often or
    how long events last.

    Returns
    -------
    float, estimated basin-average emission rate [kg/h]
    """
    idx = rng.choice(len(pop), size=N, replace=False)
    vals = []
    for i in idx:
        p = pop[i]
        t, s = simulate_states(p, horizon_days, rng)
        r = rate_at(p, t, s, np.array([rng.uniform(0, horizon_days)]))[0]
        vals.append(r if r >= mdl_kg_h else 0.0)
    return float(np.mean(vals))


def cms_estimate(pop, m, T_days, rng, p_detect=0.6):
    """One fake continuous-monitoring campaign: m sites, hourly for T_days.

    Steps:
        1. Pick m different sites at random from the basin.
        2. For each, simulate its history and read the rate every hour.
        3. Each hour is only 'seen' with probability p_detect, mimicking wind
           carrying the plume away from the sensor.
        4. Site estimate = average of seen hours; basin estimate = average of
           the m site estimates.

    Strength: sees events come and go, so it captures frequency and duration.
    Weakness: only m sites, so if sites differ (het > 0) they may not
    represent the basin, and monitoring longer cannot fix that.

    Returns
    -------
    float, estimated basin-average emission rate [kg/h]
    """
    idx = rng.choice(len(pop), size=m, replace=False)
    hrs = np.arange(0, T_days, 1 / 24)
    site_means = []
    for i in idx:
        p = pop[i]
        t, s = simulate_states(p, T_days, rng)
        r = rate_at(p, t, s, hrs)
        obs = rng.random(len(hrs)) < p_detect
        site_means.append(r[obs].mean() if obs.any() else 0.0)
    return float(np.mean(site_means))


def score(estimates, truth):
    """Turn many repeated estimates into two error numbers.

    bias_pct : how far the AVERAGE estimate is from the truth (%). Non-zero
               means the method is systematically high or low.
    rmse_pct : typical size of the error of a SINGLE campaign (%), combining
               bias and scatter. This is what the plots show.

    Returns
    -------
    dict with keys 'bias_pct' and 'rmse_pct'
    """
    e = np.asarray(estimates)
    return {"bias_pct": 100 * (e.mean() - truth) / truth,
            "rmse_pct": 100 * np.sqrt(np.mean((e - truth) ** 2)) / truth}


def run_experiments(out, reps=200, n_pop=2000, seed=1):
    """Main experiment loop: every method x every setting x every het level.

    For each heterogeneity level (0, 0.5, 1.0):
        1. Build a basin of n_pop sites and compute the exact truth
           (mean of analytic_mean_rate over all sites).
        2. Aircraft: for N in [10 .. 1000], run `reps` fake campaigns, score.
        3. CMS: for m in [1, 2, 5] sites and T in [7 .. 365] days, run `reps`
           fake campaigns, score.

    Parameters
    ----------
    out   : folder to write step3_results.csv
    reps  : repeats per setting (more = smoother curves, slower)
    n_pop : number of sites in the basin
    seed  : random seed for reproducibility

    Returns
    -------
    pd.DataFrame with columns: method, het, sites, T_days, bias_pct, rmse_pct
    """
    rng = np.random.default_rng(seed)
    rows = []
    for het in [0.0, 0.5, 1.0]:
        pop = make_population(n_pop, het, rng)
        truth = np.mean([analytic_mean_rate(p) for p in pop])   # basin-wide long-run mean
        for N in [10, 30, 100, 300, 1000]:
            est = [aircraft_estimate(pop, N, rng) for _ in range(reps)]
            rows.append({"method": "aircraft", "het": het, "sites": N, "T_days": 0, **score(est, truth)})
        for m in [1, 2, 5]:
            for T in [7, 30, 90, 365]:
                est = [cms_estimate(pop, m, T, rng) for _ in range(reps)]
                rows.append({"method": "cms", "het": het, "sites": m, "T_days": T, **score(est, truth)})
        print(f"het={het} done (truth={truth:.3f} kg/h)")
    df = pd.DataFrame(rows)
    df.to_csv(out / "step3_results.csv", index=False)
    return df


def plot(df, out):
    """Draw the results side by side and save out/step3_rmse.png.

    Left panel : aircraft error vs number of sites N (one line per het level).
    Right panel: CMS error vs monitoring days T (colour = number of sites m,
                 line style = het level: solid 0, dashed 0.5, dotted 1.0).

    How to read it: lines going down = more effort helps. A CMS line that goes
    flat means longer monitoring no longer helps, which is the signature of
    non-ergodic (different) sites.
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
    axes[1].set(xscale="log", xlabel="Monitoring duration T (days)", title="CMS: m sites x T days\n(solid het=0, dashed 0.5, dotted 1.0)")
    axes[0].legend(); axes[1].legend(fontsize=8)
    for ax in axes: ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout(); fig.savefig(out / "step3_rmse.png", dpi=150)


if __name__ == "__main__":
    # 1) Check the single-flare model against the hand calculation.
    # 2) Run all fake measurement campaigns and score them.
    # 3) Plot the error curves.
    out = Path("out"); out.mkdir(exist_ok=True)
    chk = check_step2(FlareParams(), np.random.default_rng(0))
    chk.to_csv(out / "step2_check.csv", index=False)
    print(chk.to_string(index=False))
    df = run_experiments(out)
    plot(df, out)
    print(df.to_string(index=False))
