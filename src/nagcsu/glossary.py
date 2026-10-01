"""Plain-language meanings for the column names and labels in nagcsu's tables.

Every table nagcsu prints or writes uses short headers (`NRMSE`, `Swing`,
`J span`). This module is the one place that says what each means and what
it should change about your next decision, so the terminal key and the
Markdown report's key never drift apart.
"""

import dataclasses
import typing


@dataclasses.dataclass(frozen=True, slots=True)
class Entry:
    """One glossary entry."""

    term: str
    meaning: str
    """What the column or label is."""

    impact: str
    """What it tells you to do, or why it matters."""


ENTRIES: tuple[Entry, ...] = (
    Entry(
        "J",
        "Combined mismatch score: the weighted sum of each scored vector's NRMSE. 0 is a perfect match.",
        "The number every search minimizes. Stop tuning once it is at or below the target.",
    ),
    Entry(
        "NRMSE",
        "Root-mean-square error of simulated against observed, divided by the observed range. 0.10 means a typical error of 10% of the history's range.",
        "Comparable across pressure, water cut and GOR. Above 1 means worse than the whole range of the history, usually a runaway GOR.",
    ),
    Entry(
        "Weight",
        "How much this vector counts toward J. Weights sum to 1.",
        "Raise a weight to make searches prioritize matching that vector.",
    ),
    Entry(
        "Weighted",
        "Weight times NRMSE: what this vector actually adds to J.",
        "Compare across rows to see which vector is costing you the most.",
    ),
    Entry(
        "Share of J",
        "This vector's weighted value as a fraction of J.",
        "Tune the parameters that drive the largest share first.",
    ),
    Entry(
        "Target",
        "The J value at or below which tuning should stop.",
        "Going far below it mostly fits the noise of a synthetic anchor, not physics.",
    ),
    Entry(
        "Default",
        "The parameter's value in the base deck, before any tuning.",
        "The reference for Change.",
    ),
    Entry(
        "Change",
        "How far the value moved from its default, in percent.",
        "A large move on a parameter the data barely constrains is a reason to be suspicious.",
    ),
    Entry(
        "Bounds",
        "The recommended search range for the parameter.",
        "Default searches stay inside it, but a `--range` may go past it, up to the hard physical limits.",
    ),
    Entry(
        "below/above registered bounds",
        "The value is outside the recommended range, which happens when a search range was deliberately extended.",
        "Fine if the data asked for it. Keep the extended `--range` for the next batch, or start from this run with `--baseline`.",
    ),
    Entry(
        "Baseline",
        "What a run started from: registered defaults, or a previous run's parameters (and deck).",
        "Chaining phases from the best run (`--baseline best`) builds on earlier progress instead of restarting from defaults.",
    ),
    Entry(
        "at low/high bound",
        "The tuned value sits at the edge of its bounds.",
        "The best value may lie outside; see `nagcsu report ranges`.",
    ),
    Entry(
        "Group",
        "A set of related parameters tuned together (aquifer, sgof_shape, ...).",
        "Groups are tuned one at a time in priority order.",
    ),
    Entry(
        "Start J / End J",
        "J before and after a group was tuned.",
        "End J below Start J is an improvement.",
    ),
    Entry(
        "Change (J)",
        "End J minus Start J for a group.",
        "Negative is an improvement. A group near zero is not worth more effort.",
    ),
    Entry(
        "Sims",
        "Simulations (OPM Flow runs) spent.",
        "The real cost, since each run takes wall-clock time.",
    ),
    Entry(
        "Passes",
        "How many times a group's parameters were revisited.",
        "Later passes search narrower windows around the best value so far.",
    ),
    Entry(
        "Pass",
        "Which pass over the group this row belongs to.",
        "Pass 1 searches the full range, later passes refine.",
    ),
    Entry(
        "Window",
        "The range searched for one parameter in one pass.",
        "Compare with Bounds: a narrow window means the search was refining, not exploring.",
    ),
    Entry(
        "Swing",
        "How much J changed when this parameter alone was moved from its low probe to its high probe.",
        "Bigger means more influential, in either direction. It does not say whether J got better, so read it together with Gain.",
    ),
    Entry(
        "J at low / high",
        "J when the parameter was set to its low probe value and to its high probe value.",
        "Compare each with the current J: lower is an improvement. FAILED means that probe did not simulate.",
    ),
    Entry(
        "Gain",
        "How much J dropped at the better of the two probes. Zero when neither probe beat the current J.",
        "The default ranking key. Zero means moving this parameter alone only made J worse, so tune it late.",
    ),
    Entry(
        "Best side",
        "Which probe, low or high, gave the gain.",
        "Search on that side of the current value first.",
    ),
    Entry(
        "Gap closed",
        "The share of the distance from the current J to the target that this gain alone would close.",
        "Shows whether one parameter can get you to the target or several are needed.",
    ),
    Entry(
        "Total gain",
        "Sum of a group's parameter gains, in J units.",
        "How far J can drop if each parameter in the group moves one step the right way.",
    ),
    Entry(
        "Gain share",
        "A group's total gain as a fraction of all groups' gain.",
        "Where the room to improve J actually is.",
    ),
    Entry(
        "pressure / watercut / gor",
        "In the detailed sensitivity table, how much that vector's own NRMSE moved between the two probes (not weighted).",
        "Shows which vector a parameter actually controls, for example an aquifer parameter that mostly moves pressure.",
    ),
    Entry(
        "~ one probe failed",
        "One of the two probes did not simulate, so the swing is estimated from the other.",
        "Treat the rank as approximate and consider narrowing that parameter's range.",
    ),
    Entry(
        "FAILED",
        "The simulation at that value failed.",
        "Part of the parameter's range is unsafe; a narrower range avoids wasted runs.",
    ),
    Entry(
        "Rank",
        "Position by swing, 1 being the most sensitive. Ties share an average rank.",
        "Low rank means tune it early.",
    ),
    Entry(
        "Low / High value",
        "The two values tried around the base value, a fraction of the range each side.",
        "Shows how far the parameter was pushed to get its swing.",
    ),
    Entry(
        "Mean rank",
        "Average rank of a group's parameters.",
        "Lower is more sensitive, and it is fair between groups of different sizes.",
    ),
    Entry(
        "Rank sum",
        "Total of a group's parameter ranks.",
        "Shown for reference. It favors groups with few parameters.",
    ),
    Entry(
        "Best rank",
        "Rank of the group's single most sensitive parameter.",
        "A good best rank with a poor mean rank means one strong parameter among dead ones.",
    ),
    Entry(
        "Total swing",
        "Sum of a group's parameter swings, in J units.",
        "The absolute size of what the group can move.",
    ),
    Entry(
        "Share",
        "A group's total swing as a fraction of all groups' swing.",
        "Where the leverage on J actually is.",
    ),
    Entry(
        "Strategy",
        "Which algorithm logged the run (sweep, random, coordinate_descent, sensitivity).",
        "Filter with `report list --stage` or `clean --strategy`.",
    ),
    Entry(
        "Stage",
        "The step within the strategy: baseline, descent/passN, sensitivity/low, sensitivity/high, sweep, random, final.",
        "Tells you whether a run was a probe or a candidate answer.",
    ),
    Entry(
        "Parameter(s) / Value(s)",
        "What the run deliberately changed and the values it used.",
        "Only single-parameter runs can be credited to that parameter.",
    ),
    Entry(
        "Health",
        "The result of checking the run's .PRT log: clean, or attention.",
        "Attention means errors, warnings or convergence trouble, so distrust that run's J.",
    ),
    Entry("*", "The lowest J in the current view.", "The run to freeze or continue from."),
    Entry(
        "Trials",
        "Single-parameter runs logged for that parameter or group.",
        "More trials means firmer conclusions.",
    ),
    Entry(
        "Failed",
        "Trials that failed to simulate or were not scored.",
        "Many failures point to an unsafe part of the range.",
    ),
    Entry("Best J", "Lowest J reached.", "How good it has gotten."),
    Entry(
        "Value at best",
        "The parameter value where Best J occurred.",
        "The natural center of the next range.",
    ),
    Entry(
        "J span",
        "Worst J minus best J across everything tried for the parameter.",
        "Near zero means the parameter does not matter, so leave it alone. Large means it still responds.",
    ),
    Entry(
        "Range tried",
        "Lowest and highest values tried for the parameter.",
        "Compare with Bounds to see what is unexplored.",
    ),
    Entry(
        "Last run", "Most recent run that changed the parameter.", "Where to look in the ledger."
    ),
    Entry(
        "Parameters touched",
        "Distinct parameters tried inside the group.",
        "Untouched ones are candidates for the next batch.",
    ),
    Entry(
        "Registered",
        "The parameter's registered bounds.",
        "The suggested range is always inside them.",
    ),
    Entry("Tried", "Lowest and highest values tried so far.", "Where the evidence comes from."),
    Entry(
        "bracketed",
        "Worse values were found on both sides of the best.",
        "The suggested range is well supported.",
    ),
    Entry(
        "open_low / open_high / open_both",
        "The best value is at the edge of what was tried.",
        "The optimum may lie beyond it, so the range extends past that edge.",
    ),
    Entry(
        "flat",
        "J barely moved across everything tried.",
        "Hold the parameter at its best value and leave it out.",
    ),
    Entry(
        "insufficient",
        "Too few distinct values tried.",
        "Sweep the parameter before trusting a range.",
    ),
    Entry(
        "Suggested range",
        "Values worth searching next, as LOW to HIGH.",
        "Pass it to `match auto --range NAME=LOW:HIGH`.",
    ),
    Entry(
        "Water cut NRMSE / GOR NRMSE",
        "NRMSE of one well's simulated water cut or GOR against its history.",
        "Finds the well that is off while the field total looks fine.",
    ),
    Entry(
        "In J",
        "Whether the well counts toward J (through the per-well weights).",
        "Wells marked no are diagnostics only.",
    ),
    Entry("n/a", "The value was not a finite number.", "Usually a failed or empty run."),
)

GLOSSARY: typing.Final[dict[str, Entry]] = {entry.term: entry for entry in ENTRIES}
"""Every entry, keyed by term."""

OBJECTIVE = ("J", "NRMSE", "Weight", "Weighted", "Share of J", "Target")
WELLS = ("Water cut NRMSE / GOR NRMSE", "In J")
STATE = (
    "Default",
    "Change",
    "Bounds",
    "at low/high bound",
    "below/above registered bounds",
    "Group",
)
TUNING_PATH = ("Start J / End J", "Change (J)", "Sims", "Passes", "Pass", "Window")
SENSITIVITY = (
    "Swing",
    "Gain",
    "Best side",
    "Gap closed",
    "Rank",
    "Low / High value",
    "J at low / high",
    "~ one probe failed",
    "FAILED",
)
DETAILED_SENSITIVITY = (
    "Swing",
    "Gain",
    "Best side",
    "Gap closed",
    "Rank",
    "pressure / watercut / gor",
    "~ one probe failed",
    "FAILED",
)
GROUP_RANKING = (
    "Mean rank",
    "Rank sum",
    "Best rank",
    "Total gain",
    "Gain share",
    "Total swing",
    "Share",
)
LEDGER = ("Strategy", "Stage", "Parameter(s) / Value(s)", "Health", "*")
PARAMETER_HISTORY = (
    "Trials",
    "Failed",
    "Best J",
    "Value at best",
    "J span",
    "Range tried",
    "Last run",
)
GROUP_HISTORY = ("Parameters touched",)
RANGES = (
    "Registered",
    "Tried",
    "bracketed",
    "open_low / open_high / open_both",
    "flat",
    "insufficient",
    "Suggested range",
)


def lookup(terms: typing.Iterable[str]) -> list[Entry]:
    """Return entries for `terms`, in order, dropping duplicates and unknown terms."""
    seen: set[str] = set()
    entries: list[Entry] = []
    for term in terms:
        if term in GLOSSARY and term not in seen:
            seen.add(term)
            entries.append(GLOSSARY[term])
    return entries


def render_markdown(terms: typing.Iterable[str]) -> list[str]:
    """Markdown lines for a "Key" section covering `terms`, or an empty list."""
    entries = lookup(terms)
    if not entries:
        return []
    lines = ["## Key", "", "| Term | What it is | What it tells you |", "| --- | --- | --- |"]
    lines.extend(f"| {entry.term} | {entry.meaning} | {entry.impact} |" for entry in entries)
    return lines
