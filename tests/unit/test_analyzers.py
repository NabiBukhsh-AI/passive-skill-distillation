"""TASK-017 to TASK-022 acceptance tests.

Every expected value here was hand-computed in `tests/fixtures/corpora.py` before the
analyzers ran, which is the only way a fixture test means anything: a table read off the
implementation's own output tests that the code is consistent with itself.
"""

from __future__ import annotations

import pytest

from psd.analysis.contrast import win_loss_contrast
from psd.analysis.error_frequency import (
    DEFAULT_TAXONOMY,
    Taxonomy,
    detect_tool_error,
    error_frequency_report,
)
from psd.analysis.ngrams import action_ngrams
from psd.analysis.profiles import (
    ALFWORLD,
    TAU2_RETAIL,
    classify_email,
    get_profile,
    registered_domains,
)
from psd.analysis.registry import (
    assert_byte_stable,
    registered_analyzers,
    run_all,
    serialize_report,
)
from psd.analysis.stalls import (
    detect_stalls,
    normalize_observation,
    observation_fingerprint,
    stall_rate,
)
from psd.core.canonicalize import DEFAULT_PROFILE, canonicalize_action
from psd.core.models import Action
from tests.fixtures.corpora import (
    FABRICATED_EMAIL,
    REAL_EMAIL,
    as_corpus,
    cycle_trajectory,
    figure2_stall_trajectory,
    progressing_trajectory,
    retail_failure_corpus,
    step_counter_trajectory,
)

# ===========================================================================
# TASK-017: action canonicalization (ALG-002)
# ===========================================================================


def text(command: str) -> Action:
    return Action(kind="text_action", text=command)


def test_the_spec_example_canonicalizes_to_slot_form() -> None:
    """ALG-002 Step 2 gives this exact example."""
    assert canonicalize_action(text("cool tomato 1 with fridge 1"), ALFWORLD) == (
        "cool <obj> with <recep>"
    )


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("go to fridge 1", "go to <recep>"),
        ("take apple 2 from countertop 1", "take <obj> from <recep>"),
        ("put mug 1 in microwave 1", "put <obj> in <recep>"),
        ("open cabinet 3", "open <recep>"),
        ("look", "look"),
        ("inventory", "inventory"),
        ("heat egg 1 with microwave 1", "heat <obj> with <recep>"),
    ],
)
def test_alfworld_commands_canonicalize(command: str, expected: str) -> None:
    assert canonicalize_action(text(command), ALFWORLD) == expected


def test_literals_are_stripped_so_the_same_action_counts_as_the_same() -> None:
    """The point of canonicalization: `fridge 1` and `fridge 2` are one symbol."""
    assert canonicalize_action(text("go to fridge 1"), ALFWORLD) == canonicalize_action(
        text("go to fridge 2"), ALFWORLD
    )


def test_an_unknown_verb_keeps_its_head_token() -> None:
    """ALG-002 edge case. Informative, without pretending to have parsed it."""
    assert canonicalize_action(text("frobnicate the widget"), ALFWORLD) == "text:frobnicate"


def test_an_empty_action_is_noop() -> None:
    assert canonicalize_action(text(""), ALFWORLD) == "noop"
    assert canonicalize_action(Action(kind="noop"), ALFWORLD) == "noop"


def tool(email: str | None) -> Action:
    args = {} if email is None else {"email": email}
    return Action(kind="tool_call", name="find_user_id_by_email", arguments=args)


def test_a_fabricated_email_canonicalizes_differently_from_a_real_one() -> None:
    """TASK-017 acceptance, stated verbatim.

    This is why ALG-002 Step 1 has a value-class escape hatch at all. A pure type
    signature makes both `email:str` and erases the paper's headline retail failure.
    """
    real = canonicalize_action(tool(REAL_EMAIL), TAU2_RETAIL)
    fabricated = canonicalize_action(tool(FABRICATED_EMAIL), TAU2_RETAIL)
    assert real != fabricated
    assert real == "tool:find_user_id_by_email(email:email_present)"
    assert fabricated == "tool:find_user_id_by_email(email:placeholder_like)"


def test_a_missing_sensitive_argument_is_its_own_symbol() -> None:
    """Calling auth with no email is a third, distinct failure."""
    assert canonicalize_action(tool(None), TAU2_RETAIL) == (
        "tool:find_user_id_by_email(email:email_absent)"
    )


def test_without_a_profile_the_distinction_collapses() -> None:
    """Negative control, and the reason profiles must DECLARE sensitive arguments.

    Under the domain-agnostic default both calls are `email:str`. That is not a bug in
    the default; it is why ALG-002 requires the declaration.
    """
    assert canonicalize_action(tool(REAL_EMAIL), DEFAULT_PROFILE) == canonicalize_action(
        tool(FABRICATED_EMAIL), DEFAULT_PROFILE
    )


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (REAL_EMAIL, "email_present"),
        ("<EMAIL_1>", "email_present"),
        ("user@example.com", "placeholder_like"),
        ("customer@example.org", "placeholder_like"),
        ("test@test.com", "placeholder_like"),
        ("not-an-email", "placeholder_like"),
        (None, "email_absent"),
        ("", "email_absent"),
        ("   ", "email_absent"),
    ],
)
def test_email_classifier(value: str | None, expected: str) -> None:
    assert classify_email(value) == expected


def test_a_redaction_placeholder_counts_as_present() -> None:
    """The redactor only emits one when a real value was there.

    Treating `<EMAIL_1>` as placeholder-like would erase the very distinction the
    redactor preserved.
    """
    assert classify_email("<EMAIL_1>") == "email_present"


def test_tool_arguments_are_sorted_so_the_symbol_is_stable() -> None:
    forward = Action(kind="tool_call", name="f", arguments={"b": 1, "a": "x"})
    backward = Action(kind="tool_call", name="f", arguments={"a": "x", "b": 1})
    assert canonicalize_action(forward) == canonicalize_action(backward)
    assert canonicalize_action(forward) == "tool:f(a:str,b:int)"


def test_code_execution_emits_sorted_api_symbols() -> None:
    """ALG-002 Step 3."""
    code = "import pandas as pd\nwb = openpyxl.load_workbook(p)\ndf = pd.read_excel(p)"
    assert canonicalize_action(Action(kind="code_execution", text=code)) == (
        "code:openpyxl.load_workbook+pd.read_excel"
    )


def test_unparsable_code_falls_back() -> None:
    assert canonicalize_action(Action(kind="code_execution", text="def (:")) == "code:unparsed"


def test_code_symbols_are_capped() -> None:
    """ALG-002 Step 3 caps at K=8 so one huge cell cannot produce a unique symbol."""
    code = "\n".join(f"mod.fn{i}()" for i in range(20))
    assert canonicalize_action(Action(kind="code_execution", text=code)).count("+") == 7


def test_every_benchmark_domain_has_a_profile() -> None:
    for domain in ("alfworld", "tau2_retail", "tau2_telecom", "ssb_verified"):
        assert domain in registered_domains()
        assert get_profile(domain).domain == domain


def test_an_unknown_domain_falls_back_rather_than_raising() -> None:
    """FR-014 requires a domain-agnostic default set."""
    assert get_profile("a_domain_nobody_has_profiled").domain == "default"


# ===========================================================================
# TASK-018: error frequency (ALG-003)
# ===========================================================================


def test_retail_fixture_reproduces_the_papers_59_percent() -> None:
    """Hand-computed: 13 of 22 trajectories, exactly the paper's figure."""
    report = error_frequency_report(retail_failure_corpus(), profile=TAU2_RETAIL)
    row = next(r for r in report.rows if r.error_type == "fabricated_auth_argument")
    assert row.trajectories_with == 13
    assert row.trajectory_rate == pytest.approx(13 / 22)
    assert row.trajectory_rate == pytest.approx(0.59, abs=0.005)


def test_retail_fixture_reproduces_the_papers_94_percent_of_tool_errors() -> None:
    """Hand-computed: 17 of 18 tool errors.

    Restricted to a tool-error-only taxonomy, which is the population the paper's 94%
    figure is over. `share_of_all_errors` is computed against whatever taxonomy is in
    force, so the taxonomy defines the denominator and must be stated.
    """
    corpus = retail_failure_corpus()
    tool_only = Taxonomy(version="tool_errors/1.0", detectors=(("tool_error", detect_tool_error),))
    report = error_frequency_report(corpus, taxonomy=tool_only, profile=TAU2_RETAIL)

    assert report.total_error_events == 18
    not_found = next(r for r in report.rows if r.error_type == "not_found")
    assert not_found.occurrences == 17
    assert not_found.share_of_all_errors == pytest.approx(17 / 18)
    assert not_found.share_of_all_errors == pytest.approx(0.94, abs=0.005)


def test_both_rates_are_first_class_outputs() -> None:
    """ALG-003's implementation note: both numbers, always.

    They answer different questions. A failure can be widespread and trivial, or rare and
    dominant, and one number cannot tell you which.
    """
    report = error_frequency_report(retail_failure_corpus(), profile=TAU2_RETAIL)
    for row in report.rows:
        assert row.trajectory_rate is not None
        assert row.share_of_all_errors is not None


def test_rows_are_sorted_by_share_then_name() -> None:
    report = error_frequency_report(retail_failure_corpus(), profile=TAU2_RETAIL)
    keys = [(-r.share_of_all_errors, r.error_type) for r in report.rows]
    assert keys == sorted(keys)


def test_evidence_pointers_are_deterministic() -> None:
    """FR-012: evidence must reconstruct the supporting transcript span."""
    corpus = retail_failure_corpus()
    first = error_frequency_report(corpus, profile=TAU2_RETAIL)
    second = error_frequency_report(list(reversed(corpus)), profile=TAU2_RETAIL)
    assert first.rows[0].evidence == second.rows[0].evidence
    assert all(e.trajectory_id and e.step_index >= 0 for e in first.rows[0].evidence)


def test_co_occurrence_is_recorded_rather_than_first_match_wins() -> None:
    """ALG-003 edge case.

    In the retail fixture every fabricating step ALSO returns not_found, so the two types
    co-occur on 17 steps. Resolving first-match-wins would hide that and make the numbers
    depend on detector ordering.
    """
    report = error_frequency_report(retail_failure_corpus(), profile=TAU2_RETAIL)
    assert report.co_occurrence[("fabricated_auth_argument", "not_found")] == 17


def test_an_error_free_corpus_returns_an_empty_report_not_a_crash() -> None:
    """ALG-003 edge case."""
    report = error_frequency_report([progressing_trajectory()], profile=ALFWORLD)
    assert report.rows == ()
    assert report.total_error_events == 0


def test_fabrication_detector_uses_only_prior_observations() -> None:
    """An address supplied at turn 6 does not justify a call at turn 2."""
    report = error_frequency_report(retail_failure_corpus(), profile=TAU2_RETAIL)
    clean = [r for r in report.rows if r.error_type == "fabricated_auth_argument"]
    # The eight clean trajectories supply the address in the same step's observation and
    # must not be flagged.
    assert clean[0].trajectories_with == 13


# ===========================================================================
# TASK-019: n-grams (ALG-004)
# ===========================================================================


def test_the_max_per_trajectory_cap_stops_one_stall_dominating() -> None:
    """TASK-019 acceptance, and ALG-004's implementation note.

    The Figure 2 trajectory is 20 identical `look` actions. Uncapped it would contribute
    20 to the unigram count on its own.
    """
    report = action_ngrams(
        [figure2_stall_trajectory()],
        profile=ALFWORLD,
        n_values=(1,),
        min_count=1,
        max_per_trajectory=5,
    )
    look = next(r for r in report.rows if r.ngram == ("look",))
    assert look.count == 5, "one trajectory contributed more than the cap"


def test_without_the_cap_the_stall_would_dominate() -> None:
    """Negative control: proves the cap is doing the work, not the fixture."""
    report = action_ngrams(
        [figure2_stall_trajectory()],
        profile=ALFWORLD,
        n_values=(1,),
        min_count=1,
        max_per_trajectory=1000,
    )
    assert next(r for r in report.rows if r.ngram == ("look",)).count == 20


def test_lift_is_computed_from_document_frequency() -> None:
    """ALG-004 Step 4.

    A gram appearing 30 times in one winning episode is not evidence it causes winning,
    so lift uses the count of trajectories containing it, not raw occurrences.
    """
    corpus = retail_failure_corpus()
    report = action_ngrams(corpus, profile=TAU2_RETAIL, n_values=(1,), min_count=1)
    fabricated = next(
        r for r in report.rows if r.ngram == ("tool:find_user_id_by_email(email:placeholder_like)",)
    )
    # 13 losing trajectories contain it, 0 winning ones.
    assert fabricated.count_in_wins == 0
    assert fabricated.count_in_losses == 13
    assert fabricated.lift == pytest.approx(-13 / 14)


def test_episodes_shorter_than_n_are_skipped() -> None:
    """ALG-004 edge case."""
    report = action_ngrams(
        [progressing_trajectory()], profile=ALFWORLD, n_values=(50,), min_count=1
    )
    assert report.rows == ()


def test_min_count_prunes() -> None:
    corpus = retail_failure_corpus()
    assert all(
        r.count >= 10
        for r in action_ngrams(corpus, profile=TAU2_RETAIL, n_values=(1,), min_count=10).rows
    )


def test_ngram_rows_are_totally_ordered() -> None:
    rows = action_ngrams(retail_failure_corpus(), profile=TAU2_RETAIL, min_count=1).rows
    keys = [(-abs(r.lift), -r.count, r.ngram) for r in rows]
    assert keys == sorted(keys)


# ===========================================================================
# TASK-020: stalls (ALG-005)
# ===========================================================================


def test_the_figure2_pattern_is_detected() -> None:
    """TASK-020 acceptance: the paper's repeated-look pattern."""
    report = detect_stalls(figure2_stall_trajectory(), ALFWORLD)
    assert report.stalled
    assert len(report.stall_runs) == 1
    run = report.stall_runs[0]
    assert (run.kind, run.start, run.end, run.length) == ("repeat_action", 0, 19, 20)
    assert run.action == "look"


def test_a_step_counter_must_not_defeat_detection() -> None:
    """TASK-020's negative test, and ALG-005's stated failure condition.

    If the fingerprint does not normalize volatile fields, every observation differs,
    nothing compares equal, and recall silently drops to zero while the analyzer keeps
    reporting cleanly. This is the single most dangerous bug in ALG-005.
    """
    report = detect_stalls(step_counter_trajectory(), ALFWORLD)
    assert report.stalled, (
        "a step counter in the observation defeated stall detection; the fingerprint is "
        "not normalizing volatile fields and recall is now zero"
    )
    assert report.stall_runs[0].length == 20


def test_the_step_counter_fixture_would_defeat_a_naive_fingerprint() -> None:
    """Proves the fixture is actually adversarial rather than trivially equal."""
    trajectory = step_counter_trajectory()
    raw = {step.observation.text for step in trajectory.steps}
    assert len(raw) == 20, "the fixture observations are not actually distinct"
    normalized = {normalize_observation(t, ALFWORLD) for t in raw}
    assert len(normalized) == 1, "normalization failed to collapse them"


@pytest.mark.parametrize(
    "volatile",
    [
        "Step 3 of 40.",
        "Turn 7.",
        "2026-08-01T09:12:03Z",
        "[12/40]",
        "431 ms",
    ],
)
def test_volatile_fields_are_normalized_away(volatile: str) -> None:
    base = "You see a cabinet 1 and a fridge 1."
    assert observation_fingerprint(f"{volatile} {base}") == observation_fingerprint(base)


def test_a_progressing_episode_is_not_a_stall() -> None:
    """Negative control. Without it, a detector that always fires would pass."""
    assert not detect_stalls(progressing_trajectory(), ALFWORLD).stalled


def test_period_two_cycles_are_detected() -> None:
    """ALG-005 Step 3."""
    report = detect_stalls(cycle_trajectory(), ALFWORLD)
    assert report.stalled
    run = report.stall_runs[0]
    assert run.kind == "cycle"
    assert run.period == 2
    assert run.length == 8


def test_a_repeat_run_is_not_reported_as_a_cycle() -> None:
    """20 identical actions is a repeat-action stall, not a period-2 cycle.

    Both detectors match it; the merge must prefer the more specific description, or the
    paper's Figure 2 pattern gets reported under the wrong name.
    """
    assert {r.kind for r in detect_stalls(figure2_stall_trajectory(), ALFWORLD).stall_runs} == {
        "repeat_action"
    }


def test_a_run_shorter_than_m_min_is_not_a_stall() -> None:
    report = detect_stalls(figure2_stall_trajectory(steps=2), ALFWORLD, m_min=3)
    assert not report.stalled


def test_m_min_boundary_is_inclusive() -> None:
    assert detect_stalls(figure2_stall_trajectory(steps=3), ALFWORLD, m_min=3).stalled


def test_step_cap_hit_is_recorded_separately_from_detection() -> None:
    """ALG-005 edge case.

    A capped episode is strong evidence of a stall, but it is evidence, not a detection.
    Conflating them would inflate the reported stall rate.
    """
    report = detect_stalls(figure2_stall_trajectory(), ALFWORLD)
    assert report.step_cap_hit is True
    assert report.stalled is True

    clean = detect_stalls(progressing_trajectory(), ALFWORLD)
    assert clean.step_cap_hit is False
    assert clean.stalled is False


def test_stall_rate_is_per_trajectory_not_per_run() -> None:
    """Spec Section 5.8: the 28.7% / 5.3% statistic counts trajectories."""
    reports = [
        detect_stalls(figure2_stall_trajectory(), ALFWORLD),
        detect_stalls(progressing_trajectory(), ALFWORLD),
        detect_stalls(cycle_trajectory(), ALFWORLD),
        detect_stalls(progressing_trajectory(), ALFWORLD),
    ]
    assert stall_rate(reports) == pytest.approx(0.5)


def test_stall_rate_of_an_empty_list_is_zero() -> None:
    assert stall_rate([]) == 0.0


# ===========================================================================
# TASK-021: win/loss contrast (ALG-006)
# ===========================================================================


def test_unpaired_contrast_finds_the_fabrication_signal() -> None:
    corpus = retail_failure_corpus()
    report = win_loss_contrast(
        corpus,
        {
            "fabricates": lambda t: any(
                s.action.arguments and s.action.arguments.get("email") == FABRICATED_EMAIL
                for s in t.steps
            )
        },
    )
    row = report.rows[0]
    assert row.test_used == "fisher_exact"
    assert row.p_given_win == 0.0
    assert row.p_given_loss == pytest.approx(13 / 14)
    assert row.lift < 0
    assert row.p_value < 0.001


def test_a_corpus_with_no_losses_returns_a_flagged_report_not_an_exception() -> None:
    """ALG-006 edge case: every lift is undefined, and the corpus is unsuitable."""
    winners = [t for t in retail_failure_corpus() if t.outcome.success]
    report = win_loss_contrast(winners, {"anything": lambda t: True})
    assert report.status == "degenerate_no_contrast"
    assert report.rows == ()


def test_zero_variance_predicates_are_dropped() -> None:
    """ALG-006 edge case."""
    report = win_loss_contrast(retail_failure_corpus(), {"always": lambda t: True})
    assert report.rows == ()


def test_paired_contrast_uses_mcnemar() -> None:
    """ALG-006 Step 3: the same task, solved one way and not the other."""
    from tests.fixtures.corpora import trajectory as make

    paired = []
    for i in range(10):
        steps = [
            {
                "index": 0,
                "observation": {"kind": "env", "text": "hi", "meta": {}},
                "output": {"text": "", "reasoning_text": None, "reasoning_token_count": 0},
                "action": {"kind": "noop"},
                "result": None,
                "tokens": {
                    "input_total": 1,
                    "input_cached": 0,
                    "output_visible": 1,
                    "output_tool_args": 0,
                    "output_reasoning": 0,
                },
                "latency_ms": 1,
                "cost_usd": 0.0,
            }
        ]
        paired.append(
            make(
                f"trj_think_{i}",
                f"task_{i}",
                steps,
                mode="think",
                reward=1.0,
                success=True,
                error_types=[],
            )
        )
        paired.append(
            make(
                f"trj_nothink_{i}",
                f"task_{i}",
                steps,
                mode="no_think",
                reward=0.0,
                success=False,
                error_types=["fabricated_auth_argument"],
            )
        )

    report = win_loss_contrast(
        paired,
        {"fabricates": lambda t: "fabricated_auth_argument" in t.labels.error_types},
        paired=True,
    )
    assert report.paired
    row = report.rows[0]
    assert row.test_used == "mcnemar_exact"
    assert row.n_win == 0  # discordant b: present in think only
    assert row.n_loss == 10  # discordant c: present in no_think only
    assert row.p_value < 0.01


def test_paired_contrast_with_no_pairs_is_flagged() -> None:
    report = win_loss_contrast(retail_failure_corpus(), {"x": lambda t: True}, paired=True)
    assert report.status == "degenerate_no_pairs"


def test_predicates_that_survive_correction_are_not_the_only_ones_reported() -> None:
    """ALG-006 implementation note.

    With 35 to 50 tasks most predicates will not survive multiplicity correction. That is
    not a bug, and filtering to significance would hand the distiller an empty table.
    """
    corpus = retail_failure_corpus()
    report = win_loss_contrast(
        corpus,
        {
            "fabricates": lambda t: any(
                s.action.arguments and s.action.arguments.get("email") == FABRICATED_EMAIL
                for s in t.steps
            ),
            "long": lambda t: len(t.steps) > 1,
        },
    )
    assert len(report.rows) == 2
    assert any(not r.significant for r in report.rows), (
        "expected at least one non-significant row to still be reported"
    )


# ===========================================================================
# TASK-022: registry and byte stability
# ===========================================================================


def test_every_analyzer_is_registered() -> None:
    assert registered_analyzers() == [
        "action_ngrams",
        "error_frequency",
        "stalls",
        "win_loss_contrast",
    ]


def test_analyzers_are_byte_stable_across_runs() -> None:
    """FR-011 harness: run every analyzer twice, diff, require zero difference."""
    assert_byte_stable(as_corpus(retail_failure_corpus()))


def test_analyzers_are_byte_stable_on_the_alfworld_fixtures() -> None:
    corpus = as_corpus(
        [figure2_stall_trajectory(), step_counter_trajectory(), progressing_trajectory()],
        domain="alfworld",
    )
    assert_byte_stable(corpus)


def test_output_does_not_depend_on_input_ordering() -> None:
    """No set or dict iteration-order dependence anywhere in the call graph."""
    corpus = retail_failure_corpus()
    forward = run_all(as_corpus(corpus))
    backward = run_all(as_corpus(list(reversed(corpus))))
    for name in forward:
        assert serialize_report(forward[name]) == serialize_report(backward[name]), (
            f"{name} output changed when the input list was reversed"
        )


def test_the_stability_harness_would_catch_an_unstable_analyzer() -> None:
    """Negative control for the harness itself.

    Without this, `assert_byte_stable` passing proves only that it ran.
    """
    import psd.analysis.registry as registry

    class UnstableAnalyzer:
        name = "unstable"
        counter = 0

        def run(self, corpus, config):
            UnstableAnalyzer.counter += 1
            from psd.core.ports import AnalyzerReport

            return AnalyzerReport(
                analyzer=self.name,
                analyzer_lib_version="0.0.0",
                corpus_merkle_root=corpus.ref.merkle_root,
                rows=[{"call": UnstableAnalyzer.counter}],
            )

    original = registry.ANALYZERS
    registry.ANALYZERS = (UnstableAnalyzer(),)
    try:
        with pytest.raises(AssertionError, match="not byte-stable"):
            registry.assert_byte_stable(as_corpus(retail_failure_corpus()))
    finally:
        registry.ANALYZERS = original


def test_reports_carry_the_corpus_merkle_root() -> None:
    """Lineage: a report must state which corpus produced it."""
    corpus = as_corpus(retail_failure_corpus())
    for report in run_all(corpus).values():
        assert report.corpus_merkle_root == corpus.ref.merkle_root
        assert report.analyzer_lib_version


def test_the_default_taxonomy_names_its_version() -> None:
    assert DEFAULT_TAXONOMY.version == "taxonomy/1.0"
    assert "fabricated_auth_argument" in DEFAULT_TAXONOMY.names()
