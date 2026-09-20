# Case Study — Growth Analytics Engine

**Repository:** [growth-analytics-engine](https://github.com/Freddricklogan/growth-analytics-engine) · **Live demo:** [freddricklogan.github.io/growth-analytics-engine](https://freddricklogan.github.io/growth-analytics-engine/) · **Author:** Freddrick Logan

---

## 1. Who has this problem

Anyone who allocates a marketing budget from a dashboard: the director of a professional-education programme deciding whether to renew paid search, the enrolment team choosing whom to call, the finance partner asking what a student costs and returns. I have sat on the programme side of that table at Illinois Tech and in edtech consulting. Everywhere the pattern repeats: several numbers, each true in its own convention, read as if they measured the same thing.

## 2. The problem, as a scenario

A programme director reviews last quarter. The dashboard says webinars earned the most enrolments, so the budget goes to webinars; the outreach campaign lifted conversion, so it expands; lifetime value covers many months, so paid social looks affordable. Three questions unasked: earned by what convention — last touch rewards whichever channel the prospect was on when they decided; lifted compared with what — last year, with different prospects; observed how long — the oldest cohorts are four months old, and month twelve is a line someone extended.

## 3. What it costs to leave it alone

Spend moves toward channels that close rather than channels that cause; campaigns expand into segments where they do nothing; unit economics rest on a value no cohort has produced. I will not put a figure on it — the stream is synthetic and the mistake scales with the budget. The cost is a budget reallocated on accounting rather than evidence, discovered a year later once the cohorts have aged.

## 4. The approach, and the alternative I rejected

I built a typed Python package that computes the four figures on one event stream and labels what each one is. Four attribution models run side by side; the Markov model solves an absorbing chain for the conversion probability and its drop when each channel is removed. Because the generator's channel effects are known, the page prints each model's rank agreement with the truth. Retention and revenue appear only where a cohort has lived them. Uplift comes from a randomised treatment — the one design that supports a causal claim — with confidence intervals overall and per segment. Acquisition cost and lifetime value combine the Markov credits with the cohort curve at a stated horizon. CI runs the package and publishes the report; every number on the live page was computed by the run that published it.

The alternative I rejected was heavier attribution — Shapley values, a logistic model, a sequence model — presented as the answer. More machinery would not change what attribution is: a rule for dividing credit among touches. The honest improvement was a section that measures cause by a different method, and a page that says which is which.

## 5. What the code does today

Real: a seeded event generator with a stated causal structure; a validating journeys CSV loader; four attribution models with removal effects from a linear solve; Spearman rank agreement; cohort matrices with unobserved cells kept missing; two-proportion uplift with Wald intervals, per-segment effects and a Qini curve; CAC, LTV, LTV:CAC and payback per channel and blended; a report generator, a CLI and a Streamlit app. Strict-mode typed Python with tests; the report ships with a strict content-security policy and no inline script.

Simulated: the programme. Prospects, touches, segments, campaign and purchases are generated; channel spend is a constant. The page says so.

Worth knowing: the Markov chain is first-order; the uplift interval is a Wald approximation, adequate at these sample sizes; segments are pre-defined, so the Qini curve is one line per segment, not a per-user ranking; the CSV loader supports attribution only.

## 6. Evidence

Measured in CI and locally with the same commands: 28 tests passing across six files; 99% coverage of statements and branches; ruff, ruff format and `mypy --strict` clean across 14 source files; bandit clean; pip-audit with no known vulnerabilities; CodeQL for Python and JavaScript. The tests solve a four-journey Markov chain by hand, check every cell of a seven-purchase cohort table, verify the uplift interval against its closed form, and confirm the treatment is significant for the lapsed segment only. The shipped report (seed 42, 6,000 prospects, 18 months): 1,112 enrolments; rank agreement with the true channel effects +0.54 last touch, −0.31 first touch, −0.09 linear, −0.26 Markov; campaign effect +4.0 points (95% interval +2.1 to +6.0), carried by the lapsed segment at +11.8 points; blended LTV:CAC at month six 1.82, payback month three. Headless Chrome on the built report: zero console errors, no horizontal scroll at 1280 or 400 px.

## 7. What it would take to run this in production

The analytics carry over; the data does not. Production needs an event pipeline with consented identifiers, a join from touches to enrolment and payment records, actual spend by channel and period, and treatment assignment logged when it happens. The uplift section requires that randomisation occurred; without it the section should be removed, not reinterpreted. Weeks of data engineering where the CRM and payment system share an identifier, longer where they do not; a privacy review first.

## 8. Limits and next steps

One conversion event, monthly grain, first-order chains, pre-defined segments, a Wald interval, synthetic spend. Next: a higher-order Markov variant to test whether rank agreement improves, a per-user uplift model with a proper Qini and confidence band, bootstrap intervals on the cohort curves, and monthly spend for per-cohort CAC.

## 9. Who should look at this

**Hiring manager:** evidence that I implement marketing-analytics methods from first principles, test them against hand-solved cases, and label what each number can claim.
**Consulting client:** a template for reviewing any growth dashboard — which figures are conventions, measurements, or extrapolations.
**Engineer:** read `src/growth_engine/attribution.py` and `tests/test_attribution.py` for the absorbing-chain solve and its hand-checked case; `cohorts.py` for missing-aware averaging.
