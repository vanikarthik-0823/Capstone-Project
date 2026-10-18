# Instamart Customer Intelligence & Support Copilot

## Overview

A single repository containing the full Copilot, built in four connected Parts on synthetic, fictional data (all money in INR):

| Part | What it does | Code |
|---|---|---|
| A | Audits and cleans raw exports, stores them in SQLite, answers 7 SQL questions, pandas cross-check | `src/part_a.py`, `src/validate.py`, `sql/queries.sql` |
| B | RFM + K-means customer segments, PCA, lapse-prediction model | `src/part_b.py` |
| C | Support-ticket triage with privacy redaction, retries, validation, token budget, routing | `src/part_c.py` |
| D | TF-IDF policy RAG assistant and a tool-using refund agent (ReAct) with safe stopping and resume | `src/part_d.py` |

`generate_data.py` and `mock_llm.py` are the supplied files, unmodified. Every language-model call goes through `mock_llm.py` (offline: no API key, no network, no cost). Libraries: Python standard library, pandas, numpy, scikit-learn, joblib. All work is on branch `main`.

## Reproduce

Run from the repository root (Python 3.11+). The runs are deterministic (fixed seeds).

1. `python -m venv .venv`
2. `source .venv/bin/activate` (Windows: `.venv\Scripts\activate`)
3. `pip install -r requirements.txt`
4. `python generate_data.py` (last line: `generated: customers=1200 products=40 orders_raw=13084 tickets=164`)
5. `python src/part_a.py` (audit, cleaning, `data/clean/*.csv`, `data/instamart.db`, SQL results, pandas checks)
6. `python src/validate.py` (prints `ALL CHECKPOINTS PASSED`)
7. `python src/part_b.py` (writes `outputs/customer_segments.csv`, `outputs/lapse_model.joblib`, `outputs/retention_list.csv`)
8. `python src/part_c.py` (writes `outputs/triage_results.csv`, `outputs/triage_budget_demo.csv`)
9. `python src/part_d.py` (writes `outputs/agent_runs.jsonl`, `outputs/refund_ledger.jsonl`, `outputs/agent_demos.jsonl`, `state/`)

Part C reads `outputs/customer_segments.csv` from Part B, so run the steps in this order. Part D deletes and rebuilds its own outputs and `state/` at the start of every run.

---

## Part A: Data Foundation

### A.1 Audit of the raw data

Shapes: customers (1200, 5), products (40, 4), orders (13084, 9).

Missing values (count / %), as loaded by pandas:

| Table | Column | Missing | % |
|---|---|---|---|
| customers | age | 61 | 5.08 |
| customers | gender | 30 | 2.50 |
| customers | customer_id, city, signup_date | 0 | 0 |
| products | all four columns | 0 | 0 |
| orders | delivery_minutes | 2541 | 19.42 |
| orders | order_amount | 53 | 0.41 |
| orders | all other columns | 0 | 0 |

Note: pandas turns the text `N/A` into a missing value on load (53 rows) but keeps `error` as text (67 rows), so the real number of unusable amounts is 53 + 67 = 120 (see below). Data types: ids, city, gender, dates, status are text; quantity and discount_pct are integers; age and delivery_minutes are floats (they contain blanks); order_amount is text in the raw file because of `N/A` / `error`.

| Audit figure | Value |
|---|---|
| raw orders.csv rows | 13,084 |
| exact duplicate rows | 119 |
| rows with status pending | 656 |
| distinct order_id | 12,309 |
| non-numeric order_amount | 120 |
| negative delivery_minutes | 106 |
| blank delivery_minutes | 2,541 |
| customers: blank age | 61 |
| customers: age outside 10-100 | 6 |
| customers: distinct raw city spellings | 24 |
| customers: distinct non-blank gender spellings | 10 |
| customers: blank gender | 30 |

### A.2 Cleaning rules and what each changed

Customers:

| Rule | Effect |
|---|---|
| City: strip spaces + title case | 24 spellings become 6 cities: Pune 227, Delhi 210, Bengaluru 197, Chennai 191, Hyderabad 191, Mumbai 184 |
| Gender: map to male / female / unknown (blank = unknown) | 10 spellings become 3 values: female 610, male 560, unknown 30 |
| Age: values outside 10-100 set to missing, then filled with the city median (`groupby().transform("median")`) | 6 impossible ages set to missing; 67 ages imputed in total (61 blank + 6); 0 missing afterwards, all ages in 18-60 |
| signup_date to datetime | all 1200 rows |

Orders, applied in this order:

| # | Rule | Effect |
|---|---|---|
| 1 | `order_ts` to datetime | all rows |
| 2 | Remove exact duplicate rows | 119 rows removed |
| 3 | For each `order_id` keep the record with the latest `order_ts` (drops the earlier `pending` record) | 656 rows removed, leaving 12,309 rows (one per order) |
| 4 | `order_amount` to numeric (bad values become missing), then fill from `round(price_inr * quantity * (1 - discount_pct/100), 2)` | 112 amounts filled (of the 120 raw unusable entries, 8 sat on rows removed in rules 2-3) |
| 5 | Negative `delivery_minutes` set to missing; for completed orders fill missing values with the median of the order's product category among completed orders (`groupby().transform`); cancelled orders stay missing | 105 negatives set to missing (1 of the 106 was on a removed row); 774 completed-order values filled (these include the 105 negatives); 1,199 cancelled orders left missing on purpose, because a cancelled order was never delivered and imputing a time would invent data |

Result: 12,309 rows; 11,110 completed and 1,199 cancelled; no pending rows; 0 missing `order_amount`; 0 missing `delivery_minutes` among completed orders; completed revenue INR 2,998,668.80.

### A.3 Storage and validation

The clean tables are written to `data/clean/{customers,products,orders}.csv` and to SQLite `data/instamart.db` (tables `customers`, `products`, `orders`; `order_ts` stored as text `YYYY-MM-DD HH:MM:SS`). `python src/validate.py` asserts 13 checkpoints (row counts, status counts, missing values, revenue, the six city counts, gender counts, age range, table existence) and prints `ALL CHECKPOINTS PASSED`.

### A.4 SQL analysis (`sql/queries.sql`, executed with `pd.read_sql_query`)

| Query | Result |
|---|---|
| Q1 WHERE + HAVING (completed, amount > 500, customer total > 8000) | 30 customers |
| Q2 three-table join, revenue by city and category | 30 rows; revenues sum to INR 2,998,668.80; largest cell Bengaluru / Electronics INR 233,909.45 |
| Q3 scalar subquery, price above catalogue average | 11 products |
| Q4 LEFT JOIN, customers with no completed order | 25 customers |
| Q5 two CTEs, spend above the average customer spend | 550 customers; average INR 2,552.06 |
| Q6 RANK / DENSE_RANK / ROW_NUMBER per city | top customers: Bengaluru C0082 (12,856.20), Chennai C0930 (10,151.40), Delhi C0291 (13,742.60), Hyderabad C0743 (12,730.05), Mumbai C0693 (14,766.55), Pune C0426 (10,377.10); 8 rows have RANK different from ROW_NUMBER |
| Q7 daily revenue with running total | 183 daily rows; last cumulative value INR 2,998,668.80 |

Why ties make RANK differ from ROW_NUMBER: customers with the same total spend get the same RANK (and the next rank is skipped), whereas ROW_NUMBER always gives every row a unique number, so after a tie the two sequences drift apart.

### A.5 pandas cross-check and exploratory analysis

`pd.pivot_table(..., margins=True)` of completed revenue by city and category has a grand total of INR 2,998,668.80, equal to the SQL total.

| City | Sum (INR) | Mean (INR) | Count |
|---|---|---|---|
| Bengaluru | 523,643.40 | 276.47 | 1,894 |
| Chennai | 463,629.15 | 272.72 | 1,700 |
| Delhi | 502,829.80 | 243.62 | 2,064 |
| Hyderabad | 528,589.90 | 284.95 | 1,855 |
| Mumbai | 479,058.50 | 297.00 | 1,613 |
| Pune | 500,918.05 | 252.48 | 1,984 |

Completed `order_amount`: mean 269.91, median 135.00, skewness 4.61, Q1 64.80, Q3 297.00, IQR fences -283.50 and 645.30, 0 outliers below the lower fence and 897 above the upper fence.

Correlation matrix (completed orders):

| | order_amount | discount_pct | quantity | delivery_minutes |
|---|---|---|---|---|
| order_amount | 1.000 | -0.095 | 0.087 | -0.003 |
| discount_pct | -0.095 | 1.000 | -0.000 | -0.010 |
| quantity | 0.087 | -0.000 | 1.000 | -0.013 |
| delivery_minutes | -0.003 | -0.010 | -0.013 | 1.000 |

Four questions (under 200 words):

*What is this data about?* Six months (April-September 2025) of quick-commerce orders from 1,200 customers in six Indian cities. *Distributions and statistics?* Order values are strongly right-skewed: the mean (269.91) is about twice the median (135.00), with skewness 4.61, because most baskets are small groceries and a few are electronics. *Outliers or anomalies?* 897 completed orders lie above the 645.30 fence and none below; these are genuine high-value electronics orders rather than errors. The real anomalies were in the raw data (duplicates, pending rows, impossible ages, negative delivery times). *Relationships?* All correlations are weak. The largest in absolute value is order_amount with discount_pct (about -0.095), slightly above order_amount with quantity (+0.087). Correlation of this size does not establish causation; it only says bigger baskets tend to carry slightly smaller discounts.

---

## Part B: Customer Segments and Lapse Prediction

### B.1 RFM table

Snapshot 2025-10-01 00:00:00, completed orders only. 1,175 customers have at least one completed order.

| Measure | Value |
|---|---|
| recency (days) | median 19, min 0, max 180 |
| frequency | mean 9.46, max 33 |
| monetary (INR) | mean 2,552.06, max 14,766.55 |

### B.2 K-means (StandardScaler, `n_init=10`, `random_state=42`)

| K | Inertia | Silhouette |
|---|---|---|
| 2 | 1820.8 | 0.465 |
| 3 | 1268.2 | 0.445 |
| 4 | 743.1 | 0.526 |
| 5 | 617.4 | 0.480 |
| 6 | 503.2 | 0.482 |
| 7 | 408.5 | 0.416 |
| 8 | 366.5 | 0.412 |

Elbow: the largest drop is from K=3 to K=4 (1268 to 743); after K=4 each extra cluster removes less (about 125 units at K=5 and under 100 from K=7). K=4 also has the highest silhouette (0.526), and four segments are easy for the business to act on (big spenders, loyal regulars, occasional buyers, dormant customers). **Final K = 4.**

Why K=8 always has lower inertia than K=2 yet that cannot pick K: inertia is the sum of squared distances to the nearest centre, so adding centres can only reduce it (at K = number of customers it would be 0). A lower number is therefore guaranteed and says nothing about whether the extra clusters are real or useful, which is why we look for the elbow, the silhouette and interpretability instead of the minimum.

### B.3 Segment profile, names and PCA

Cluster numbers are only labels with no ordering (segment 3 is not "bigger" than segment 1).

| segment_id | Name | Size | Mean recency (days) | Mean frequency | Mean monetary (INR) | Recommended action |
|---|---|---|---|---|---|---|
| 0 | Occasional Buyers | 436 | 28.86 | 3.77 | 1,057.62 | Bundle offers and nudges to raise basket size and order frequency |
| 1 | **Premium Big Spenders** (is_premium) | 138 | 24.12 | 6.77 | 7,129.70 | Early access to electronics, priority support, protect from lapse |
| 2 | Loyal Regulars | 419 | 8.77 | 19.40 | 3,410.02 | Loyalty points and grocery subscriptions; keep delivery fast |
| 3 | Dormant / At-Risk | 182 | 109.04 | 2.20 | 685.99 | Win-back coupon and a reminder campaign |

Premium is the segment with the highest mean monetary value (segment 1). PCA on the scaled RFM matrix: explained_variance_ratio_ = 0.634 and 0.218, total 0.852. `outputs/customer_segments.csv` has columns customer_id, recency, frequency, monetary, segment_id, segment_name, is_premium.

### B.4 Lapse problem

Cutoff 2025-09-01. Cohort = customers with at least one completed order before the cutoff: **1,156 customers, 443 lapsed (38.32%)**. Label `lapsed` = 1 if no completed order in 2025-09-01 to 2025-09-30. Features (built only from orders dated before the cutoff plus the clean customer table): recency, frequency, monetary, average discount_pct, cancellation rate (cancelled / all final-status orders), average delivery_minutes, favourite product category, city, age, gender, tenure in days. Split: `train_test_split(test_size=0.2, stratify=y, random_state=42)` gives 924 train and 232 test rows.

**Baseline** (always predict not lapsed): test accuracy 61.6%, lapsed-class F1 0.00.

### B.5 Preprocessing without leakage

A scikit-learn `Pipeline` with a `ColumnTransformer` (StandardScaler on numeric columns; `OneHotEncoder(drop="first", handle_unknown="ignore")` on city, gender and favourite category, i.e. k-1 columns each) is fitted inside the pipeline on training rows only. The learned mean of `frequency` in the training scaler is 8.1147, while the full-cohort mean is 8.0285. They differ because the scaler has only seen the 924 training rows, not the 232 test rows that are part of the full cohort. There are 0 missing values in any feature column (checked by an assertion in the code).

**No encoder, scaler or resampler saw test rows; no data from 2025-09-01 onwards entered the features; and the segment labels from Part B task 3 were not used as features.**

### B.6 Models (5-fold CV, scoring F1 of the lapsed class)

| Model | Grid | Best parameters | Best CV F1 |
|---|---|---|---|
| Logistic regression | C in {0.01, 0.1, 1, 10, 100} | C = 100 | 0.651 |
| Random forest (100 trees, seed 42) | max_depth in {3, 5, 8} x min_samples_leaf in {5, 10, 20} (45 fits) | max_depth 8, min_samples_leaf 10 | 0.687 |

Test set (lapsed class; confusion matrix rows = true, columns = predicted: [[TN, FP], [FN, TP]]):

| Model | Confusion matrix | Precision | Recall | F1 |
|---|---|---|---|---|
| Baseline | [[143, 0], [89, 0]] | 0 | 0 | 0.00 |
| Logistic regression | [[111, 32], [31, 58]] | 0.644 | 0.652 | 0.648 |
| Random forest | [[111, 32], [32, 57]] | 0.640 | 0.640 | 0.640 |

**Selected model: random forest**, chosen on cross-validated F1 (0.687 against 0.651) before looking at the test set; its test F1 is 0.640, well above the 0.55 requirement, and the two models are essentially tied on the test set. Missing a customer who is about to lapse means losing all their future revenue with no chance to react, whereas wrongly targeting a loyal customer only costs a small coupon, so a model that catches more lapsing customers (recall) is worth some extra false alarms, and F1 balances the two. The forest also captures non-linear effects of frequency and recency without manual feature engineering. The logistic regression remains a good, more explainable alternative.

### B.7 Class-balance experiment

The training set was over-sampled only: all original rows kept, lapsed rows duplicated at random with replacement (`random_state=42`) until both classes have 570 rows. The best logistic regression was refit on that data and evaluated on the unchanged test set.

| Logistic regression | Precision | Recall | F1 | Confusion matrix |
|---|---|---|---|---|
| Original training data | 0.644 | 0.652 | 0.648 | [[111, 32], [31, 58]] |
| Balanced training data | 0.589 | 0.854 | 0.697 | [[90, 53], [13, 76]] |

Recall moved most (0.652 to 0.854), while precision fell (0.644 to 0.589): it did not rise. For the retention team this means catching many more would-be lapsers (76 of 89 instead of 58) at the price of more wrongly targeted loyal customers (53 instead of 32), which is a good trade when a coupon is cheap compared with a lost customer.

### B.8 Interpretation, persistence and retention list

Top five random-forest features (`feature_importances_`): monetary 0.278, frequency 0.277, recency 0.111, avg_delivery 0.062, avg_discount 0.053. Reading: how much and how often someone buys dominates; how long ago they last ordered matters next; delivery speed and discount reliance matter a little.

Logistic regression coefficients (standardised features, C = 100):

| Feature | Coefficient | Business reading |
|---|---|---|
| frequency | -1.697 | The more orders a customer has placed, the less likely they are to lapse |
| monetary | +0.283 | Given the same frequency, higher spend is slightly linked to lapsing (frequency and monetary are correlated, so read with care) |
| recency | +0.039 | A longer gap since the last order raises lapse risk a little, once frequency is accounted for |

The selected model was saved with `joblib.dump` to `outputs/lapse_model.joblib`; the reloaded model reproduces the original test predictions exactly (checked by an assertion).

Retention list: of the 232 test customers, 40 are in the Premium segment; the 10 with the highest predicted lapse probability are written to `outputs/retention_list.csv` (customer_id, lapse_probability, actually_lapsed). **1 of those 10 truly lapsed.** The Premium flag comes from the whole-window RFM, which includes September, the window that defines the lapse label, so this list is illustrative and not a leakage-free evaluation.

---

## Part C: Support Ticket Triage with Guardrails

Pipeline (`src/part_c.py`): tickets are processed in file order, read with `keep_default_na=False`.

### C.1 Input guardrails

Empty or whitespace-only text gets `skipped_empty`; `est_tokens(text) > 400` gets `flagged_long`; neither is ever sent to the model. Result: 4 empty tickets and 4 long tickets (each estimates to 734 tokens).

### C.2 Redaction

Applied in this order before any prompt: email to `[EMAIL]`, phone `\b[6-9]\d{9}\b` to `[PHONE]`, order id to `[ORDER_ID]`.

| Type | Substitutions |
|---|---|
| email | 55 |
| phone | 91 |
| order id | 132 |
| total | 278 |

Re-applying the three patterns to the redacted texts finds 0 matches. The user message sent to the model is exactly the redacted text; the code asserts this and asserts that no pattern matches any prompt.

### C.3 Prompts

Zero-shot (about 24 estimated tokens) and few-shot (about 98): both contain the word CLASSIFY, the four categories and the JSON-only reply format; only the few-shot prompt has an `EXAMPLES:` line with four labelled examples, including a fraud example.

### C.4 Retries

Every call goes through a retry wrapper: maximum 3 retries, wait = `0.05 * 2**n * random.uniform(0.9, 1.15)`, one log line per retry (ticket id, retry number, wait), `mock_llm.reset()` before each full run. A full run over the 156 eligible tickets hit **19 TransientLLMErrors, all recovered** by the first retry (retry number 0; waits between 0.0451 and 0.0558 s, inside the 0.045-0.0575 s range); no ticket ended with a permanent error.

### C.5 Output validation

JSON with an allowed category gives `classified_json`; otherwise the lowercase words (`re.findall(r"[a-z]+", ...)`) are checked and exactly one allowed category word gives `classified_keyword`; everything else (including JSON with category `shipping`) gives `flagged_invalid_output` and goes to human review. Full run over 156 eligible tickets: **135 classified_json, 12 classified_keyword, 9 flagged_invalid_output.**

### C.6 Zero-shot versus few-shot (no budget)

| Run | Classified | Accuracy | Total tokens | Tokens per classified ticket |
|---|---|---|---|---|
| Zero-shot | 147 | 85.7% (126/147) | 8,152 | 55.5 |
| Few-shot | 147 | 100.0% (147/147) | 19,696 | 134.0 |

Zero-shot confusion table (rows = true, columns = predicted):

| true \ predicted | billing | delivery | fraud | technical |
|---|---|---|---|---|
| billing | 38 | 0 | 0 | 0 |
| delivery | 0 | 36 | 0 | 0 |
| fraud | 11 | 0 | 23 | 0 |
| technical | 10 | 0 | 0 | 29 |

When is few-shot worth it? It costs about 78 extra tokens per classified ticket (134.0 against 55.5, roughly 2.4 times the zero-shot cost) but lifts accuracy from 85.7% to 100%. All 21 zero-shot errors were fraud tickets (11) or technical tickets (10) wrongly sent to billing; a fraud ticket in the billing queue misses the 2-hour security deadline, so the extra tokens are clearly worth paying for here, whereas for a low-stakes classification where 85% is good enough the cheaper prompt would be the better choice.

### C.7 Budget, routing and outputs

Budget demo (zero-shot, `TOKEN_BUDGET = 6000`, saved as `outputs/triage_budget_demo.csv`): the run used 6,012 tokens and **41 tickets ended `budget_exhausted`** (101 classified_json, 8 classified_keyword, 6 flagged_invalid_output, 4 skipped_empty, 4 flagged_long).

Final run (`outputs/triage_results.csv`, 164 rows, few-shot, `TOKEN_BUDGET = 100000`), routing from `outputs/customer_segments.csv` (a customer with no segment row counts as not premium; all 164 tickets' customers did have one):

| Route | Tickets | Priority |
|---|---|---|
| security_team (category fraud) | 34 | HIGH |
| premium_desk (premium customer, not fraud) | 11 | HIGH |
| support_queue | 102 | NORMAL |
| human_review | 17 | NORMAL |

Rule-validated by code (JSON or single-keyword classification): 147 tickets (135 + 12). Sent to humans: 17 tickets, namely 4 empty, 4 too long, 9 with invalid model output.

---

## Part D: Policy RAG Assistant and Refund Agent

### D.1 Chunking, retrieval and answering

Each policy file is split into 40-word chunks with a 10-word overlap (step 30), keeping the source file name. **15 chunks:** refund 4, delivery 3, escalation 3, payment 3, privacy 2. A `TfidfVectorizer()` is fitted on the chunk texts; `retrieve(query, k=3)` transforms the query with the same vectorizer and ranks chunks by cosine similarity (returns dictionaries with source, score, text). For `refund policy late delivery` the top three chunks all come from `refund_policy.txt`.

TF (term frequency) measures how often a word appears in a chunk; IDF (inverse document frequency) down-weights words that appear in many chunks, so rare, informative words count more. A limitation: TF-IDF matches words, not meaning, so a question phrased with different words than the policy (for example "money back" versus "refund") can retrieve the wrong chunk.

`answer(question)` retrieves three chunks and sends two messages to `mock_llm.chat`: a system message containing POLICY ASSISTANT (answer only from context, otherwise say the information could not be found) and a user message `CONTEXT:` + the chunks, each prefixed by `[<file>]` on its own line, + `QUESTION:` + the question. It returns the text and the sources used.

### D.2 Evaluation (`policy_eval.csv`: 12 answerable, 2 unanswerable)

| Setting | Chunks | Expected doc in top 3 (of 12) | Expected fact in answer (of 12) | Fallback on unanswerable (of 2) |
|---|---|---|---|---|
| 40 words, overlap 10 | 15 | 12 | 12 | 2 |
| 40 words, overlap 0 | 13 | 12 | 10 | 2 |

With zero overlap two questions break: "What delivery time makes a completed order eligible for a full refund?" (the phrase "more than 45 minutes" sits at the end of one chunk and the rest of the sentence at the start of the next, so the answer returned is "receives a full refund of the order amount." without the number) and "How soon is money returned when a payment fails but is deducted?" (the sentence that holds "48 hours" is cut in two at a chunk boundary, so the answer lacks it). Overlap keeps boundary sentences whole in at least one chunk.

### D.3 Tools

`TOOLS_SPEC` lists five tools (name, one-sentence description, JSON parameter spec) and `TOOL_MAP` maps each name to a Python function: `get_order` (reads `data/instamart.db`), `search_policy` (uses the retrieval above), `calculate_refund` (completed and more than 45 minutes: 100% of the amount, else 0), `issue_refund` and `escalate_to_human` (each append one JSON line with a timestamp and the tool name to `outputs/refund_ledger.jsonl`). The agent looks up the requested tool name in `TOOL_MAP` and calls it with the model's arguments.

### D.4 ReAct loop: the seven scenarios

`run_agent(user_message, scenario_id, max_iterations=6, token_limit=6000, time_limit_seconds=10)` follows the specified message format and returns status, final_answer, steps, llm_calls and tokens_used. One line per run is in `outputs/agent_runs.jsonl`.

| Scenario | Status | Refund (INR) | Model calls | Tokens used |
|---|---|---|---|---|
| S1 | REFUND | 128.25 | 5 | 952 |
| S2 | REFUND | 135.00 | 5 | 952 |
| S3 | ESCALATE | 1999.00 | 5 | 939 |
| S4 | DENIED | 0.00 | 4 | 649 |
| S5 | DENIED | 0.00 | 3 | 370 |
| S6 | NOT_FOUND | 0.00 | 2 | 94 |
| S7 | FALLBACK_MAX_ITERATIONS | 0.00 | 6 | 3,066 |

All refund amounts equal `expected_refund_inr` within 0.01 (asserted in the code). S7 finishes all six model calls under the default `token_limit=6000`.

### D.5 Stopping safely

Three limits are enforced: `max_iterations` (checked at the top of each iteration after the timeout check), `token_limit` (checked after each model call; the requested tool is not executed) and `time_limit_seconds` (`time.monotonic()`, elapsed >= limit checked at the top of each iteration). Every fallback returns the steps so far and a short human-handoff string. These two demonstrations are recorded in `outputs/agent_demos.jsonl`:

| Demonstration | Status | Detail |
|---|---|---|
| S1 with `token_limit=200` | FALLBACK_TOKEN_LIMIT | stopped after 3 model calls and 387 tokens; only `get_order` and `search_policy` had run |
| S1 with `time_limit_seconds=0` | FALLBACK_TIMEOUT | stopped before the first model call (0 calls, 0 tokens) |

S7 (the `[LOOP]` scenario) ends with FALLBACK_MAX_ITERATIONS.

### D.6 Crash and resume

After every tool observation the run writes `state/<run_id>.json` (run id, scenario id, full message list, tokens_used, step number, tools executed so far, plus the steps, model-call count and limits needed to continue). For S1 a simulated crash was raised right after the second tool execution (checkpoint already saved); `resume_agent("S1_crash_demo")` loaded the checkpoint and continued. Tools executed across the crash and the resume, in order, four executions and none repeated:

1. `get_order`
2. `search_policy`
3. `calculate_refund`
4. `issue_refund`

Final status REFUND with INR 128.25. The demo writes to `outputs/refund_ledger_demo.jsonl` so that the main ledger `outputs/refund_ledger.jsonl` holds only the three real actions (S1 and S2 issued, S3 escalated) and no duplicate S1 refund. The `state/` folder holds one checkpoint file per run that executed at least one tool: the 7 scenario runs, the token-limit demo and the crash-and-resume demo (9 files).

### D.7 ReAct versus plan-and-execute (under 250 words)

My ReAct agent asked the model for a decision after every observation: 5 model calls for S1, S2 and S3, 4 for S4, 3 for S5 and 2 for S6, with 952 tokens for a normal refund. Each call re-sends the growing message list, so tokens grow with every step (S7 reached 3,066 tokens in 6 calls). A plan-and-execute design would ask once for the whole step list and then execute it, so on the happy path it would need about 2 model calls (plan, final answer) and fewer tokens, because the history is not re-sent each time. This is an estimate: the supplied mock only implements ReAct.

The difference shows on unexpected observations. For NOT_FOUND (S6) or a cancelled order (S5) ReAct simply stops after 2 or 3 calls. A fixed plan would still contain "search policy, calculate refund, issue refund", which become wrong or harmful once the order is missing or cancelled, so it must detect the error and replan, which costs another model call and needs careful error handling. Branching also matters here: whether to issue or escalate depends on the amount, known only after calculation.

For this refund workflow I would choose ReAct. The path is short (at most 5 calls), it branches on tool results (status, minutes, INR 1000 threshold), and money is involved, so deciding each step from the latest observation is safer than following a pre-written plan. The extra tokens are small compared with the cost of a wrong refund. Plan-and-execute would pay off for long, predictable pipelines.

---

## Limitations and Next Steps

- All data is synthetic and the language model is a deterministic mock driven by keywords, so classification accuracy (85.7% / 100%), the 19 transient errors and the agent's behaviour demonstrate the engineering patterns, not real model quality.
- The Premium flag is a post-hoc whole-window label: it uses the whole April-September RFM, which includes the September window that defines the lapse label, so the retention list is illustrative and not a leakage-free evaluation. A cleaner design would compute segments only from orders before the cutoff.
- The lapse model uses one cutoff and one split of 1,156 customers; its test F1 (about 0.64) has wide uncertainty. Next steps: repeated cross-validation, probability calibration, choosing the decision threshold from campaign costs, and time-based validation.
- Over-sampling by duplication can overfit duplicated rows; class weights or SMOTE-style methods could be compared.
- TF-IDF retrieval matches words, not meaning; embeddings or a hybrid search would handle paraphrases. Chunking by sentence would avoid the boundary failures seen with zero overlap.
- The agent assumes every request is inside the 7-day refund window and does not check that the customer owns the order or that an order was not already refunded (an idempotency check on the ledger would prevent double refunds on a re-run).
- Imputed ages and delivery times are medians; this is simple and transparent but understates real variability.
