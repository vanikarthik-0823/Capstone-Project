"""Part C: support-ticket triage with guardrails, redaction, retries, validation, token budget and routing."""
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import random
import re
import time
from collections import Counter

import pandas as pd

import mock_llm
from mock_llm import TransientLLMError, est_tokens

MAX_INPUT_TOKENS = 400
BASE_DELAY = 0.05
MAX_RETRIES = 3
ALLOWED = ["billing", "delivery", "technical", "fraud"]

PATTERNS = [("EMAIL", re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")),
            ("PHONE", re.compile(r"\b[6-9]\d{9}\b")),
            ("ORDER_ID", re.compile(r"ORD-\d{6}"))]

ZERO_SHOT = ('You CLASSIFY support tickets. Allowed categories: billing, delivery, technical, fraud. '
             'Reply with JSON only: {"category": "...", "reason": "..."}')
FEW_SHOT = ('You CLASSIFY support tickets. Allowed categories: billing, delivery, technical, fraud. '
            'Reply with JSON only: {"category": "...", "reason": "..."}\n'
            'EXAMPLES:\n'
            'Ticket: There is an unauthorized transaction on my card. -> {"category": "fraud", "reason": "unauthorized transaction"}\n'
            'Ticket: The app crashes when I open the cart. -> {"category": "technical", "reason": "app crash"}\n'
            'Ticket: I was charged twice for my order. -> {"category": "billing", "reason": "double charge"}\n'
            'Ticket: My rider has not arrived. -> {"category": "delivery", "reason": "late rider"}')
assert "CLASSIFY" in ZERO_SHOT and "EXAMPLES:" not in ZERO_SHOT and "EXAMPLES:" in FEW_SHOT

RETRY_LOG = []


def redact(text, counter=None):
    for name, pat in PATTERNS:
        text, n = pat.subn(f"[{name}]", text)
        if counter is not None:
            counter[name] += n
    return text


def call_with_retry(messages, ticket_id):
    """Up to MAX_RETRIES retries; wait = BASE_DELAY * 2**n * uniform(0.9, 1.15); one log line per retry."""
    for n in range(MAX_RETRIES + 1):
        try:
            return mock_llm.chat(messages)
        except TransientLLMError as e:
            if n == MAX_RETRIES:
                raise
            wait = BASE_DELAY * 2 ** n * random.uniform(0.9, 1.15)
            RETRY_LOG.append((ticket_id, n, wait))
            print(f"[retry] ticket={ticket_id} retry_number={n} wait={wait:.4f}s reason={e}")
            time.sleep(wait)


def validate(content):
    """Return (status, category). Never guesses."""
    try:
        obj = json.loads(content)
        if isinstance(obj, dict) and obj.get("category") in ALLOWED:
            return "classified_json", obj["category"]
        return "flagged_invalid_output", ""
    except (json.JSONDecodeError, TypeError):
        words = re.findall(r"[a-z]+", (content or "").lower())
        found = {w for w in words if w in ALLOWED}
        if len(found) == 1:
            return "classified_keyword", next(iter(found))
        return "flagged_invalid_output", ""


def run_pipeline(tickets, system_prompt, token_budget=None, label="run"):
    mock_llm.reset()
    RETRY_LOG.clear()
    total_tokens = 0
    sent_prompts = []
    out = []
    for t in tickets.itertuples():
        text = t.text
        rec = {"ticket_id": t.ticket_id, "customer_id": t.customer_id, "status": "", "category": "", "tokens_used": 0}
        if not text.strip():
            rec["status"] = "skipped_empty"
        elif est_tokens(text) > MAX_INPUT_TOKENS:
            rec["status"] = "flagged_long"
        elif token_budget is not None and total_tokens >= token_budget:
            rec["status"] = "budget_exhausted"
        else:
            clean = redact(text)
            messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": clean}]
            sent_prompts.append((messages[1]["content"], clean))
            resp = call_with_retry(messages, t.ticket_id)
            used = resp.usage.total_tokens
            total_tokens += used
            rec["tokens_used"] = used
            rec["status"], rec["category"] = validate(resp.choices[0].message.content)
        out.append(rec)
    df = pd.DataFrame(out)
    # guardrail assertions: prompts contain exactly the redacted text and no PII
    for user_msg, clean in sent_prompts:
        assert user_msg == clean
        for _, pat in PATTERNS:
            assert not pat.search(user_msg), "PII leaked into prompt"
    return df, total_tokens, len(RETRY_LOG)


def summarise(df, tickets, total_tokens, retries, label):
    m = df.merge(tickets[["ticket_id", "true_category"]], on="ticket_id")
    cls = m[m.status.isin(["classified_json", "classified_keyword"])]
    acc = (cls.category == cls.true_category).mean() * 100
    print(f"\n--- {label} ---")
    print("status counts:", dict(Counter(df.status)))
    print(f"classified {len(cls)} | correct {(cls.category == cls.true_category).sum()} | accuracy {acc:.1f}% | "
          f"total tokens {total_tokens} | tokens/classified ticket {total_tokens / len(cls):.1f} | "
          f"TransientLLMErrors recovered {retries}")
    return cls, acc


if __name__ == "__main__":
    tickets = pd.read_csv("data/raw/tickets.csv", keep_default_na=False)
    print("tickets:", len(tickets), "| empty:", int((tickets.text.str.strip() == "").sum()),
          "| long:", int((tickets.text.map(est_tokens) > MAX_INPUT_TOKENS).sum()),
          "| long est_tokens:", sorted(set(tickets.text.map(est_tokens)[tickets.text.map(est_tokens) > MAX_INPUT_TOKENS])))

    # ---- redaction audit over all 164 tickets
    cnt = Counter()
    red = tickets.text.map(lambda x: redact(x, cnt))
    left = sum(len(p.findall(x)) for x in red for _, p in PATTERNS)
    print("redactions:", dict(cnt), "total", sum(cnt.values()), "| left after re-applying patterns:", left)
    print("system prompt est tokens: zero-shot", est_tokens(ZERO_SHOT), "few-shot", est_tokens(FEW_SHOT))

    # ---- zero-shot vs few-shot, no budget
    df0, tok0, r0 = run_pipeline(tickets, ZERO_SHOT, label="zero")
    cls0, acc0 = summarise(df0, tickets, tok0, r0, "ZERO-SHOT (no budget)")
    print("retry waits: min %.4f max %.4f n=%d" % (min(w for _, _, w in RETRY_LOG), max(w for _, _, w in RETRY_LOG), len(RETRY_LOG)))
    print("retry numbers used:", sorted({n for _, n, _ in RETRY_LOG}))
    print("zero-shot confusion (rows=true, cols=predicted):")
    print(pd.crosstab(cls0.true_category, cls0.category).to_string())
    df1, tok1, r1 = run_pipeline(tickets, FEW_SHOT, label="few")
    cls1, acc1 = summarise(df1, tickets, tok1, r1, "FEW-SHOT (no budget)")

    # ---- budget demo: zero-shot, TOKEN_BUDGET=6000
    dfb, tokb, rb = run_pipeline(tickets, ZERO_SHOT, token_budget=6000)
    dfb = dfb.merge(tickets[["ticket_id"]], on="ticket_id")
    dfb["segment_name"] = ""
    dfb[["ticket_id", "customer_id", "status", "category", "tokens_used"]].to_csv("outputs/triage_budget_demo.csv", index=False)
    print("\n--- BUDGET DEMO (zero-shot, TOKEN_BUDGET=6000) ---")
    print("status counts:", dict(Counter(dfb.status)), "| tokens used:", tokb,
          "| budget_exhausted:", int((dfb.status == "budget_exhausted").sum()))

    # ---- final run: few-shot, budget 100000, routing
    dff, tokf, rf = run_pipeline(tickets, FEW_SHOT, token_budget=100000)
    seg = pd.read_csv("outputs/customer_segments.csv")
    dff = dff.merge(seg[["customer_id", "segment_name", "is_premium"]], on="customer_id", how="left")
    dff["is_premium"] = dff["is_premium"].fillna(False).astype(bool)
    HUMAN = {"skipped_empty", "flagged_long", "flagged_invalid_output", "budget_exhausted"}

    def route(r):
        if r.status in HUMAN:
            return pd.Series(["human_review", "NORMAL"])
        if r.category == "fraud":
            return pd.Series(["security_team", "HIGH"])
        if r.is_premium:
            return pd.Series(["premium_desk", "HIGH"])
        return pd.Series(["support_queue", "NORMAL"])

    dff[["route", "priority"]] = dff.apply(route, axis=1)
    dff["segment_name"] = dff["segment_name"].fillna("")
    dff[["ticket_id", "customer_id", "status", "category", "segment_name", "priority", "route", "tokens_used"]] \
        .to_csv("outputs/triage_results.csv", index=False)
    print("\n--- FINAL (few-shot, budget 100000) ---")
    print("rows:", len(dff), "| status:", dict(Counter(dff.status)))
    print("routes:", dict(Counter(dff.route)), "| priorities:", dict(Counter(dff.priority)))
    print("customers without segment row:", int((dff.segment_name == "").sum()))
