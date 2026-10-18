"""Part D: policy RAG assistant (TF-IDF) + refund agent (ReAct) with limits, checkpoints and resume."""
import glob
import json
import os
import sqlite3
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import mock_llm

POLICY_DIR = "data/policies"
DB_PATH = "data/instamart.db"
LEDGER_PATH = "outputs/refund_ledger.jsonl"
STATE_DIR = "state"


# ======================================================================= 1. chunking
def chunk_documents(size=40, overlap=10):
    chunks = []
    step = size - overlap
    for path in sorted(glob.glob(os.path.join(POLICY_DIR, "*.txt"))):
        words = open(path, encoding="utf-8").read().split()
        start = 0
        while True:
            end = start + size
            chunks.append({"source": os.path.basename(path), "text": " ".join(words[start:end])})
            if end >= len(words):
                break
            start += step
    return chunks


# ======================================================================= 2. vectorise + retrieve
class PolicyIndex:
    def __init__(self, size=40, overlap=10):
        self.chunks = chunk_documents(size, overlap)
        self.vec = TfidfVectorizer()
        self.matrix = self.vec.fit_transform([c["text"] for c in self.chunks])

    def retrieve(self, query, k=3):
        sims = cosine_similarity(self.vec.transform([query]), self.matrix)[0]
        order = sims.argsort()[::-1][:k]
        return [{"source": self.chunks[i]["source"], "score": float(sims[i]), "text": self.chunks[i]["text"]} for i in order]

    # ------------------------------------------------------------------- 3. augment + answer
    def answer(self, question):
        hits = self.retrieve(question, 3)
        context = "\n".join(f"[{h['source']}]\n{h['text']}" for h in hits)
        messages = [
            {"role": "system", "content": "You are a POLICY ASSISTANT. Answer only from the supplied context. "
                                          "If the answer is not in the context, say: I could not find that information in the policy documents."},
            {"role": "user", "content": f"CONTEXT:\n{context}\nQUESTION:\n{question}"},
        ]
        resp = mock_llm.chat(messages)
        return resp.choices[0].message.content, [h["source"] for h in hits]


INDEX = PolicyIndex(40, 10)


def retrieve(query, k=3):
    return INDEX.retrieve(query, k)


def answer(question):
    return INDEX.answer(question)


def evaluate(index, label):
    ev = pd.read_csv("data/raw/policy_eval.csv", keep_default_na=False)
    doc_ok = fact_ok = fb_ok = n_ans = n_un = 0
    failures = []
    for r in ev.itertuples():
        text, sources = index.answer(r.question)
        if r.expected_doc == "NONE":
            n_un += 1
            fb_ok += text.startswith("I could not find")
        else:
            n_ans += 1
            d = r.expected_doc in sources
            f = r.expected_fact in text
            doc_ok += d
            fact_ok += f
            if not (d and f):
                failures.append((r.question, r.expected_doc, d, f, text))
    print(f"[{label}] chunks={len(index.chunks)} | expected_doc in top-3: {doc_ok}/{n_ans} | "
          f"expected_fact in answer: {fact_ok}/{n_ans} | fallback on unanswerable: {fb_ok}/{n_un}")
    for q, doc, d, f, text in failures:
        print(f"   FAIL: {q!r} doc_ok={d} fact_ok={f}\n         answer: {text}")


# ======================================================================= 5. tools
def get_order(order_id):
    con = sqlite3.connect(DB_PATH)
    row = con.execute("SELECT order_id, status, order_amount, delivery_minutes FROM orders WHERE order_id = ?", (order_id,)).fetchone()
    con.close()
    if row is None:
        return {"error": "order not found"}
    return {"order_id": row[0], "status": row[1], "order_amount": row[2], "delivery_minutes": row[3]}


def search_policy(query):
    return {"results": [{"source": h["source"], "score": round(h["score"], 4), "text": h["text"]} for h in retrieve(query)]}


def calculate_refund(order_amount, delivery_minutes):
    if delivery_minutes is not None and delivery_minutes > 45:
        return {"refund_inr": round(float(order_amount), 2)}
    return {"refund_inr": 0}


def _ledger(tool, order_id, refund_inr):
    os.makedirs(os.path.dirname(LEDGER_PATH), exist_ok=True)
    with open(LEDGER_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps({"timestamp": datetime.now(timezone.utc).isoformat(), "tool": tool,
                            "order_id": order_id, "refund_inr": refund_inr}) + "\n")
    return {"refund_inr": refund_inr}


def issue_refund(order_id, refund_inr):
    return _ledger("issue_refund", order_id, refund_inr)


def escalate_to_human(order_id, refund_inr):
    return _ledger("escalate_to_human", order_id, refund_inr)


TOOLS_SPEC = [
    {"name": "get_order", "description": "Look up an order in the database by order id.",
     "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]}},
    {"name": "search_policy", "description": "Search the policy documents and return the most relevant chunks.",
     "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}},
    {"name": "calculate_refund", "description": "Compute the refund amount in INR from the order amount and delivery minutes.",
     "parameters": {"type": "object", "properties": {"order_amount": {"type": "number"}, "delivery_minutes": {"type": "number"}},
                    "required": ["order_amount", "delivery_minutes"]}},
    {"name": "issue_refund", "description": "Issue an automatic refund of at most INR 1000 and record it in the ledger.",
     "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}, "refund_inr": {"type": "number"}},
                    "required": ["order_id", "refund_inr"]}},
    {"name": "escalate_to_human", "description": "Send a refund above INR 1000 to a human support lead and record it.",
     "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}, "refund_inr": {"type": "number"}},
                    "required": ["order_id", "refund_inr"]}},
]
TOOL_MAP = {"get_order": get_order, "search_policy": search_policy, "calculate_refund": calculate_refund,
            "issue_refund": issue_refund, "escalate_to_human": escalate_to_human}

AGENT_SYSTEM = "You are a refund AGENT. Use the tools to decide the refund request, then give a final answer."


# ======================================================================= 6-8. ReAct loop, limits, checkpoints
class SimulatedCrash(Exception):
    pass


def _save_checkpoint(st):
    os.makedirs(STATE_DIR, exist_ok=True)
    with open(os.path.join(STATE_DIR, f"{st['run_id']}.json"), "w", encoding="utf-8") as f:
        json.dump(st, f, indent=1, default=str)


def _fallback(st, status):
    done = ", ".join(st["tools_executed"]) or "none"
    st["final_answer"] = (f"{status}: stopping safely and handing over to a human agent. Scenario {st['scenario_id']}: "
                          f"{st['step']} steps completed (tools: {done}), {st['llm_calls']} model calls, {st['tokens_used']} tokens used.")
    return _result(st, status)


def _result(st, status):
    return {"status": status, "final_answer": st["final_answer"], "steps": st["steps"],
            "llm_calls": st["llm_calls"], "tokens_used": st["tokens_used"], "run_id": st["run_id"]}


def _loop(st, crash_after_tool=None):
    start = time.monotonic()
    while True:
        if time.monotonic() - start >= st["time_limit_seconds"]:
            return _fallback(st, "FALLBACK_TIMEOUT")
        if st["llm_calls"] >= st["max_iterations"]:
            return _fallback(st, "FALLBACK_MAX_ITERATIONS")
        resp = mock_llm.chat(st["messages"], TOOLS_SPEC)
        st["llm_calls"] += 1
        st["tokens_used"] += resp.usage.total_tokens
        if st["tokens_used"] > st["token_limit"]:
            return _fallback(st, "FALLBACK_TOKEN_LIMIT")           # tool just requested is NOT executed
        msg = resp.choices[0].message
        if not msg.tool_calls:
            st["final_answer"] = msg.content
            return _result(st, msg.content.split(":")[0].strip())
        tc = msg.tool_calls[0]
        st["messages"].append({"role": "assistant", "content": None,
                               "tool_calls": [{"id": tc.id, "name": tc.name, "arguments": tc.arguments}]})
        obs = TOOL_MAP[tc.name](**tc.arguments)
        st["messages"].append({"role": "tool", "tool_call_id": tc.id, "name": tc.name, "content": json.dumps(obs)})
        st["steps"].append({"tool": tc.name, "arguments": tc.arguments, "observation": obs})
        st["tools_executed"].append(tc.name)
        st["step"] += 1
        _save_checkpoint(st)                                          # checkpoint AFTER every observation
        if crash_after_tool is not None and len(st["tools_executed"]) == crash_after_tool:
            raise SimulatedCrash(f"simulated crash after tool #{crash_after_tool}")


def run_agent(user_message, scenario_id, max_iterations=6, token_limit=6000, time_limit_seconds=10,
              run_id=None, crash_after_tool=None):
    st = {"run_id": run_id or scenario_id, "scenario_id": scenario_id, "user_message": user_message,
          "messages": [{"role": "system", "content": AGENT_SYSTEM}, {"role": "user", "content": user_message}],
          "tokens_used": 0, "step": 0, "tools_executed": [], "steps": [], "llm_calls": 0, "final_answer": None,
          "max_iterations": max_iterations, "token_limit": token_limit, "time_limit_seconds": time_limit_seconds}
    return _loop(st, crash_after_tool)


def resume_agent(run_id):
    st = json.load(open(os.path.join(STATE_DIR, f"{run_id}.json"), encoding="utf-8"))
    return _loop(st)


def refund_of(res):
    for s in res["steps"]:
        if s["tool"] in ("issue_refund", "escalate_to_human"):
            return s["observation"]["refund_inr"]
    return 0


if __name__ == "__main__":
    print("##### CHUNKS #####")
    ch = chunk_documents(40, 10)
    print("total", len(ch), pd.Series([c["source"] for c in ch]).value_counts().to_dict())
    print("top-3 sources for 'refund policy late delivery':", [h["source"] for h in retrieve("refund policy late delivery")])
    print("\n##### EVALUATION #####")
    evaluate(PolicyIndex(40, 10), "overlap 10")
    evaluate(PolicyIndex(40, 0), "overlap 0")

    # fresh outputs for reproducibility
    for p in ["outputs/agent_runs.jsonl", "outputs/refund_ledger.jsonl", "outputs/refund_ledger_demo.jsonl",
              "outputs/agent_demos.jsonl"]:
        if os.path.exists(p):
            os.remove(p)
    for p in glob.glob(f"{STATE_DIR}/*.json"):
        os.remove(p)

    print("\n##### AGENT: 7 SCENARIOS #####")
    sc = pd.read_csv("data/raw/agent_scenarios.csv", keep_default_na=False)
    rows = []
    with open("outputs/agent_runs.jsonl", "a", encoding="utf-8") as f:
        for r in sc.itertuples():
            res = run_agent(r.message, r.scenario_id)
            refund = refund_of(res)
            f.write(json.dumps({"scenario_id": r.scenario_id, "status": res["status"], "final_answer": res["final_answer"],
                                "llm_calls": res["llm_calls"], "tokens_used": res["tokens_used"], "refund_inr": refund,
                                "steps": res["steps"]}) + "\n")
            exp = r.expected_outcome
            ok_status = res["status"] == exp or (exp == "FALLBACK" and res["status"] == "FALLBACK_MAX_ITERATIONS")
            ok_amt = abs(refund - float(r.expected_refund_inr)) <= 0.01 if exp != "FALLBACK" else True
            print(f"{r.scenario_id} | {res['status']} | refund {refund:.2f} | model calls {res['llm_calls']} | "
                  f"tokens {res['tokens_used']} | expected {exp} {r.expected_refund_inr} | match {ok_status and ok_amt}")
            assert ok_status and ok_amt
            rows.append(res)

    print("\n##### STOP-SAFELY DEMOS #####")
    s1 = sc.iloc[0]
    demos = []
    d1 = run_agent(s1.message, "S1", token_limit=200, run_id="S1_token_limit_demo")
    print("S1 token_limit=200 ->", d1["status"], "| calls", d1["llm_calls"], "| tokens", d1["tokens_used"], "| steps", len(d1["steps"]))
    print("   handoff:", d1["final_answer"])
    d2 = run_agent(s1.message, "S1", time_limit_seconds=0, run_id="S1_timeout_demo")
    print("S1 time_limit_seconds=0 ->", d2["status"], "| calls", d2["llm_calls"])
    print("   handoff:", d2["final_answer"])
    assert d1["status"] == "FALLBACK_TOKEN_LIMIT" and d2["status"] == "FALLBACK_TIMEOUT"

    print("\n##### CRASH AND RESUME (S1) #####")
    LEDGER_PATH = "outputs/refund_ledger_demo.jsonl"      # keep the real ledger free of a duplicate S1 refund
    try:
        run_agent(s1.message, "S1", run_id="S1_crash_demo", crash_after_tool=2)
        raise AssertionError("crash expected")
    except SimulatedCrash as e:
        print("crash:", e)
    ck = json.load(open("state/S1_crash_demo.json"))
    print("checkpoint at crash: step", ck["step"], "tools", ck["tools_executed"], "tokens", ck["tokens_used"])
    res = resume_agent("S1_crash_demo")
    ck2 = json.load(open("state/S1_crash_demo.json"))
    print("after resume: status", res["status"], "| refund", refund_of(res), "| tools executed in order:", ck2["tools_executed"])
    assert ck2["tools_executed"] == ["get_order", "search_policy", "calculate_refund", "issue_refund"]
    assert res["status"] == "REFUND" and abs(refund_of(res) - 128.25) < 0.01
    with open("outputs/agent_demos.jsonl", "w") as f:
        for name, d in [("token_limit_200", d1), ("time_limit_0", d2), ("crash_resume", res)]:
            f.write(json.dumps({"demo": name, "status": d["status"], "final_answer": d["final_answer"],
                                "llm_calls": d["llm_calls"], "tokens_used": d["tokens_used"], "steps": d["steps"]}) + "\n")
    print("ledger lines:", sum(1 for _ in open("outputs/refund_ledger.jsonl")), "| state files:", len(os.listdir("state")))
