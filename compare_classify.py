"""PoC: mortgage document classification, Jev vs Claude Opus, via OpenRouter.

Both models classify the same made-up documents into one of CATEGORIES.
We record latency, tokens, cost and accuracy per call, then print a summary.

Usage:
    python compare_classify.py            # all docs
    python compare_classify.py --limit 4  # quick smoke test
"""
import argparse
import os
import statistics
import sys
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv

JEV_URL = "https://openrouter.ai/api/v1/systemone"
CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
JEV_MODEL = "typesafe/jev-1.14"
OPUS_MODEL = "anthropic/claude-opus-5.5"

CATEGORIES = {
    "loan_application": "Uniform residential loan application (Form 1003) or borrower's request for a mortgage",
    "pay_stub": "Employee pay stub or earnings statement showing wages for a pay period",
    "bank_statement": "Bank or brokerage account statement showing balances and transactions",
    "tax_return": "Personal income tax return or W-2 / 1099 tax form",
    "appraisal_report": "Property appraisal estimating the market value of the home",
    "title_report": "Title search / commitment listing ownership, liens and encumbrances",
    "closing_disclosure": "Closing Disclosure or Loan Estimate listing loan terms, rate and closing costs",
    "insurance_declaration": "Homeowner's insurance declarations page showing coverage and premium",
}

# (true_label, text) -- all invented.
DOCS = [
    ("loan_application", "UNIFORM RESIDENTIAL LOAN APPLICATION. Borrower: Maria Alvarez. Subject property: 14 Birch Lane, Dayton OH. Loan amount requested: $285,000. Purpose: purchase. Employer: Lakeside Dental. Monthly income $6,200. Declarations: no bankruptcies."),
    ("loan_application", "I, Tom Reyes, am applying for a 30-year fixed mortgage of $410,000 to buy a condo at 88 Harbor St. My co-borrower is my spouse Ana. We have lived at our current address for 4 years and have two dependents."),
    ("pay_stub", "ACME LOGISTICS INC. Earnings Statement. Employee: J. Patel. Pay period 03/01-03/15. Gross pay $2,884.62. Federal tax $412.10. FICA $220.67. 401k $144.23. Net pay $2,107.62. YTD gross $14,423.10."),
    ("pay_stub", "Pay date 06/30. Hourly rate $31.50, regular hours 80, overtime 6. Gross $2,835.00. State tax withheld $98.40. Health premium $75.00. Direct deposit to account ending 4471. Employer: Northwind Hospital."),
    ("bank_statement", "FIRST COMMUNITY BANK. Checking account ending 8820. Statement period 05/01-05/31. Beginning balance $12,440.55. Deposits $6,210.00. Withdrawals $4,982.31. Ending balance $13,668.24. Large deposit 05/12 $3,000 transfer from savings."),
    ("bank_statement", "Vanguard Brokerage Account Summary as of 12/31. Total account value $87,320.14. Holdings: index fund 410 shares, bond fund 220 shares. Dividends reinvested $612.40. No withdrawals this quarter."),
    ("tax_return", "Form 1040 U.S. Individual Income Tax Return, tax year 2025. Filing status: Married filing jointly. Wages $118,400. Adjusted gross income $112,050. Total tax $14,210. Refund $1,330."),
    ("tax_return", "Form W-2 Wage and Tax Statement 2025. Employer EIN 12-3456789. Employee: Linda Cho. Box 1 wages $74,500.00. Box 2 federal income tax withheld $9,120.00. Box 3 social security wages $74,500.00."),
    ("appraisal_report", "UNIFORM RESIDENTIAL APPRAISAL REPORT. Subject: 14 Birch Lane. Site 0.22 acres, 3 bed / 2 bath, 1,640 sq ft, built 1998. Comparable sales: 3 within 0.5 miles. Opinion of market value: $297,000 as of 04/02. Condition C3."),
    ("appraisal_report", "Based on the sales comparison approach and cost approach, the undersigned appraiser estimates the as-is value of the property at 402 Elm Court to be $525,000. Neighborhood trend: stable. Adjustments for garage and lot size are detailed on page 3."),
    ("title_report", "PRELIMINARY TITLE COMMITMENT. Property: 88 Harbor St Unit 5B. Vesting: current owner Harbor Holdings LLC. Exceptions: 2019 utility easement; open deed of trust in favor of Metro Savings $120,000; unpaid 2025 property taxes $1,840."),
    ("title_report", "Title search results show a chain of ownership from 1987. A judgment lien of $4,300 was recorded against the seller in 2023 and must be cleared before closing. No boundary disputes found."),
    ("closing_disclosure", "CLOSING DISCLOSURE. Loan term 30 years. Purpose Purchase. Product Fixed Rate. Loan amount $285,000. Interest rate 6.375%. Monthly principal & interest $1,778. Cash to close $41,260. Origination charges $2,850."),
    ("closing_disclosure", "LOAN ESTIMATE. Estimated closing costs $9,430. Rate lock expires 07/21. APR 6.52%. In 5 years you will have paid $104,200 in principal, interest, mortgage insurance and loan costs. Prepayment penalty: none."),
    ("insurance_declaration", "HOMEOWNERS POLICY DECLARATIONS. Insured: Maria Alvarez. Dwelling coverage A $320,000. Personal property $160,000. Liability $300,000. Deductible $2,500. Annual premium $1,740. Mortgagee clause: Metro Savings Bank ISAOA."),
    ("insurance_declaration", "Evidence of property insurance for 402 Elm Court. Policy period 08/01 to 08/01. Wind/hail deductible 1% of dwelling limit. Replacement cost endorsement included. Lender listed as additional insured."),
]


def classify_jev(client, text):
    body = {
        "model": JEV_MODEL,
        "state": text,
        "questions": {"doc_type": {
            "type": "choice",
            "instructions": "What type of mortgage document is this?",
            "criteria": CATEGORIES,
        }},
    }
    r = client.post(JEV_URL, json=body)
    r.raise_for_status()
    d = r.json()
    u = d.get("usage", {})
    return (d["answers"]["doc_type"]["choice"], u.get("input_tokens", 0),
            u.get("output_tokens", 0), u.get("cost"))


def classify_opus(client, text):
    options = "\n".join(f"- {k}: {v}" for k, v in CATEGORIES.items())
    prompt = (f"Classify this mortgage document into exactly one category.\n\n{options}\n\n"
              f"Document:\n{text}\n\nReply with only the category name.")
    r = client.post(CHAT_URL, json={
        "model": OPUS_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 30,
    })
    r.raise_for_status()
    d = r.json()
    u = d.get("usage", {})
    label = d["choices"][0]["message"]["content"].strip().strip("`'\" .").lower()
    return label, u.get("prompt_tokens", 0), u.get("completion_tokens", 0), u.get("cost")


def main() -> int:
    load_dotenv(Path(__file__).parent / ".env")
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        print("OPENROUTER_API_KEY not set", file=sys.stderr)
        return 1
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=len(DOCS))
    args = ap.parse_args()

    models = {"Jev": classify_jev, "Opus": classify_opus}
    stats = {m: {"lat": [], "in": 0, "out": 0, "cost": 0.0, "correct": 0, "n": 0} for m in models}

    with httpx.Client(headers={"Authorization": f"Bearer {key}"}, timeout=120) as client:
        print(f"{'#':>2} {'truth':<22} {'Jev':<22} {'s':>5}  {'Opus':<22} {'s':>5}")
        for i, (truth, text) in enumerate(DOCS[:args.limit], 1):
            row = {}
            for name, fn in models.items():
                t0 = time.perf_counter()
                try:
                    label, tin, tout, cost = fn(client, text)
                except Exception as e:  # PoC: record failure, keep going
                    label, tin, tout, cost = f"ERR {type(e).__name__}", 0, 0, 0
                dt = time.perf_counter() - t0
                s = stats[name]
                s["lat"].append(dt); s["in"] += tin; s["out"] += tout
                s["cost"] += cost or 0.0; s["n"] += 1; s["correct"] += label == truth
                row[name] = (label, dt)
            mark = lambda n: ("ok " if row[n][0] == truth else "XX ") + row[n][0]
            print(f"{i:>2} {truth:<22} {mark('Jev'):<22} {row['Jev'][1]:>5.2f}  "
                  f"{mark('Opus'):<22} {row['Opus'][1]:>5.2f}")

    print("\n=== Summary ===")
    print(f"{'':<6}{'acc':>8}{'avg s':>8}{'p50 s':>8}{'total s':>9}{'in tok':>9}{'out tok':>9}{'cost $':>11}")
    for name, s in stats.items():
        n = max(s["n"], 1)
        print(f"{name:<6}{s['correct']:>4}/{s['n']:<3}{statistics.mean(s['lat']):>8.2f}"
              f"{statistics.median(s['lat']):>8.2f}{sum(s['lat']):>9.1f}{s['in']:>9}{s['out']:>9}{s['cost']:>11.5f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
