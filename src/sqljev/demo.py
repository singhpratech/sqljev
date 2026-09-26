#!/usr/bin/env python3
"""Demo data: a 100,000-row, 10-domain set of realistic tables with exact ground-truth labels.

    python -m sqljev.demo --rows-per-table 10000 --out bench/bench.duckdb      # the benchmark (bench/run.py)
    from sqljev.demo import TASKS, generate                                    # in a notebook

Each table is generated from parts whose meaning is known, so labels are exact. Slots (names, amounts, dates,
products, codes) make rows distinct; a few percent are exact repeats, as in real tables. Every task includes
hard negatives (a neutral complaint is not angry, a hospital *visit* for tests is not a serious event, ...).

TASKS below is the benchmark: table, the columns the model may see (never the label), the plain-language
question, its kind and options, and the label column.
"""
import argparse
import random

TASKS = [
    # name, table, columns shown to the model, question, kind, options, label column
    ("ticket_angry", "tickets", ["subject", "body"], "the customer is angry", "noul", None, "angry"),
    ("ticket_team", "tickets", ["subject", "body"], "which team should handle this ticket?", "choice",
     ["billing", "technical", "sales", "security"], "team"),
    ("review_sentiment", "reviews", ["title", "text"], "how does the reviewer feel about the product?", "score",
     ["negative", "mixed", "positive"], "sentiment"),
    ("review_defect", "reviews", ["title", "text"], "the review reports a defective or broken product", "noul", None,
     "defect"),
    ("ae_serious", "adverse_events", ["drug", "narrative"],
     "the adverse event was serious: the patient died, was hospitalised, or it was life-threatening", "noul", None,
     "serious"),
    ("ae_system", "adverse_events", ["drug", "narrative"], "which body system does the adverse event affect?",
     "choice", ["liver", "heart", "skin", "nervous system", "digestive system"], "body_system"),
    ("txn_category", "transactions", ["merchant", "memo"], "what kind of business expense is this?", "choice",
     ["travel", "meals", "software", "office supplies", "entertainment"], "category"),
    ("job_seniority", "job_posts", ["title", "description"], "how senior is this role?", "score",
     ["junior", "mid-level", "senior"], "seniority"),
    ("job_remote", "job_posts", ["title", "description"], "the job can be done fully remotely", "noul", None, "remote"),
    ("listing_pets", "listings", ["headline", "description"], "pets are allowed in this rental", "noul", None, "pets"),
    ("clause_type", "clauses", ["text"], "what type of contract clause is this?", "choice",
     ["termination", "liability", "confidentiality", "payment", "intellectual property"], "clause_type"),
    ("email_guarantee", "emails", ["subject", "body"], "the email promises or guarantees investment returns",
     "noul", None, "violation"),
    ("company_match", "company_pairs", ["record_a", "record_b"], "record_a and record_b are the same company",
     "noul", None, "same"),
]

FIRST = ["Maria", "John", "Aisha", "Wei", "Carlos", "Priya", "Tom", "Fatima", "Lukas", "Yuki", "Olga", "Samuel",
         "Chloe", "Ahmed", "Ingrid", "Diego", "Nora", "Raj", "Emma", "Kofi"]
LAST = ["Garcia", "Smith", "Khan", "Zhang", "Rossi", "Patel", "Nguyen", "Muller", "Kowalski", "Silva", "Brown",
        "Tanaka", "Okafor", "Larsen", "Haddad", "Novak", "Moreau", "Kim"]
MONTHS = ["March", "April", "May", "June", "July", "August"]


def pick(rng, xs):
    return rng.choice(xs)


# ------------------------------------------------------------------ tickets
T_ISSUE = {
    "billing": ["I was charged {amt} twice for order {order}.", "My invoice {inv} shows {amt}, more than my plan.",
                "The refund of {amt} promised on {date} never arrived.", "I cancelled but you still billed me {amt}."],
    "technical": ["The app crashes when I open settings, error {err}.", "Since {date} I cannot log in: {err}.",
                  "Data export fails halfway with {err}.", "Pages take 30 seconds to load since the update."],
    "sales": ["What would 40 seats cost per year?", "Can we get a quote for our second office?",
              "Is there a discount for annual billing?", "Could someone demo the Enterprise plan to us?"],
    "security": ["I got a login alert from a country I have never visited.", "Someone changed my account email.",
                 "We leaked an API key in a public repo, please revoke it.", "2FA codes arrive for logins I did not make."],
}
T_TONE = {
    True: ["This is unacceptable and I am furious.", "Fix it TODAY or I cancel and tell everyone.",
           "I am sick of your useless support.", "WHY IS THIS STILL BROKEN? I AM DONE.",
           "You people are incompetent, I have wasted hours."],
    False: ["Thanks a lot for your help!", "No rush.", "Kind regards.", "This is the second time, please fix it.",
            "Could you check and get back to me?", "I would like this resolved soon."],
}
T_SUBJ = {"billing": ["Charge", "Invoice", "Refund"], "technical": ["Bug", "Crash", "Cannot log in"],
          "sales": ["Pricing", "Quote", "Demo"], "security": ["Security", "Suspicious login", "Account access"]}


def tickets(rng, i):
    team = rng.choices(list(T_ISSUE), [35, 35, 15, 15])[0]
    angry = rng.random() < .3
    slots = {"amt": "$%d.%02d" % (rng.randint(9, 4999), rng.randint(0, 99)), "order": "#%d" % rng.randint(100000, 999999),
             "inv": "INV-%d" % rng.randint(10000, 99999), "date": "%s %d" % (pick(rng, MONTHS), rng.randint(1, 28)),
             "err": pick(rng, ["E%d" % rng.randint(100, 999), "HTTP 500", "timeout"])}
    body = " ".join([pick(rng, ["Hi,", "Hello,", ""]), pick(rng, T_ISSUE[team]).format(**slots), pick(rng, T_TONE[angry])]).strip()
    return {"id": i, "subject": pick(rng, T_SUBJ[team]), "body": body, "angry": angry, "team": team}


# ------------------------------------------------------------------ product reviews
PRODUCTS = ["blender", "headphones", "desk lamp", "running shoes", "coffee grinder", "backpack", "phone case",
            "air fryer", "keyboard", "yoga mat", "water bottle", "office chair"]
R_POS = ["Absolutely love it, works perfectly.", "Best purchase this year.", "Great quality for the price.",
         "Exceeded my expectations, highly recommend."]
R_NEG = ["Complete waste of money.", "Very disappointed, returning it.", "Would not recommend to anyone.",
         "Cheap and poorly made."]
R_MIX = ["It does the job but the design is clumsy.", "Good sound, but uncomfortable after an hour.",
         "Nice looking, though it feels flimsy.", "Works fine, shipping took forever though."]
R_DEFECT = ["It stopped working after {n} days.", "Arrived with a cracked {part}.", "The {part} broke the first time I used it.",
            "It overheats and shuts off.", "One of the buttons is dead out of the box."]
R_FINE = ["Arrived on time and well packed.", "Setup took two minutes.", "The {part} feels solid.", "Battery lasts all day."]


def reviews(rng, i):
    sentiment = rng.choices(["negative", "mixed", "positive"], [30, 25, 45])[0]
    defect = sentiment != "positive" and rng.random() < .55
    prod, part = pick(rng, PRODUCTS), pick(rng, ["lid", "handle", "switch", "cable", "zipper", "base"])
    tone = {"negative": R_NEG, "mixed": R_MIX, "positive": R_POS}[sentiment]
    detail = pick(rng, R_DEFECT if defect else R_FINE).format(n=rng.randint(2, 40), part=part)
    text = " ".join(rng.sample([pick(rng, tone), detail], 2))
    stars = {"negative": pick(rng, ["1/5", "2/5"]), "mixed": "3/5", "positive": pick(rng, ["4/5", "5/5"])}[sentiment]
    title = "%s: %s" % (prod.capitalize(), pick(rng, {"negative": ["Avoid", "Disappointing", "Meh"],
                                                      "mixed": ["Okay", "Mixed feelings", "Decent"],
                                                      "positive": ["Great", "Love it", "Excellent"]}[sentiment]))
    return {"id": i, "product": prod, "title": title, "text": text, "stars_hidden": stars,
            "sentiment": sentiment, "defect": defect}


# ------------------------------------------------------------------ pharmacovigilance: adverse event reports
DRUGS = ["atorvastatin", "metformin", "amoxicillin", "ibuprofen", "sertraline", "lisinopril", "omeprazole",
         "levothyroxine", "warfarin", "gabapentin", "prednisone", "clopidogrel"]
AE = {
    "liver": ["elevated liver enzymes (ALT {n}x ULN)", "jaundice and dark urine", "drug-induced hepatitis"],
    "heart": ["palpitations and an irregular heartbeat", "QT prolongation on ECG", "chest pain with tachycardia"],
    "skin": ["a widespread itchy rash", "hives on the arms and neck", "blistering skin lesions"],
    "nervous system": ["severe dizziness and headache", "tingling and numbness in both feet", "a seizure"],
    "digestive system": ["persistent nausea and vomiting", "severe abdominal pain and diarrhoea", "gastrointestinal bleeding"],
}
SERIOUS = ["Patient was admitted to hospital for {d} days.", "The event was considered life-threatening; ICU admission.",
           "Patient died {d} days after onset.", "Required emergency hospitalisation and IV treatment."]
NONSERIOUS = ["Symptoms resolved after the drug was stopped.", "Managed as an outpatient; recovered fully.",
              "Patient visited the hospital lab for routine tests; no admission.", "Mild; the dose was reduced and symptoms eased."]


def adverse_events(rng, i):
    system = pick(rng, list(AE))
    serious = rng.random() < .35
    drug = pick(rng, DRUGS)
    d = rng.randint(2, 21)
    narrative = "A %d-year-old %s taking %s %dmg developed %s about %d days after starting treatment. %s" % (
        rng.randint(18, 89), pick(rng, ["man", "woman"]), drug, pick(rng, [5, 10, 20, 40, 50, 100, 500]),
        pick(rng, AE[system]).format(n=rng.randint(3, 12)), rng.randint(1, 60),
        pick(rng, SERIOUS if serious else NONSERIOUS).format(d=d))
    return {"id": i, "report_no": "AE-%06d" % rng.randint(0, 999999), "drug": drug, "narrative": narrative,
            "serious": serious, "body_system": system}


# ------------------------------------------------------------------ expense transactions
TXN = {
    "travel": [("Delta Air Lines", "flight {c1} to {c2}"), ("Marriott", "hotel {n} nights in {c2}"),
               ("Uber", "ride to the airport"), ("Hertz", "car rental in {c2}"), ("Amtrak", "train to {c2}")],
    "meals": [("Olive Garden", "team lunch, {n} people"), ("Starbucks", "coffee with a client"),
              ("DoorDash", "dinner while working late"), ("Chipotle", "lunch at the offsite")],
    "software": [("GitHub", "Team plan, {n} seats"), ("Atlassian", "Jira subscription"), ("AWS", "cloud hosting for {m}"),
                 ("Figma", "design licences"), ("Slack", "annual workspace plan")],
    "office supplies": [("Staples", "printer paper and toner"), ("IKEA", "desk and chair for the new hire"),
                        ("Amazon", "USB-C cables and a keyboard"), ("Office Depot", "whiteboard markers and notebooks")],
    "entertainment": [("Ticketmaster", "concert tickets for the client"), ("AMC Theatres", "team movie night"),
                      ("TopGolf", "team outing"), ("StubHub", "basketball game with partners")],
}
CITIES = ["Boston", "Denver", "Austin", "Chicago", "Seattle", "Miami", "Atlanta"]


def transactions(rng, i):
    cat = pick(rng, list(TXN))
    merchant, memo = pick(rng, TXN[cat])
    memo = memo.format(c1=pick(rng, CITIES), c2=pick(rng, CITIES), n=rng.randint(2, 12), m=pick(rng, MONTHS))
    return {"id": i, "merchant": merchant.upper() + " " + str(rng.randint(100, 9999)), "memo": memo,
            "amount": round(rng.uniform(4, 2400), 2), "category": cat}


# ------------------------------------------------------------------ job posts
ROLES = ["Software Engineer", "Data Analyst", "Product Designer", "Accountant", "Marketing Manager", "DevOps Engineer",
         "Nurse", "Sales Representative"]
SEN = {"junior": (["Junior", "Graduate", "Entry-level"], "0-2 years of experience; we will train you."),
       "mid-level": (["", "Mid-level"], "3-5 years of experience working independently."),
       "senior": (["Senior", "Lead", "Principal"], "8+ years of experience; you will mentor the team and own the roadmap.")}
REMOTE = ["This role is 100% remote, work from anywhere in the EU.", "Fully remote; we never meet in an office.",
          "Remote-first team, you choose where you work."]
ONSITE = ["You will work on site at our {c} office five days a week.", "Hybrid: three days a week in our {c} office.",
          "On-site role at the {c} hospital.", "Occasional remote days, mostly in the {c} office."]


def job_posts(rng, i):
    sen = pick(rng, list(SEN))
    remote = rng.random() < .4
    role = pick(rng, ROLES)
    titles, exp = SEN[sen]
    title = ("%s %s" % (pick(rng, titles), role)).strip()
    desc = "%s %s %s Salary %s." % (
        pick(rng, ["We are growing fast.", "Join our friendly team.", "Help us build the future of logistics."]), exp,
        pick(rng, REMOTE if remote else ONSITE).format(c=pick(rng, CITIES)),
        "$%dk-$%dk" % ((b := rng.randint(40, 180)), b + rng.randint(10, 40)))
    return {"id": i, "title": title, "description": desc, "seniority": sen, "remote": remote}


# ------------------------------------------------------------------ rental listings
PETS_YES = ["Cats and dogs welcome!", "Pet-friendly building with a dog park.", "Small pets allowed with a deposit."]
PETS_NO = ["No pets, please.", "Sorry, the landlord does not allow animals.", "Pets are not permitted in this building.",
           "Close to a large park, perfect for walks (no pets in the unit)."]


def listings(rng, i):
    pets = rng.random() < .45
    beds = rng.randint(1, 4)
    desc = "%d-bedroom apartment in %s, %d sq ft, %s. %s %s" % (
        beds, pick(rng, CITIES), rng.randint(450, 2200), pick(rng, ["newly renovated", "sunny corner unit", "quiet street",
                                                                  "walk to the metro"]),
        pick(rng, ["In-unit laundry.", "Balcony with a view.", "Gym in the building.", "Parking included."]),
        pick(rng, PETS_YES if pets else PETS_NO))
    return {"id": i, "headline": "%d BR in %s - $%d/mo" % (beds, pick(rng, CITIES), rng.randint(900, 5200)),
            "description": desc, "pets": pets}


# ------------------------------------------------------------------ contract clauses
CLAUSES = {
    "termination": ["Either party may terminate this Agreement upon {n} days' written notice.",
                    "This Agreement terminates automatically if the Customer becomes insolvent."],
    "liability": ["In no event shall either party's aggregate liability exceed the fees paid in the prior {n} months.",
                  "Neither party is liable for indirect, incidental or consequential damages."],
    "confidentiality": ["Each party shall keep the other's Confidential Information secret for {n} years.",
                        "The Recipient shall not disclose Confidential Information to any third party."],
    "payment": ["Invoices are payable within {n} days of receipt; late amounts accrue 1.5% interest per month.",
                "The Customer shall pay the annual fees in advance."],
    "intellectual property": ["All intellectual property created under this Agreement vests in the Customer.",
                              "The Supplier retains all rights, title and interest in its pre-existing software."],
}


def clauses(rng, i):
    t = pick(rng, list(CLAUSES))
    text = "%d.%d %s" % (rng.randint(2, 18), rng.randint(1, 9), pick(rng, CLAUSES[t]).format(n=pick(rng, [12, 30, 45, 60, 90])))
    return {"id": i, "contract": "MSA-%04d" % rng.randint(1, 9999), "text": text, "clause_type": t}


# ------------------------------------------------------------------ compliance emails
E_BAD = ["This fund is guaranteed to return {n}% a year, zero risk.", "I promise you will double your money by {m}.",
         "Our clients never lose: a guaranteed {n}% return.", "Risk-free investment with assured profits of {n}%."]
E_OK = ["Past performance does not guarantee future results.", "The fund returned {n}% last year; returns may vary.",
        "Attached is the quarterly statement for your review.", "Let's meet on {m} to review your risk profile.",
        "Markets were volatile this quarter; your portfolio is diversified."]


def emails(rng, i):
    bad = rng.random() < .25
    body = "Dear %s, %s %s Best, %s" % (pick(rng, FIRST), pick(rng, E_BAD if bad else E_OK).format(
        n=rng.randint(5, 40), m=pick(rng, MONTHS)), pick(rng, ["", "Call me anytime.", "Happy to discuss."]),
        pick(rng, FIRST))
    return {"id": i, "subject": pick(rng, ["Your portfolio", "Opportunity", "Quarterly update", "Follow-up"]),
            "body": body, "violation": bad}


# ------------------------------------------------------------------ entity resolution: company pairs
CO_A = ["Acme", "Globex", "Initech", "Umbrella", "Stark", "Wayne", "Hooli", "Vandelay", "Wonka", "Tyrell", "Cyberdyne",
        "Soylent", "Massive Dynamic", "Pied Piper", "Aperture"]
CO_B = ["Logistics", "Labs", "Industries", "Foods", "Energy", "Analytics", "Health", "Robotics", "Capital", "Media"]
SUFFIX = [("Inc.", "Incorporated"), ("LLC", "L.L.C."), ("Ltd", "Limited"), ("Corp.", "Corporation"), ("GmbH", "GmbH")]


def company_pairs(rng, i):
    a, b = pick(rng, CO_A), pick(rng, CO_B)
    s1, s2 = pick(rng, SUFFIX)
    city = pick(rng, CITIES)
    left = "%s %s %s, %s" % (a, b, s1, city)
    same = rng.random() < .5
    if same:
        variants = ["%s %s %s, %s" % (a.upper(), b.upper(), s2, city), "The %s %s Company (%s)" % (a, b, city),
                    "%s-%s %s, %s" % (a, b, s1, city), "%s %s, %s office" % (a, b, city)]
        right = pick(rng, variants)
    else:
        other = pick(rng, [x for x in CO_B if x != b]) if rng.random() < .6 else b
        a2 = a if other != b else pick(rng, [x for x in CO_A if x != a])
        right = "%s %s %s, %s" % (a2, other, pick(rng, SUFFIX)[0], pick(rng, CITIES))
    return {"id": i, "record_a": left, "record_b": right, "same": same}


TABLES = {"tickets": tickets, "reviews": reviews, "adverse_events": adverse_events, "transactions": transactions,
          "job_posts": job_posts, "listings": listings, "clauses": clauses, "emails": emails,
          "company_pairs": company_pairs}


def generate(rows_per_table, seed=11, dup_rate=0.03):
    out = {}
    for t, fn in TABLES.items():
        rng = random.Random("%s:%s" % (seed, t))
        n = rows_per_table * (2 if t == "company_pairs" else 1)     # 9 tables; pairs get double -> 10 x rows
        rows = []
        for i in range(1, n + 1):
            rows.append({**rng.choice(rows), "id": i} if rows and rng.random() < dup_rate else fn(rng, i))
        out[t] = rows
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--rows-per-table", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--out", default="bench/bench.duckdb")
    a = p.parse_args()
    import duckdb
    import pyarrow as pa
    con = duckdb.connect(a.out)
    total = 0
    for t, rows in generate(a.rows_per_table, a.seed).items():
        tbl = pa.Table.from_pylist(rows)      # noqa: F841 -- referenced by name in SQL
        con.execute("CREATE OR REPLACE TABLE %s AS SELECT * FROM tbl" % t)
        total += len(rows)
        print("  %-15s %6d rows" % (t, len(rows)))
    print("wrote %d rows, %d tasks, to %s" % (total, len(TASKS), a.out))
