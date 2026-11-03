from pathlib import Path
from datetime import datetime
import sqlite3
from functools import wraps

import numpy as np
import pandas as pd
from flask import Flask, flash, redirect, render_template, request, session, url_for
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from werkzeug.security import check_password_hash, generate_password_hash
from io import BytesIO
from flask import send_file
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
import re
import pdfplumber

BASE_DIR     = Path(__file__).resolve().parent
DATA_DIR     = BASE_DIR / "data"
DATA_FILE    = DATA_DIR / "electricity_dataset.csv"
HISTORY_FILE = DATA_DIR / "prediction_history.csv"
DB_FILE      = DATA_DIR / "app.db"

# (internal_key, csv_column_name, display_name, unit_label, max_value, step)
FEATURE_DEFS = [
    ("tv",             "TV (hrs/day)",               "TV",              "hrs/day",  24,  0.5),
    ("fan",            "Fan (hrs/day)",              "Fan",             "hrs/day",  24,  0.5),
    ("ac",             "AC (hrs/day)",               "AC",              "hrs/day",  24,  0.5),
    ("refrigerator",   "Refrigerator (hrs/day)",     "Refrigerator",    "hrs/day",  24,  0.5),
    ("washing_machine","Washing Machine (hrs/week)",  "Washing Machine", "hrs/week",  7,  0.5),
    ("geyser",         "Geyser (hrs/day)",           "Heater",          "hrs/day",  24,  0.5),
    ("lights",         "Lights (hrs/day)",           "Lights",          "hrs/day",  24,  0.5),
]

APPLIANCES    = [f[0] for f in FEATURE_DEFS]
CSV_COL       = {f[0]: f[1] for f in FEATURE_DEFS}
DISPLAY_NAMES = {f[0]: f[2] for f in FEATURE_DEFS}
UNITS         = {f[0]: f[3] for f in FEATURE_DEFS}
MAX_VALS      = {f[0]: f[4] for f in FEATURE_DEFS}
STEPS         = {f[0]: f[5] for f in FEATURE_DEFS}

app = Flask(__name__)
app.secret_key = "electricity-bill-analyzer-secret-key"
RATE_PER_UNIT = 5.0   
TN_SLABS = [(100, 0.0), (200, 2.25), (400, 4.50), (500, 6.00),
            (600, 8.00), (800, 9.00), (1000, 10.00), (float("inf"), 11.00)]
FIXED_BIMONTH = 45.0


def tn_bill(monthly_units):
    units = monthly_units * 2          # bi-monthly
    if units <= 0:
        return 0.0
    amount, prev = 0.0, 0
    for limit, rate in TN_SLABS:
        if units > prev:
            amount += (min(units, limit) - prev) * rate
        prev = limit
    amount += FIXED_BIMONTH
    return round(amount / 2, 2)        


@app.context_processor
def inject_today():
    now = datetime.now()
    return {
        "current_day": now.strftime("%A"),
        "current_date": now.strftime("%B %d, %Y"),
    }


# ── Database ──────────────────────────────────────────────────────────────────

def get_db_connection():
    DATA_DIR.mkdir(exist_ok=True)
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with get_db_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                name          TEXT NOT NULL,
                email         TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                created_at    TEXT NOT NULL
            )
        """)

        cur = conn.execute("PRAGMA table_info(electricity_bill)")
        cols = {row[1]: row[2] for row in cur.fetchall()}
        if cols and cols.get("tv") == "TEXT":
            conn.execute("DROP TABLE electricity_bill")

        conn.execute("""
            CREATE TABLE IF NOT EXISTS electricity_bill (
                customer_id     INTEGER PRIMARY KEY AUTOINCREMENT,
                tv              REAL NOT NULL,
                fan             REAL NOT NULL,
                ac              REAL NOT NULL,
                refrigerator    REAL NOT NULL,
                washing_machine REAL NOT NULL,
                geyser          REAL NOT NULL,
                lights          REAL NOT NULL,
                units_consumed  REAL NOT NULL,
                bill_amount     REAL NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS budgets (
                user_id INTEGER PRIMARY KEY,
                amount  REAL NOT NULL
            )
        """)
        conn.commit()


def current_user():
    uid = session.get("user_id")
    if not uid:
        return None
    with get_db_connection() as conn:
        return conn.execute(
            "SELECT id, name, email FROM users WHERE id = ?", (uid,)
        ).fetchone()


def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not session.get("user_id"):
            flash("Please login to continue.", "error")
            return redirect(url_for("login"))
        return view(*args, **kwargs)
    return wrapped


# ── ML ────────────────────────────────────────────────────────────────────────

def load_dataset():
    data = (pd.read_excel(DATA_FILE)
            if DATA_FILE.suffix.lower() in (".xlsx", ".xls")
            else pd.read_csv(DATA_FILE, sep=None, engine="python"))

    rename = {v: k for k, v in CSV_COL.items()}
    for col in data.columns:
        if "Bill_Amount" in col or "Bill Amount" in col or "Bill" in col:
            rename[col] = "Bill_Amount"
    data = data.rename(columns=rename)

    required = APPLIANCES + ["Units_Consumed", "Bill_Amount"]
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError(f"Dataset missing columns: {', '.join(missing)}")

    for key in APPLIANCES:
        data[key] = pd.to_numeric(data[key], errors="coerce")

    # Data augmentation to cover zero/low usage combinations
    augmented_rows = []
    np.random.seed(42)

    # 1. Zero usage explicitly
    augmented_rows.append({a: 0.0 for a in APPLIANCES} | {"Units_Consumed": 0.0, "Bill_Amount": 0.0})

    # 2. Single appliance usage patterns
    rates = {
        "tv": 4.97,
        "fan": 3.05,
        "ac": 17.97,
        "refrigerator": 1.95,
        "washing_machine": 9.08,
        "geyser": 13.92,
        "lights": 2.03
    }

    for app_name, rate in rates.items():
        for hrs in np.linspace(0.5, 16, 20):
            row = {a: 0.0 for a in APPLIANCES}
            row[app_name] = round(hrs, 1)
            u = hrs * rate
            b = u * 7.15
            row["Units_Consumed"] = round(u, 2)
            row["Bill_Amount"] = round(b, 2)
            augmented_rows.append(row)

    # 3. Random low & partial combinations
    for _ in range(600):
        tv   = np.random.choice([0, 1, 2, 4, 6, 8])
        fan  = np.random.choice(range(0, 17))
        ac   = np.random.choice([0, 1, 2, 4, 6, 8, 10])
        ref  = np.random.choice([0, 6, 12, 18, 24])
        wm   = np.random.choice([0, 1, 2, 3, 4, 5])
        gey  = np.random.choice([0, 0.5, 1, 1.5, 2])
        lig  = np.random.choice(range(0, 13))

        u = (tv * 4.97) + (fan * 3.05) + (ac * 17.97) + (ref * 1.95) + (wm * 9.08) + (gey * 13.92) + (lig * 2.03)
        b = u * 7.15
        augmented_rows.append({
            "tv": tv, "fan": fan, "ac": ac, "refrigerator": ref,
            "washing_machine": wm, "geyser": gey, "lights": lig,
            "Units_Consumed": round(u, 2), "Bill_Amount": round(b, 2)
        })

    aug_df = pd.DataFrame(augmented_rows)
    combined_df = pd.concat([data[required], aug_df], ignore_index=True)
    return combined_df.dropna()


def train_model():
    data = load_dataset()
    X = data[APPLIANCES]
    y = data[["Units_Consumed", "Bill_Amount"]]

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_tr, y_tr)

    preds = model.predict(X_te)
    metrics = {
        "mae":     round(mean_absolute_error(y_te, preds), 2),
        "r2":      round(r2_score(y_te, preds), 2) if len(X_te) > 1 else 1,
        "records": len(data),
    }
    return model, metrics


# ── History ───────────────────────────────────────────────────────────────────

def empty_history():
    return pd.DataFrame(columns=["User_ID", "Date", *APPLIANCES, "Units_Consumed", "Bill_Amount"])


def read_history(user_id=None):
    if HISTORY_FILE.exists():
        hist = pd.read_csv(HISTORY_FILE)
        if "tv" not in hist.columns:
            hist = empty_history()
        if "User_ID" not in hist.columns:
            hist.insert(0, "User_ID", 0)
    else:
        hist = empty_history()

    if user_id is not None:
        hist = hist[hist["User_ID"].astype(str) == str(user_id)]
    return hist


def save_prediction(user_id, inputs, units, bill):
    DATA_DIR.mkdir(exist_ok=True)
    hist = read_history()
    row = {
        "User_ID": user_id,
        "Date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        **{a: inputs[a] for a in APPLIANCES},
        "Units_Consumed": round(units, 2),
        "Bill_Amount": round(bill, 2),
    }
    hist = pd.concat([hist, pd.DataFrame([row])], ignore_index=True) if not hist.empty else pd.DataFrame([row])
    hist.to_csv(HISTORY_FILE, index=False)

    with get_db_connection() as conn:
        conn.execute(
            """INSERT INTO electricity_bill
               (tv, fan, ac, refrigerator, washing_machine, geyser, lights,
                units_consumed, bill_amount)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (inputs["tv"], inputs["fan"], inputs["ac"], inputs["refrigerator"],
             inputs["washing_machine"], inputs["geyser"], inputs["lights"],
             round(units, 2), round(bill, 2)),
        )
        conn.commit()


def next_month_prediction(user_id, cur_units, cur_bill):
    if cur_units <= 0 or cur_bill <= 0:
        return 0.0, 0.0

    hist = read_history(user_id)
    growth_factor = 1.025  # Standard +2.5% seasonal variation

    if not hist.empty and len(hist) >= 2:
        recent_units = hist["Units_Consumed"].tail(3).astype(float).values
        if len(recent_units) > 0 and recent_units[-1] > 0:
            trend = (cur_units - recent_units[-1]) / recent_units[-1]
            growth_factor = round(1.0 + float(np.clip(trend * 0.2, -0.05, 0.05)), 3)

    season = {1: 1.00, 2: 1.03, 3: 1.08, 4: 1.12, 5: 1.10, 6: 1.02,
              7: 0.98, 8: 0.98, 9: 1.00, 10: 1.00, 11: 0.97, 12: 0.98}
    next_month = (datetime.now().month % 12) + 1
    growth_factor = round(growth_factor * season[next_month], 3)

    next_units = round(cur_units * growth_factor, 1)
    unit_rate  = cur_bill / cur_units
    next_bill  = round(next_units * unit_rate, 2)

    return next_units, next_bill
           

def make_suggestion(inputs, units, bill):
    heavy = []
    if inputs.get("ac", 0) >= 4:      heavy.append("AC")
    if inputs.get("geyser", 0) >= 1:   heavy.append("Heater")
    if inputs.get("tv", 0) >= 8:       heavy.append("TV")
    if inputs.get("washing_machine", 0) >= 4: heavy.append("Washing Machine")
    if inputs.get("refrigerator", 0) >= 20:   heavy.append("Refrigerator")

    if units == 0:
        return "No appliance usage selected. Total consumption and bill are 0."
    if bill > 2000:
        focus = ", ".join(heavy[:3]) if heavy else "high-power appliances"
        return (f"Your bill is high. Continuous usage of {focus} is the main driver. "
                "Try turning off unused devices and adjusting AC temperature settings.")
    if units < 150:
        return "Great! Usage is well-controlled. Keep using energy-efficient appliances and switching off idle devices."
    return "Usage is moderate. Monitor AC, heater, and appliance runtimes to prevent higher bills."


# ── Auth routes ───────────────────────────────────────────────────────────────

@app.route("/register", methods=["GET", "POST"])
def register():
    if session.get("user_id"):
        return redirect(url_for("index"))

    if request.method == "POST":
        name  = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        pwd   = request.form.get("password", "")
        cpwd  = request.form.get("confirm_password", "")

        if not name or not email or not pwd:
            flash("All fields are required.", "error")
        elif pwd != cpwd:
            flash("Passwords do not match.", "error")
        elif len(pwd) < 6:
            flash("Password must have at least 6 characters.", "error")
        else:
            try:
                with get_db_connection() as conn:
                    cur = conn.execute(
                        "INSERT INTO users (name, email, password_hash, created_at) VALUES (?,?,?,?)",
                        (name, email, generate_password_hash(pwd),
                         datetime.now().strftime("%Y-%m-%d %H:%M")),
                    )
                    conn.commit()
                    session["user_id"]   = cur.lastrowid
                    session["user_name"] = name
                flash("Registration successful.", "success")
                return redirect(url_for("index"))
            except sqlite3.IntegrityError:
                flash("Email already registered. Please login.", "error")

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if session.get("user_id"):
        return redirect(url_for("index"))

    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        pwd   = request.form.get("password", "")

        with get_db_connection() as conn:
            user = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()

        if user and check_password_hash(user["password_hash"], pwd):
            session["user_id"]   = user["id"]
            session["user_name"] = user["name"]
            flash("Login successful.", "success")
            return redirect(url_for("index"))

        flash("Invalid email or password.", "error")

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    flash("Logged out successfully.", "success")
    return redirect(url_for("login"))


# ── Main route ────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET", "POST"])
@login_required
def index():
    user = current_user()
    model, metrics = train_model()

    if request.method == "POST":
        inputs = {}
        for a in APPLIANCES:
            try:
                val = float(request.form.get(a, 0))
                val = max(0.0, min(val, float(MAX_VALS[a])))
            except (ValueError, TypeError):
                val = 0.0
            inputs[a] = val

        if sum(inputs.values()) == 0:
            units = 0.0
            bill = 0.0
        else:
            pred  = model.predict(pd.DataFrame([inputs], columns=APPLIANCES))[0]
            units = max(0.0, round(float(pred[0]), 2))
            bill  = max(0.0, round(float(pred[1]), 2))
            bill  = tn_bill(units)

        next_u, next_b = next_month_prediction(user["id"], units, bill)
        suggestion     = make_suggestion(inputs, units, bill)
        save_prediction(user["id"], inputs, units, bill)

        session["last_calculation"] = {
            "selected": inputs,
            "result": {
                "units":      units,
                "bill":       bill,
                "next_units": next_u,
                "next_bill":  next_b,
                "suggestion": suggestion,
                "status":     "High Bill" if bill > 2000 else "Normal Bill",
            }
        }
        return redirect(url_for("index"))

    last_calc = session.pop("last_calculation", None)
    if last_calc:
        selected = last_calc.get("selected", {a: 0.0 for a in APPLIANCES})
        result   = last_calc.get("result")
    else:
        selected = {a: 0.0 for a in APPLIANCES}
        result   = None

    history = read_history(user["id"]).tail(8).to_dict("records")
    return render_template(
        "index.html",
        appliances=APPLIANCES,
        display_names=DISPLAY_NAMES,
        units_map=UNITS,
        max_vals=MAX_VALS,
        steps=STEPS,
        selected=selected,
        result=result,
        history=history,
        metrics=metrics,
        user=user,
        budget=get_budget(user["id"]),
        current_day=datetime.now().strftime("%A"),
    )


@app.route("/insights")
@login_required
def insights():
    user = current_user()
    model, metrics = train_model()
    history = read_history(user["id"]).tail(12).to_dict("records")

    if history:
        bills = [float(row["Bill_Amount"]) for row in history]
        units = [float(row["Units_Consumed"]) for row in history]
        average_bill = round(sum(bills) / len(bills), 2)
        average_units = round(sum(units) / len(units), 2)
        highest_bill = round(max(bills), 2)
        total_saved = round(max(0, highest_bill - average_bill), 2)
    else:
        average_bill = average_units = highest_bill = total_saved = 0

    return render_template(
        "insights.html",
        user=user,
        history=history,
        metrics=metrics,
        average_bill=average_bill,
        average_units=average_units,
        highest_bill=highest_bill,
        total_saved=total_saved,
    )


@app.route("/history")
@login_required
def history_page():
    user = current_user()
    hist = read_history(user["id"]).reset_index(drop=True)
    hist["idx"] = hist.index
    history = hist.tail(24).to_dict("records")
    return render_template("history.html", user=user, history=history)
@app.route("/report")
@app.route("/report/<int:row>")
@login_required
def report(row=None):
    user = current_user()
    hist = read_history(user["id"]).reset_index(drop=True)
    if hist.empty:
        flash("Run an analysis first to download a report.", "error")
        return redirect(url_for("index"))

    if row is None or row >= len(hist):
        row = len(hist) - 1
    last = hist.iloc[row]
    inputs = {a: float(last[a]) for a in APPLIANCES}
    units  = float(last["Units_Consumed"])
    bill   = float(last["Bill_Amount"])
    next_u, next_b = next_month_prediction(user["id"], units, bill)
    suggestion = make_suggestion(inputs, units, bill)
    status = "High Bill" if bill > 2000 else "Normal Bill"

    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4)
    s = getSampleStyleSheet()
    els = [
        Paragraph("AI Electricity Bill Analyzer - Report", s["Title"]),
        Paragraph(f"User: {user['name']} ({user['email']})", s["Normal"]),
        Paragraph(f"Generated: {datetime.now():%d %B %Y, %H:%M}", s["Normal"]),
        Spacer(1, 16),
        Paragraph("Appliance Usage", s["Heading2"]),
    ]

    rows = [["Appliance", "Usage"]]
    for a in APPLIANCES:
        rows.append([DISPLAY_NAMES[a], f"{inputs[a]} {UNITS[a]}"])
    t1 = Table(rows, colWidths=[220, 200])
    t1.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f766e")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
    ]))
    els += [t1, Spacer(1, 16), Paragraph("Prediction Summary", s["Heading2"])]
    summ = [
        ["Units consumed", f"{units} kWh"],
        ["Estimated bill", f"Rs. {bill}"],
        ["Next month units", f"{next_u} kWh"],
        ["Next month bill", f"Rs. {next_b}"],
        ["Status", status],
    ]
    t2 = Table(summ, colWidths=[220, 200])
    t2.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.5, colors.grey)]))
    els += [t2, Spacer(1, 16),
            Paragraph("Saving Suggestion", s["Heading2"]),
            Paragraph(suggestion, s["Normal"])]

    doc.build(els)
    buf.seek(0)
    return send_file(buf, as_attachment=True,
                     download_name="electricity_report.pdf",
                     mimetype="application/pdf")
def get_budget(user_id):
    with get_db_connection() as conn:
        row = conn.execute("SELECT amount FROM budgets WHERE user_id = ?", (user_id,)).fetchone()
    return row["amount"] if row else None


@app.route("/set-budget", methods=["POST"])
@login_required
def set_budget():
    user = current_user()
    try:
        amount = max(0.0, float(request.form.get("budget", 0)))
    except ValueError:
        amount = 0.0
    with get_db_connection() as conn:
        conn.execute(
            "INSERT INTO budgets (user_id, amount) VALUES (?, ?) "
            "ON CONFLICT(user_id) DO UPDATE SET amount = excluded.amount",
            (user["id"], amount),
        )
        conn.commit()
    flash("Budget saved.", "success")
    return redirect(url_for("index"))                     
@app.route("/upload-bill", methods=["POST"])
@login_required
def upload_bill():
    user = current_user()
    f = request.files.get("bill")
    if not f or not f.filename.lower().endswith(".pdf"):
        flash("Please upload a PDF bill.", "error")
        return redirect(url_for("index"))

    try:
        with pdfplumber.open(f) as pdf:
            text = "\n".join((p.extract_text() or "") for p in pdf.pages)
    except Exception:
        flash("Could not read this PDF.", "error")
        return redirect(url_for("index"))

    m_u = re.search(r"UNITS\s*:?\s*([\d,]+(?:\.\d+)?)", text, re.I)
    m_b = re.search(r"BILL\s*AMOUNT\s*:?\s*([\d,]+(?:\.\d+)?)", text, re.I)
    if not m_u or not m_b:
        flash("Units / bill amount not found in this PDF.", "error")
        return redirect(url_for("index"))

    units = float(m_u.group(1).replace(",", ""))
    bill = float(m_b.group(1).replace(",", ""))
    inputs = {a: 0.0 for a in APPLIANCES}

    next_u, next_b = next_month_prediction(user["id"], units, bill)
    suggestion = make_suggestion(inputs, units, bill)
    save_prediction(user["id"], inputs, units, bill)

    session["last_calculation"] = {
        "selected": inputs,
        "result": {
            "units": units, "bill": bill,
            "next_units": next_u, "next_bill": next_b,
            "suggestion": suggestion,
            "status": "High Bill" if bill > 2000 else "Normal Bill",
        },
    }
    flash("Bill uploaded successfully.", "success")
    return redirect(url_for("index"))


init_db()

if __name__ == "__main__":
    app.run(debug=True)
