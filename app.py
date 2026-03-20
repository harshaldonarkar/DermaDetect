import os
import re
import uuid
import base64
from io import BytesIO
from datetime import datetime

from flask import (
    Flask, request, render_template, abort,
    send_file, redirect, url_for, flash,
)
from flask_login import (
    LoginManager, UserMixin,
    login_user, logout_user, login_required, current_user,
)
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.security import generate_password_hash, check_password_hash
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from bson import ObjectId
from PIL import Image
import numpy as np

import skin_cancer_detection as SCD
from gradcam import generate_gradcam, overlay_gradcam, to_base64
from report_generator import generate_pdf

# ── App setup ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dermadetect-dev-secret-change-in-prod")

# ── Rate limiter ──────────────────────────────────────────────────────────────
limiter = Limiter(
    key_func=get_remote_address,
    app=app,
    storage_uri="memory://",
    default_limits=[],
)

# ── Flask-Login ───────────────────────────────────────────────────────────────
login_manager = LoginManager(app)
login_manager.login_view = "login"
login_manager.login_message = "Please log in to access this page."


class User(UserMixin):
    def __init__(self, doc):
        self.id       = str(doc["_id"])
        self.username = doc["username"]
        self.email    = doc["email"]


@login_manager.user_loader
def load_user(user_id):
    if not mongo_available:
        return None
    doc = users_col.find_one({"_id": ObjectId(user_id)})
    return User(doc) if doc else None


# ── MongoDB ───────────────────────────────────────────────────────────────────
try:
    mongo_uri = os.environ.get("MONGO_URI", "mongodb://localhost:27017/")
    client = MongoClient(mongo_uri, serverSelectionTimeoutMS=3000)
    client.admin.command("ping")
    db          = client["skin_cancer_db"]
    collection  = db["patient_data"]
    users_col   = db["users"]
    users_col.create_index("username", unique=True)
    users_col.create_index("email",    unique=True)
    mongo_available = True
except ConnectionFailure:
    print("WARNING: MongoDB not reachable. Login and patient records disabled.")
    mongo_available = False

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "bmp", "webp"}
MAX_IMAGE_BYTES    = 10 * 1024 * 1024  # 10 MB

# Per-class low-confidence thresholds (higher = stricter warning)
CONFIDENCE_THRESHOLDS = {
    6: 70.0,  # Melanoma — most serious, raise the bar
    0: 65.0,  # Actinic keratosis — pre-malignant
}
DEFAULT_THRESHOLD = 60.0

# In-memory PDF cache {report_id: (pdf_bytes, patient_name)}
_report_cache = {}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def sanitize(text):
    """Strip HTML injection characters from user text input."""
    if not text:
        return text
    return re.sub(r'[<>]', '', str(text)).strip()


# ── Auth routes ───────────────────────────────────────────────────────────────
@app.route("/register", methods=["GET", "POST"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for("runhome"))

    if request.method == "POST":
        username = sanitize(request.form.get("username", ""))
        email    = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        confirm  = request.form.get("confirm_password", "")

        if not all([username, email, password, confirm]):
            flash("All fields are required.", "error")
        elif password != confirm:
            flash("Passwords do not match.", "error")
        elif len(password) < 6:
            flash("Password must be at least 6 characters.", "error")
        elif not mongo_available:
            flash("Database unavailable. Cannot register.", "error")
        else:
            if users_col.find_one({"$or": [{"username": username}, {"email": email}]}):
                flash("Username or email already taken.", "error")
            else:
                users_col.insert_one({
                    "username":      username,
                    "email":         email,
                    "password_hash": generate_password_hash(password),
                    "created_at":    datetime.now(),
                })
                flash("Account created! Please log in.", "success")
                return redirect(url_for("login"))

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("runhome"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        if not mongo_available:
            flash("Database unavailable. Cannot log in.", "error")
        else:
            doc = users_col.find_one({"username": username})
            if doc and check_password_hash(doc["password_hash"], password):
                login_user(User(doc), remember=request.form.get("remember") == "on")
                return redirect(request.args.get("next") or url_for("runhome"))
            flash("Invalid username or password.", "error")

    return render_template("login.html")


@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("runhome"))


# ── Public routes ─────────────────────────────────────────────────────────────
@app.route("/", methods=["GET", "POST"])
def runhome():
    return render_template("page1.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/contact", methods=["GET", "POST"])
def contact():
    if request.method == "POST":
        name    = sanitize(request.form.get("name"))
        email   = request.form.get("email", "").strip()
        message = sanitize(request.form.get("message"))
        if mongo_available:
            collection.insert_one({
                "form_type": "contact_us",
                "name": name, "email": email, "message": message,
                "timestamp": datetime.now(),
            })
        return render_template("contact_success.html", name=name)
    return render_template("contact.html")


# ── Protected routes ──────────────────────────────────────────────────────────
@app.route("/page3")
@login_required
def upload_page():
    return render_template("page3.html")


def _build_records(docs):
    """Format a list of MongoDB docs for template rendering."""
    for r in docs:
        r["record_id"] = str(r.pop("_id"))
        if "timestamp" in r:
            r["timestamp_raw"] = r["timestamp"].strftime("%Y-%m-%d")
            r["timestamp"]     = r["timestamp"].strftime("%d %b %Y, %H:%M")
    return docs


def _compute_stats(cursor):
    stats = {"total": 0, "cancer": 0, "non_cancer": 0, "caution": 0, "diagnosis_counts": {}}
    for r in cursor:
        diag = r.get("predicted_class", "")
        if not diag:
            continue
        stats["total"] += 1
        stats["diagnosis_counts"][diag] = stats["diagnosis_counts"].get(diag, 0) + 1
        if "Non-Cancerous" in diag:
            stats["non_cancer"] += 1
        elif "lead" in diag:
            stats["caution"] += 1
        else:
            stats["cancer"] += 1
    return stats


@app.route("/history")
@login_required
def history():
    records = []
    stats   = {"total": 0, "cancer": 0, "non_cancer": 0, "caution": 0, "diagnosis_counts": {}}

    if mongo_available:
        docs = list(
            collection.find({"form_type": {"$exists": False}})
            .sort("timestamp", -1).limit(200)
        )
        records = _build_records(docs)
        stats   = _compute_stats(
            collection.find({"form_type": {"$exists": False}}, {"predicted_class": 1})
        )

    return render_template("history.html", records=records,
                           mongo_available=mongo_available, stats=stats)


@app.route("/delete_record/<record_id>", methods=["POST"])
@login_required
def delete_record(record_id):
    if not mongo_available:
        abort(503, "Database unavailable.")
    try:
        collection.delete_one({"_id": ObjectId(record_id)})
    except Exception:
        abort(400, "Invalid record ID.")
    flash("Record deleted successfully.", "success")
    return redirect(url_for("history"))


@app.route("/profile")
@login_required
def profile():
    records = []
    stats   = {"total": 0, "cancer": 0, "non_cancer": 0, "caution": 0, "diagnosis_counts": {}}

    if mongo_available:
        docs = list(
            collection.find({"form_type": {"$exists": False}, "recorded_by": current_user.username})
            .sort("timestamp", -1).limit(200)
        )
        records = _build_records(docs)
        stats   = _compute_stats(iter(records))   # records already formatted; re-iterate for stats

    # Re-compute stats from fresh cursor (records are already mutated by _build_records)
    if mongo_available:
        stats = _compute_stats(
            collection.find(
                {"form_type": {"$exists": False}, "recorded_by": current_user.username},
                {"predicted_class": 1}
            )
        )

    return render_template("profile.html", records=records,
                           mongo_available=mongo_available, stats=stats)


@app.route("/export_csv")
@login_required
def export_csv():
    if not mongo_available:
        abort(503, "Database unavailable.")
    import csv
    from io import StringIO
    records = list(
        collection.find({"form_type": {"$exists": False}}, {"_id": 0})
        .sort("timestamp", -1)
    )
    si = StringIO()
    fields = ["patient_name", "patient_id", "patient_age", "patient_sex",
              "patient_ethnicity", "medical_history", "predicted_class",
              "confidence", "recorded_by", "timestamp"]
    writer = csv.DictWriter(si, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for r in records:
        if "timestamp" in r:
            r["timestamp"] = r["timestamp"].strftime("%d %b %Y %H:%M")
        writer.writerow({f: r.get(f, "") for f in fields})
    output = si.getvalue().encode("utf-8")
    return send_file(
        BytesIO(output),
        mimetype="text/csv",
        as_attachment=True,
        download_name=f"DermaDetect_records_{datetime.now().strftime('%Y%m%d')}.csv",
    )


@app.route("/showresult", methods=["POST"])
@login_required
@limiter.limit("10 per minute")
def show():
    pic = request.files.get("pic")
    if not pic or not allowed_file(pic.filename):
        abort(400, "Please upload a valid image file (jpg, jpeg, png, bmp, webp).")

    # Read bytes and validate size
    pic_bytes = pic.read()
    if len(pic_bytes) > MAX_IMAGE_BYTES:
        abort(400, "Image too large. Maximum allowed size is 10 MB.")

    # Validate it is actually a decodable image
    try:
        inputimg = Image.open(BytesIO(pic_bytes)).convert("RGB")
    except Exception:
        abort(400, "Uploaded file could not be read as an image.")

    patient_name      = sanitize(request.form.get("patient_name"))
    patient_id        = sanitize(request.form.get("patient_id"))
    patient_age       = request.form.get("patient_age")
    patient_sex       = request.form.get("patient_sex")
    patient_ethnicity = request.form.get("patient_ethnicity")
    medical_history   = sanitize(request.form.get("medical_history"))

    input_size = SCD.INPUT_SHAPE[:2]
    inputimg   = inputimg.resize((input_size[1], input_size[0]))
    img = np.array(inputimg, dtype=np.float32).reshape(-1, *SCD.INPUT_SHAPE)

    result        = SCD.model.predict(img)
    probabilities = result[0].tolist()
    max_prob      = max(probabilities)
    class_ind     = probabilities.index(max_prob)
    result_label  = SCD.classes[class_ind]
    confidence_pct = round(max_prob * 100, 1)
    threshold      = CONFIDENCE_THRESHOLDS.get(class_ind, DEFAULT_THRESHOLD)
    low_confidence = confidence_pct < threshold

    original_display = np.array(inputimg)
    heatmap      = generate_gradcam(SCD.model, img, class_ind)
    gradcam_uri  = overlay_gradcam(original_display, heatmap)
    original_uri = to_base64(inputimg)

    info_dict = {
        0: "Actinic keratosis also known as solar keratosis or senile keratosis are names given to intraepithelial keratinocyte dysplasia. As such they are a pre-malignant lesion or in situ squamous cell carcinomas and thus a malignant lesion.",
        1: "Basal cell carcinoma is a type of skin cancer. Basal cell carcinoma begins in the basal cells — a type of cell within the skin that produces new skin cells as old ones die off. Basal cell carcinoma often appears as a slightly transparent bump on the skin, though it can take other forms. Basal cell carcinoma occurs most often on areas of the skin that are exposed to the sun, such as your head and neck.",
        2: "Benign lichenoid keratosis (BLK) usually presents as a solitary lesion that occurs predominantly on the trunk and upper extremities in middle-aged women. The pathogenesis of BLK is unclear; however, it has been suggested that BLK may be associated with the inflammatory stage of regressing solar lentigo (SL).",
        3: "Dermatofibromas are small, noncancerous (benign) skin growths that can develop anywhere on the body but most often appear on the lower legs, upper arms or upper back. These nodules are common in adults but are rare in children. They can be pink, gray, red, or brown in color and may change color over the years. They are firm and often feel like a stone under the skin.",
        4: "A melanocytic nevus (also known as nevocytic nevus, nevus-cell nevus and commonly as a mole) is a type of melanocytic tumor that contains nevus cells. Some sources equate the term mole with 'melanocytic nevus', but there are also sources that equate the term mole with any nevus form.",
        5: "Pyogenic granulomas are skin growths that are small, round, and usually bloody red in color. They tend to bleed because they contain a large number of blood vessels. They're also known as lobular capillary hemangioma or granuloma telangiectaticum.",
        6: "Melanoma, the most serious type of skin cancer, develops in the cells (melanocytes) that produce melanin — the pigment that gives your skin its color. Melanoma can also form in your eyes and, rarely, inside your body, such as in your nose or throat. The exact cause of all melanomas isn't clear, but exposure to ultraviolet (UV) radiation from sunlight or tanning lamps and beds increases your risk of developing melanoma.",
    }
    info = info_dict[class_ind]

    gradcam_b64 = gradcam_uri.split(",", 1)[1]
    gradcam_pil = Image.open(BytesIO(base64.b64decode(gradcam_b64)))

    pdf_bytes = generate_pdf(
        patient_name, patient_id, patient_age, patient_sex,
        patient_ethnicity, medical_history,
        result_label, confidence_pct, low_confidence, info,
        inputimg, gradcam_pil,
    )
    report_id = str(uuid.uuid4())
    _report_cache[report_id] = (pdf_bytes, patient_name or "patient")

    if mongo_available:
        collection.insert_one({
            "patient_name": patient_name, "patient_id": patient_id,
            "patient_age": patient_age, "patient_sex": patient_sex,
            "patient_ethnicity": patient_ethnicity, "medical_history": medical_history,
            "predicted_class": result_label, "confidence": confidence_pct,
            "recorded_by": current_user.username,
            "timestamp": datetime.now(),
        })

    return render_template(
        "results.html",
        result=result_label, info=info,
        confidence=confidence_pct, low_confidence=low_confidence,
        original_image=original_uri, gradcam_image=gradcam_uri,
        report_id=report_id,
    )


@app.route("/download_report/<report_id>")
@login_required
def download_report(report_id):
    entry = _report_cache.get(report_id)
    if not entry:
        abort(404, "Report not found or expired. Please run a new scan.")
    pdf_bytes, patient_name = entry
    filename = f"DermaDetect_{patient_name.replace(' ', '_')}.pdf"
    return send_file(
        BytesIO(pdf_bytes),
        mimetype="application/pdf",
        as_attachment=True,
        download_name=filename,
    )


@app.errorhandler(400)
def bad_request(e):
    return render_template("error.html", code=400, message=str(e.description)), 400

@app.errorhandler(404)
def not_found(e):
    return render_template("error.html", code=404, message="The page you're looking for doesn't exist."), 404

@app.errorhandler(429)
def rate_limit_exceeded(e):
    return render_template("error.html", code=429,
        message="Too many scan requests. Please wait a moment before submitting again."), 429

@app.errorhandler(500)
def server_error(e):
    return render_template("error.html", code=500, message="Something went wrong on our end. Please try again."), 500


if __name__ == "__main__":
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=5002, debug=debug)
