"""
ContractLens - Flask app.

Serves the HTML/CSS/JS frontend (templates/, static/) and exposes one JSON
API endpoint that the frontend's JavaScript calls with fetch():

POST /analyze  {"text": "<raw contract text>"}
    -> runs the pipeline (segment -> classify -> flag -> rewrite) and
    returns per-clause results plus the overall document score as JSON

All the actual NLP/ML/rules work lives in src/ and is untouched by this
file - this is purely the web layer.
"""

from flask import Flask, render_template, request, jsonify

from src.segmenter import segment_into_clauses
from src.classifier import ClauseClassifier
from src.red_flags import evaluate_clause, score_document
from src.plain_english import rewrite_clause

app = Flask(__name__)

# Load the trained classifier once when the server starts, not on every
# request - loading the saved model from disk is the "slow" part and
# there's no reason to repeat it per request.
classifier = ClauseClassifier()


def run_full_analysis(raw_text: str) -> dict:
    """Run the whole pipeline and return a JSON-serializable result dict."""
    clauses = segment_into_clauses(raw_text)

    clause_rows = []
    clause_results = []
    for i, clause_text in enumerate(clauses, start=1):
        category, confidence = classifier.predict_with_confidence(clause_text)
        clause_result = evaluate_clause(clause_text, category)
        clause_results.append(clause_result)

        plain_english = ""
        reasons = [f.reason for f in clause_result.flags]
        if clause_result.is_flagged:
            plain_english = rewrite_clause(clause_text, category, reasons)

        clause_rows.append({
            "index": i,
            "text": clause_text,
            "category": category,
            "confidence": confidence,
            "flagged": clause_result.is_flagged,
            "risk_score": clause_result.risk_score,
            "risk_level": clause_result.risk_level,
            "reasons": reasons,
            "plain_english": plain_english,
        })

    return {
        "clauses": clause_rows,
        "document": score_document(clause_results),
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json(silent=True) or {}
    raw_text = (data.get("text") or "").strip()

    if not raw_text:
        return jsonify({"error": "No text provided."}), 400

    result = run_full_analysis(raw_text)

    if not result["clauses"]:
        return jsonify({"error": "No clauses were detected in this text. Try pasting more complete sentences."}), 400

    return jsonify(result)


if __name__ == "__main__":
    # use_reloader=False: the reloader watches every file under this
    # directory (including large model/data files) and can restart the
    # server mid-request, which hangs in-flight requests.
    app.run(debug=True, port=3000, use_reloader=False)
