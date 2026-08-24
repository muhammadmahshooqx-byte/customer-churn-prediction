from flask import Flask, render_template_string, request

from src.predict import predict

app = Flask(__name__)

FORM = """
<!doctype html><title>Churn Risk</title><style>body{font:16px system-ui;max-width:680px;margin:48px auto;padding:0 20px;color:#17324d}label{display:block;margin:14px 0 6px}input,select,button{font:inherit;padding:10px;width:100%;box-sizing:border-box}button{margin-top:22px;background:#e76f51;color:white;border:0;cursor:pointer}.result{margin-top:24px;padding:18px;background:#f1faee}</style>
<h1>Customer churn risk</h1><form method=post><label>Contract</label><select name=Contract><option>Month-to-month</option><option>One year</option><option>Two year</option></select><label>Tenure (months)</label><input name=tenure type=number value=12><label>Monthly charges</label><input name=MonthlyCharges type=number step=.01 value=75><label>Total charges</label><input name=TotalCharges type=number step=.01 value=900><label>Internet service</label><select name=InternetService><option>Fiber optic</option><option>DSL</option><option>No</option></select><button>Assess risk</button></form>{% if result %}<div class=result><strong>{{ result.risk }} risk</strong><br>Estimated churn probability: {{ '%.1f'|format(result.churn_probability * 100) }}%</div>{% endif %}
"""


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        result = predict(request.form.to_dict())
    return render_template_string(FORM, result=result)


if __name__ == "__main__":
    app.run(debug=True)
