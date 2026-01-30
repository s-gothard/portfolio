from flask import Flask, render_template, request

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    results = []
    query = ""
    if request.method == "POST":
        query = request.form["query"]
        results = rank_documents(query, documents)
    return render_template("index.html", query=query, results=results)

if __name__ == "__main__":
    app.run(debug=True)