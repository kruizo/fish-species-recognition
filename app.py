from flask import Flask
from backend.routes import setup_routes

app = Flask(__name__, static_folder="frontend", template_folder="frontend")

@app.route("/")
def home():
    return app.send_static_file("home.html")
    
setup_routes(app)

if __name__ == "__main__":
    app.run(debug=True)