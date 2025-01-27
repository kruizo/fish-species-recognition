from flask import Flask, render_template
from backend.routes import api  

app = Flask(__name__, 
            static_folder='frontend/static', 
            template_folder='frontend/template', 
            static_url_path='/static')

app.register_blueprint(api)

@app.route('/')
def home():
    return render_template('home.html')  

if __name__ == "__main__":
    app.run(debug=True)
