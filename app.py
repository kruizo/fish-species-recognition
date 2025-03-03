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

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

if __name__ == "__main__":
    app.run(debug=True)
