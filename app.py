import os
from flask import Flask, render_template, request, flash, redirect, url_for
from flask_mail import Mail, Message
from backend.routes import api  
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__, 
            static_folder='frontend/static', 
            template_folder='frontend/template', 
            static_url_path='/static')

app.register_blueprint(api)

app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')
app.config['MAIL_SERVER'] = os.getenv('MAIL_SERVER')  
app.config['MAIL_PORT'] = os.getenv('MAIL_PORT')
app.config['MAIL_USE_TLS'] = os.getenv('MAIL_USE_TLS', 'False').lower() == 'true'
app.config['MAIL_USE_SSL'] = os.getenv('MAIL_USE_SSL', 'False').lower() == 'true'
app.config['MAIL_USERNAME'] = os.getenv('MAIL_USERNAME')  
app.config['MAIL_PASSWORD'] = os.getenv('MAIL_PASSWORD')
app.config['MAIL_DEFAULT_SENDER'] = os.getenv('MAIL_DEFAULT_SENDER')

mail = Mail(app)

@app.route('/')
def home():
    return render_template('home.html')  

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/howtouse')
def howtouse():
    return render_template('howtouse.html')

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        message = request.form['message']

        msg = Message(subject=f"New Contact Form Submission from {name}",
                      sender=email,
                      recipients=['kynethruizo@gmail.com']) 
        msg.body = f"Name: {name}\nEmail: {email}\n\nMessage:\n{message}"
        try:
            mail.send(msg)
            flash("Your message has been sent successfully!", "success")
        except Exception as e:
            flash(f"Error sending message: {e}", "danger")

        return redirect(url_for('contact'))

    return render_template('contact.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=os.getenv("PORT", 8000))
