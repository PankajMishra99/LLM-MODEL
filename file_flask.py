import flask
from flask import Flask,render_template,url_for,request,Response,session,redirect,flash
import os
from model import web_chat
import hashlib
import pymongo
key =os.getenv('secrets_key')
app = Flask(__name__,template_folder="templates")
app.secret_key=key
mongo_uri = os.getenv('mongo_uri')

if not key:
    raise ValueError("Secret key not set in the environment variables..")
app.secret_key=key


if not mongo_uri:
    raise ValueError("Mongodb uri not set in the enviroment variables")
try:
    client = pymongo.MongoClient(mongo_uri)
    print("connected to Mongodb")
    db = client['db1']
    user_collection = db['collect']
    print("Database and collection accessible")
except pymongo.errors.ConnectionError as e:
    print(f"Failed to connect to MongoDB: {e}")
    raise



# print(mongo_uri)
# hasshing password for the security purpose..
def hash_password(password:str)->str:
    return hashlib.sha256(password.encode('utf-8')).hexdigest()


def verify_password(stored_password:str,provided_password:str)->str:
    return hashlib.sha256(provided_password.encode('utf-8')).hexdigest()==stored_password


@app .route('/')
def home():
    if "username" in session:
        return redirect('chatbot')
    else:
        return redirect('login')
    
@app .route('/chatbot')
def chatbot():
    if 'username' not in session:
        flash("need to login first","error")
        return render_template('login')
    return render_template('chatbot.html',username=session['username'])
    
@app .route('/register',methods = ['POST','GET'])
def register():
    if request.method=='POST':
        first_name = request.form['first_name']
        last_name = request.form['last_name']
        username = request.form['username']
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        if password != confirm_password:
            flash("Password do not match !",'error')
            return redirect(url_for('register'))
        
        if user_collection.find_one({'username': username}):
            flash("Username already exists", 'error')
            return redirect(url_for('register'))
        
        hashed_password = hash_password(password)
        try:
            user_collection.insert_one({
                "First Name":first_name,
                "Last Name":last_name,
                "Username":username,
                "Password":hashed_password

            })
            flash("Registration successfull ! Please login","success")
            return redirect(url_for('login'))
        except Exception as e:
            flash(f"An error occured: {e}","error")
            return redirect(url_for('register'))
    return render_template("register.html")

@app.route('/login',methods=['GET', 'POST'])
def login():
    if request.method=='POST':
        username = request.form['username']
        password = request.form['password']

        user = user_collection.find_one({'Username': username})
        if user and verify_password(user['password'],password):
            session['username']= username
            flash("login successfull",'success')
            return redirect(url_for('chatbot'))
        else:
            flash("Invalied credentials","error")
    return render_template("login.html")


@app.route('/profle')
def profile():
    if 'username' in session:
        return f"Hello {session['username']} ! Welcome to your profile.."
    else:
        return redirect(url_for('login'))

@app.route('/logout')
def logout():
    session.pop("username",None)
    return redirect(url_for('login'))


@app.errorhandler(404)
def not_found(error):
    return "Page Not found !",404

if __name__=="__main__":
    app.run(debug=True)



# print(client.list_database_names())

        







