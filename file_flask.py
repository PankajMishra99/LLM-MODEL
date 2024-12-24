import flask
from flask import Flask,render_template,url_for,request,Response,session,redirect,flash
import os
import hashlib
import pymongo
from werkzeug.middleware.proxy_fix import ProxyFix
import logging
import json


with open ("param.json",'r') as file:
    config = json.load(file)


config_db = config["mongodb"]["uri"]
config_username = config["mongodb"]["username"]
config_password = config["mongodb"]["password"]
config_database = config["mongodb"]["database"]
config_collection = config["mongodb"]["collection"]
config_timeout_ms = config["mongodb"]["server_selection_timeout_ms"]
config_source = config["mongodb"]["auth_source"]
config_replica = config["mongodb"]["replica_set"]

# for flask app..
config_host = config["app"]["flask_config"]["host"]
config_port = config["app"]["flask_config"]["port"]


key =os.getenv('secrets_key')
app = Flask(__name__,template_folder="templates")
app.secret_key=key
mongo_uri = os.getenv('mongo_uri')
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_port=1)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Increase if necessary




if not key:
    raise ValueError("Secret key not set in the environment variables..")
app.secret_key=key


if not mongo_uri:
    raise ValueError("Mongodb uri not set in the enviroment variables")
try:
    client = pymongo.MongoClient(mongo_uri)
    print("connected to Mongodb")
    db = client[config_database]             # will replace by config_database
    user_collection = db[config_collection]   # same as ..
    print("Database and collection accessible")
except pymongo.errors.ConnectionError as e:
    print(f"Failed to connect to MongoDB: {e}")
    raise



# print(mongo_uri)
# hasshing password for the security purpose..
def hash_password(password:str=config_password)->str:
    return hashlib.sha256(password.encode('utf-8')).hexdigest()


def verify_password(stored_password:str=config_password,provided_password:str=config_password)->str:
    return hashlib.sha256(provided_password.encode('utf-8')).hexdigest()==stored_password


@app.route('/')
def home():
    if "username" in session:
        return redirect('chatbot')
    else:
        return redirect('login')
    
    

@app.route('/chatbot', methods=['GET', 'POST'])
def chatbot():
    logging.basicConfig(level=logging.DEBUG)
    
    if 'username' not in session:
        flash("You need to login first", "error")
        return render_template('login.html')

    answer = ""
    if request.method == 'POST':
        question = request.form.get('question')
        logging.debug(f"Question received: {question}")
        
        if question:
            try:
                # Get the file and question from the form
                file = request.files.get("file")
                file_type = request.form.get("file_type")
                logging.debug(f"Receved_file : {file} and File Type : {file_type}")
                
                # Process the file and generate an answer if a file and question are provided
                if  question:
                    from model1 import main_text,qa_chain_function,flask_chat
                    # extract_text = main_text(file, file_type)  # Assuming this is the file processing function
                    # logging.debug(f"Extracted text : {extract_text}")
                    answer = qa_chain_function(question, 'llama3' )  # Replace this with your actual logic
                    # answer = list(set(answer))
                    logging.debug(f"Answer generated: {answer}")
                    return render_template('chatbot.html', username=session['username'], answer=answer)


            except Exception as e:
                print(f"Error: {e}")
                answer = "An error occurred. Please try again later."
    
    return render_template('chatbot.html', answer=answer)

    
@app.route('/register',methods = ['POST','GET'])
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
        if user and verify_password(user['Password'],password):
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



if __name__== "__main__":
    app.run(debug=True,host=config_host,port=config_port)




# print(client.list_database_names())

        







