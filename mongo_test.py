import pymongo
import os
import streamlit as st
import hashlib
import model
from model import web_chat


mongo_uri = os.getenv('mongo_uri')

# Hashing password for the security purpose..
def hash_password(password:str)-> str:
     return hashlib.sha256(password.encode('utf-8')).hexdigest()

def verify_password(stored_password:str,provided_password:str):
     return hashlib.sha256(provided_password.encode('utf-8')).hexdigest()==stored_password





def register_user(first_name,last_name,username,password,confim_password):
        client = pymongo.MongoClient(mongo_uri,serverSelectionTimeoutMS=5000)
        db = client['db1']
        user_collection =db['collect1']
        hash_assword = hash_password(password)
        confim_password = hash_password(password)
        if user_collection.find_one({"username":username}):
            return 'username already exist..'
        try:
            user_collection.insert_one({"First Name" : first_name,
                                        "Last Name" : last_name,
                                        "username":username,
                                        "password":hash_assword,
                                        "Confirm Password":confim_password})
            return "Registration successfull.."
        except Exception as e:
             return f"An error occurred: {e}"
      


def user_login(username, password):
    client = pymongo.MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
    db = client['db1']
    user_collection = db['collect1']
    
    # Check if the username and password match in the database
    try:
        user = user_collection.find_one({"username": username})
        if user and verify_password(user["password"],password):
            return "Login successful."
        else:
            return "Either username or password is wrong."
    except Exception as e:
         st.error(f"An error occurred during login: {e}")
         return False
    
def login_page():
     if "logged_in" in st.session_state and st.session_state["logged_in"]:
          st.experimental_rerun()
     st.title("Login Page")
    #  option =st.selectbox("Chose Option",("Login","Register"),index=0)
     st.markdown(
          """
            <style>
        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 20px;
            background-color: #f4f4f4;  /* Background color */
        }
        .right-align{
            text-align: right;
        }
        .center-content{
            display: flex;
            justify-content: center;
            align-items: center;
            margin-top: 20px;
        }
        .from-container{
            width: 80%;
            padding: 20px;
            border-radius: 10px;
            background-color: darkgrey;
        }
        .button{
            width: 100%;
            padding: 10px;
            background-color: #4CAF50;
            color: white;
            border-radius: 5px;
            cursor: pointer;
        }
        .button:hover{
            background-color: cadetblue;
        }
    </style>
    <a href="/app.py" target="_self">
                    <button style="background-color:green; color:white; padding:10px; border:none; border-radius:5px; cursor:pointer;">
                        Go to app
                    </button>
                </a>
          """,
          unsafe_allow_html=True
    )
     st.markdown('<div class="center-content">',unsafe_allow_html=True)
    #  col1,col2 = st.columns(2)
     option =st.selectbox("Chose Option",("Login","Register"),index=0)
     
     if option=="Register":
        st.markdown('<div class="form-content">',unsafe_allow_html=True)
        st.subheader("Register")
        first_name = st.text_input("First Name")
        last_name = st.text_input("Last Name")
        username = st.text_input("Username")
        password = st.text_input("Password",type="password")
        confirm_password = st.text_input("Confirm Password", type="password", key="confirm_password")  # Key assigned
        if st.button("Register"):
                if password == confirm_password:
                    st.write(register_user(first_name,last_name,username,password,confirm_password))
                else:
                    st.error("Password do not match..")
        st.markdown('</div>',unsafe_allow_html=True)
     if option=="Login":
        st.markdown('<div class="center-content">',unsafe_allow_html=True)  
        st.subheader("Login")
        username = st.text_input("Username")
        password = st.text_input("Password",type="password")
        if st.button("Login"):
            result = user_login(username, password)
            if result:
                    st.success("Login successful")
            else:
                    st.error("Invalid credentials. Please try again.")
        st.markdown('</div>',unsafe_allow_html=True)
     st.markdown('</div>',unsafe_allow_html=True)

     if 'logged_in' not in st.session_state:
          st.session_state['logged_in'] = False
     if st.session_state['logged_in']:
        web_chat()
     else:
          user_login(username, password)

if __name__=="__main__":
     login_page()
