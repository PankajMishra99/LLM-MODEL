import pymongo
import os
import streamlit as st
import hashlib


mongo_uri = os.getenv('mongo_uri')

# Hashing password for the security purpose..
def hash_password(password:str)-> str:
     return hashlib.sha256(password.encode('utf-8')).hexdigest()

def verify_password(stored_password:str,provided_password:str):
     return hashlib.sha256(provided_password.encode('utf-8')).hexdigest()==stored_password





def register_user(username,password):
        client = pymongo.MongoClient(mongo_uri,serverSelectionTimeoutMS=5000)
        db = client['db1']
        user_collection =db['collect1']
        hash_assword = hash_password(password)
        if user_collection.find_one({"username":username}):
            return 'username already exist..'
        user_collection.insert_one({"username":username,
                                    "password":hash_assword})
        return "Registration successfull.."


def user_login(username, password):
    client = pymongo.MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
    db = client['db1']
    user_collection = db['collect1']
    
    # Check if the username and password match in the database
    user = user_collection.find_one({"username": username})
    if user and verify_password(user["password"],password):
        return "Login successful."
    else:
        return "Either username or password is wrong."

def login_page():
     st.title("Login Page")
     option =st.selectbox("Chose Option",("Login","Register"))
     username = st.text_input("Username")
     password = st.text_input("Password",type="password")
     
     if option=="Register":
          if st.button("Register"):
               st.write(register_user(username,password))
    
     if option=="Login":
          if st.button("Login"):
               st.write(user_login(username,password))
               print("Login sucessfull")

# if __name__=="__main__":
#      login_page()

# if __name__=="__main__":
#      print(hash_password('pankaj@1'))
# #     print("Welcome to Pymongo..")
# #     client = pymongo.MongoClient(mongo_uri)
# #     db=client['test1']
# #     user_collection = db['test_collection']
# #     insert2 = user_collection.insert_one({'name':"Ashmita",
# #                                           "title":"Mishra"
#                                           })
     
     

         





