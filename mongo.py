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





def register_user(first_name,last_name,username,password,confim_password):
        client = pymongo.MongoClient(mongo_uri,serverSelectionTimeoutMS=5000)
        db = client['db1']
        user_collection =db['collect1']
        hash_assword = hash_password(password)
        confim_password = hash_password(password)
        if user_collection.find_one({"username":username}):
            return 'username already exist..'
        user_collection.insert_one({"First Name" : first_name,
                                    "Last Name" : last_name,
                                    "username":username,
                                    "password":hash_assword,
                                    "Confirm Password":confim_password})
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
     option =st.selectbox("Chose Option",("Login","Register"),index=0)

     col1,col2 = st.columns(2)
     
     with col2:
          if option=="Register":
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
     with col1:
          if option == "Login":
               st.subheader("Login")
               username = st.text_input("Username")
               password = st.text_input("Password",type="password")
               if st.button("Login"):
                    result = user_login(username, password)
                    if result:
                         st.success("Login successful")
                    else:
                         st.error("Invalid credentials. Please try again.")

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
     
     

         





