import os
import streamlit as st
import re
import ollama
import pickle
import base64
import numpy as np
from pathlib import Path
from fpdf import FPDF
from PIL import Image
import hashlib
import time

# Password hashing function
st.set_page_config(
    page_title="Mental Health Support",  # Title of the page
    page_icon="🧠",  # Icon for the page
    layout="wide",  # Center the content on the page
    initial_sidebar_state="collapsed"
)


# Load models
try:
    model = pickle.load(open("df1.pkl", 'rb'))
    svc = pickle.load(open("svc.pkl", 'rb'))
    model1 = pickle.load(open("df.pkl", 'rb'))
    svc1 = pickle.load(open("svc1.pkl", 'rb'))
    model2 = pickle.load(open("df2.pkl", 'rb'))
    GB = pickle.load(open("GB.pkl", 'rb'))
except FileNotFoundError as e:
    st.error(f"Model file not found: {e}")
    st.stop()
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# User database
USER_DB = Path("user_db1.pkl")
if not USER_DB.exists():
    with open(USER_DB, 'wb') as f:
        pickle.dump({}, f)

def authenticate(username, password):
    # Load users from pickle file
    try:
        with open(USER_DB, "rb") as f:
            users = pickle.load(f)  # Expected to be a list of dictionaries
    except FileNotFoundError:
        print(f"User database '{USER_DB}' not found.")
        return False
    except Exception as e:
        print(f"Error loading user database: {e}")
        return False

    # Convert the list of users to a dictionary {username: hashed_password}
    user_dict = {user["username"]: user["password"] for user in users}

    # Check if username exists and password matches
    return user_dict.get(username) == hash_password(password)

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


def save_to_pickle(data):
    # Check if the pickle file exists
    if os.path.exists(USER_DB):
        # Load existing data
        with open(USER_DB, "rb") as file:
            try:
                existing_data = pickle.load(file)  # Load existing user data
            except EOFError:
                existing_data = []  # File exists but is empty
    else:
        existing_data = []  # No file exists, initialize an empty list

    # Ensure existing data is a list
    if not isinstance(existing_data, list):
        existing_data = []

        # Add new user to data
    existing_data.append(data)

    # Save updated data back to pickle file
    with open(USER_DB, "wb") as file:
        pickle.dump(existing_data, file)

# Function to check if username already exists
def is_username_taken(username):
    if os.path.exists(USER_DB):
        with open(USER_DB, "rb") as file:
            existing_data = pickle.load(file)
        return any(user["username"] == username for user in existing_data)
    return False

def is_valid_email(email):
    """Validate an email address using regex."""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None



def preprocess_input(name, Country, mobile_no,landline_no,Ocupation,self_employed, age, academic_pressure, cgpa, study_satisfaction, study_hours, gender, sleep_duration, dietary, suicide, family_illness):
    gender_encoded = 0 if gender == 'Male' else 1

    dietary_encoded = {
        'Healthy': 1,
        'Moderate': 2,
        'Unhealthy': 3,
        'Others': 4
    }.get(dietary, 4)

    suicide_encoded = 1 if suicide == 'Yes' else 0
    family_illness_encoded = 1 if family_illness == 'Yes' else 0

    return np.array([age, academic_pressure, cgpa, study_satisfaction,
                     study_hours, gender_encoded, sleep_duration,
                     dietary_encoded, suicide_encoded, family_illness_encoded]).reshape(1, -1)

def preprocess_input1(name,age,mobile_no,landline_no,Gender,Country,Ocupation,self_employed,family_history,Days_Indoors,Growing_Stress,Changes_Habits,Mental_Health_History,Mood_Swings,Coping_Struggles,Work_Interest,Social_Weakness,mental_health_interview,care_options):
    """
    Function to preprocess the input into the format expected by the model.
    """
    gender = 0 if Gender == 'Male' else 1
    
    country = {
        "United States": 1,
        "Poland": 2,
        "Australia": 3,
        "Canada": 4,
        "United": 5,
        "Kingdom": 6,
        "South Africa": 7,
        "New Zealand": 8,
        "Netherlands": 9,
        "India": 10,
        "Belgium": 11,
        "Ireland": 12,
        "France": 13,
        "Portugal": 14, 
        "Brazil": 15,
        "Costa Rica": 16,
        "Russia": 17, 
        "Germany": 18,
        "Switzerland": 19,
        "Finland": 20,
        "Israel": 21,
        "Italy": 22,
        "Bosnia and Herzegovina": 23,
        "Singapore": 24,
        "Nigeria": 25,
        "Croatia": 26, 
        "Thailand": 27,
        "Denmark": 28,
        "Mexico": 29,
        "Greece": 30,
        "Moldova": 31, 
        "Colombia": 32,
        "Georgia": 33,
        "Czech Republic": 34,
        "Philippines": 35
    }.get(Country, 35)

    ocupation = {
        'Corporate': 1,
        'Student': 2,
        'Business': 3,
        'Housewife': 4,
        'Others': 5
    }.get(Ocupation, 5)

    Self_employed = {
        'No' : 0,
        'Yes': 1,
        'Nan': 2,
    }.get(self_employed, 2)

    Family_history = 0 if family_history == 'NO' else 1

    # gender = 0 if Gender == 'Male' else 1
    
    days_Indoors = {
        '1-14 days': 0,
        'Go out Every day': 1,
        'More than 2 months':2,
        '15-30 days':3,
        '31-60 days': 4
    }.get(Days_Indoors, 4)

    growing_Stress = {
        'No' : 0,
        'Yes': 1,
        'Maybe': 2,
    }.get(Growing_Stress, 2)

    changes_Habits = {
        'No' : 0,
        'Yes': 1,
        'Maybe': 2,
    }.get(Changes_Habits, 2)

    mental_Health_History = {
        'No' : 0,
        'Yes': 1,
        'Maybe': 2,
    }.get(Mental_Health_History, 2)

    mood_Swings = {
        'Medium' : 0,
        'Low': 1,
        'High': 2,
    }.get(Mood_Swings, 2)

    coping_Struggles = 0 if Coping_Struggles == 'NO' else 1

    mental_Health_History = {
        'No' : 0,
        'Yes': 1,
        'Maybe': 2,
    }.get(Mental_Health_History, 2)

    work_Interest = {
        'No' : 0,
        'Yes': 1,
        'Maybe': 2,
    }.get(Work_Interest, 2)

    social_Weakness = {
        'No' : 0,
        'Yes': 1,
        'Maybe': 2,
    }.get(Social_Weakness, 2)
    
    Mental_health_interview = {
        'No' : 0,
        'Yes': 1,
        'Maybe': 2,
    }.get(mental_health_interview, 2)
    
    Care_options = {
        'No' : 0,
        'Yes': 1,
        'Not sure': 2,
    }.get(care_options, 2)
    

    return np.array([gender, country, ocupation, Self_employed,
                     Family_history, days_Indoors, growing_Stress,
                     changes_Habits, mental_Health_History, mood_Swings,coping_Struggles,
                     work_Interest,social_Weakness,Mental_health_interview,Care_options,
                    ]).reshape(1, -1)


def main():
    if st.session_state.page == "Landing":
        landing_page()
    elif st.session_state.page == "Signup":
        sign_up_page()
    elif st.session_state.page == "Login":
        login_page()
    elif st.session_state.page == "Home":
        app_pages()
    elif st.session_state.page == "depress":
        depress()
    elif st.session_state.page == "meditate":
        meditate()
    elif st.session_state.page == "breath":
        breath()
    elif st.session_state.page == "Report":
        Report()
    elif st.session_state.page == "Report1":
        Report1()
    elif st.session_state.page == "Report2":
        Report2()


if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if "page" not in st.session_state:
    st.session_state.page = "Landing"

# Toggle test visibility state
if "show_student_test" not in st.session_state:
    st.session_state.show_student_test = False
if "show_working_test" not in st.session_state:
    st.session_state.show_working_test = False
if "show_common_for_all_test" not in st.session_state:
    st.session_state.show_common_for_all_test = False

def landing_page():
    # Custom CSS for enhanced UI
    st.markdown("""
        <style>
        body {
            background-color: #f5f;
        }
        .title {
            font-size: 3em;
            font-weight: bold;
            color: #2E8B57;
            text-align: center;
            margin-top: 30px;
        }
        .subheader {
            font-size: 1.5em;
            font-weight: normal;
            text-align: center;
            color: #333;
            margin-bottom: 20px;
        }
        .section {
            text-align: center;
            margin-top: 30px;
            padding: 20px;
            background: #ffffff;
            border-radius: 15px;
            box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
        }
        .feature-box {
            display: flex;
            align-items: center;
            justify-content: center;
            background: #ffffff;
            border-radius: 15px;
            box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
            padding: 20px;
            margin: 20px 0;
        }
        .feature-img {
            width: 20%;
            border-radius: 10px;

            display: block;
            margin:auto;
        }
        .feature-text {
            width: 40%;
            text-align: left;
            font-size: 1.2em;
            color: black;
        }
        .stButton>button {
            width: 100%;
            padding: 12px;
            font-size: 16px;
            font-weight: bold;
            color: white;
            background: #4CAF50;
            border-radius: 25px;
            transition: all 0.3s ease-in-out;
        }
        .stButton>button:hover {
            background: #45a049;
            transform: scale(1.05);
        }
        </style>
    """, unsafe_allow_html=True)

    # Title
    st.markdown('<h1 class="title">🤗 CompanionMind - Your AI for Mental Well-being</h1>', unsafe_allow_html=True)
    st.markdown('<h2 class="subheader">Mental Health Matters</h2>',unsafe_allow_html=True)

    # Introduction Section
    st.markdown("""
        <div class="section">
        <p style="font-size: 1.2em; color: black;">
            Mental health is just as important as physical health. It
                affects how we think, feel, and act. Taking care of our well-being
            helps us build resilience, manage stress, and lead a more balanced
            life.
        </p>
        </div>
    """, unsafe_allow_html=True)

    # Features Section
    st.markdown('<h2 class="subheader">🌟 Key Features</h2>',unsafe_allow_html=True)

    # 1️⃣ Depression Assessment
    col1, col2 = st.columns([1, 1])
    with col1:
        st.image("depression_test.png", width=300)  # Replace with
    # actual image path
    with col2:
        st.markdown("""
            ### 🧠 Depression Assessment
            Our AI-powered assessment helps users understand their
            mental state through a series of scientifically designed questions.
            **💡 Personalized Recommendations** are provided based on
            the responses.
        """)

    # 2️⃣ AI Chatbot for Support
    col3, col4 = st.columns([1, 1])
    with col3:
        st.markdown("""
            ### 🤖 AI-Powered Chatbot
            Our **AI-powered chatbot** is always available to
            **provide emotional support, answer queries, and guide users toward
            positive coping mechanisms**.
                ⌨️ Engages in **empathetic conversations** to help
                users feel heard.
                😊 Offers **mental health tips and self-care
                recommendations**.
        """)
    with col4:
        st.image("chatbot.png")  # Replace with actual image path

    # 3️⃣ Mindfulness Meditation
    col5, col6 = st.columns([1, 1])
    with col5:
        st.image("meditation.png")  # Replace with actual image path
    with col6:
        st.markdown("""
            ### 🧘‍♀️ Mindfulness Meditation
            Experience **guided meditation** with soothing background
            music to help relax the mind and improve focus.
            **🎵 Breathing techniques and relaxation exercises** are included.
        """)

    # 4️⃣ Breathing Exercises
    col7, col8 = st.columns([1, 1])
    with col7:
        st.markdown("""
            ### 🌬️ Breathing Exercises
            Our guided breathing exercises help **reduce anxiety,
            increase relaxation, and improve sleep quality**.
            **📊 Track your progress over time.**
        """)
    with col8:
        st.image("breathing.png")  # Replace with actual image path


    # Login & Signup Buttons
    st.markdown('<h2 class="subheader">🔑 Get Started</h2>',unsafe_allow_html=True)
    colA, colB, colC = st.columns([1, 2, 1])

    with colB:
        colX, colY = st.columns(2)

    with colX:
        if st.button("Login"):
            st.session_state.page = "Login"
            st.rerun()

    with colY:
        if st.button("Sign Up"):
            st.session_state.page = "Signup"
            st.rerun()

def get_base64_image(image_path):
    with open(image_path, "rb") as file:
        encoded = base64.b64encode(file.read()).decode()
    return f"data:jpeg;base64,{encoded}"


def add_custom_css():
    st.markdown("""
    <style>
        body {
            text-align: center;
            background: linear-gradient(to right, #ff7e5f, #feb47b);  /* Gradient background */
            font-family: 'Poppins', sans-serif;
        }
        .stButton>button {
            text-align: center;
            background-color: white;
            color: black;
            border-radius: 20%;
            font-size: 12px;
            transition: 0.3s ease;
        }
        .stButton>button:hover {
            text-align: center;
            background-color: cyan;    
        }
        .stTextInput>div>div>input {
            color: black; /* Ensures text in the input is black */
            text-align: center;
            background-color: white;
            padding: 10px;
            border-radius: 10px;
            border-color: black;
            border: 1px solid black;
            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
            transition: 0.3s;
        }
        .stTextInput>div>div>input:focus {
            color: black; /* Maintain black text on focus */
            text-align: center;
            border-color: black;
            box-shadow: 0 0 0 2px rgba(255, 111, 97, 0.4);
        }
        .stTextInput>div>div>input:hover {
            color: black; /* Maintain black text on hover */
            text-align: center;
            border-color: white;
            box-shadow: 0 0 0 2px rgba(255, 111, 97, 0.4);
        }
        .Title {
            text-align: center;
            font-size: 38px;
            color: black;
            font-weight: bold;
            padding-top: 50px;
        }
        div[data-testid="stTextInput"] label {
            color: black !important;
        }
        .Subheader {
            text-align: center;
            display: center;
            font-size: 18px;
            color: black;
            font-weight: 500;
            margin-bottom: 30px;
        }
        .form-container {
            display: center;
            text-align: center;
            background-color: black;
            padding: 40px;
            border-radius: 2px;
            box-shadow: 0px 10px 20px rgba(0, 0, 0, 0.1);
            max-width: 500px;
            margin: 0 auto;
        }
        .stForm {
            text-align: center;
            display: center;
            border-color: black;
            flex-direction: row;
            align-items: center;
        }
        .form-container input {
            color: black; /* Ensures text in input is black */
            text-align: center;
            margin-bottom: 20px;
        }
        .white {
            color: black;
        }
        div[data-testid="stForm"] {
            border: 2px solid black !important;  /* Change 'red' to any color */
            border-radius: 10px;  /* Optional: Round the corners */
            padding: 10px;  /* Optional: Add some padding */
        }
    </style>
    """, unsafe_allow_html=True)

# Define the sign-up page function
def sign_up_page():
    image_base64 = get_base64_image("login.jpg")  # Replace with your local image path
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("{image_base64}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

    col1,col2,col3 = st.columns(3)
    with col1:
        # Apply custom CSS styling
        add_custom_css()
        
        # Create a form for sign-up inputs
        with st.form("signup_form"):
            st.markdown("<div class='Title'>New Here? Sign Up Now!</div>", unsafe_allow_html=True)
            st.markdown(f"<div style ='color : black'>_____________________________________________________</div>",unsafe_allow_html=True)

            # Adding fields to the form
            # fname = st.text_input("First Name", max_chars=50, placeholder="Enter your first name")
            # lname = st.text_input("Last Name", max_chars=50, placeholder="Enter your last name")
            username = st.text_input("Username", max_chars=20, placeholder="Choose a unique username")
            email = st.text_input("Email", placeholder="example@gmail.com")
            password = st.text_input("Password", type="password", placeholder="Enter a secure password")
            confirm_password = st.text_input("Confirm Password", type="password", placeholder="Re-enter your password")
            
            submit_button = st.form_submit_button("🚀 Sign Up")
            st.markdown("<div class='white'>Already have an account?</div>", unsafe_allow_html=True)
            login_page = st.form_submit_button("Login")

        if submit_button:
            # Validation checks
            if not (email and username and password):
                st.error("❗ All fields are required!")
            elif password != confirm_password:
                st.error("❗ Passwords do not match!")
            elif is_username_taken(username):  # Assuming you have a function to check if the username is already taken
                st.error("❗ Username already exists. Please choose another.")
            elif not is_valid_email(email):  # Assuming you have a phone number validation function
                st.error("❗ Invalid Email Id! Please enter in the format: example@gmail.com")
            else:
                # Hash the password and save the data
                user_data = {
                    "User_Name": username,
                    "Email Id": email,
                    "username": username,
                    "password": hash_password(password)  # Make sure to hash the password before storing
                }
                save_to_pickle(user_data)  # Save user data to your storage
                st.success("✅ Sign up successful! Your data has been saved.")

                # Redirect to login page after successful registration
                st.session_state.page = "Login"
                st.rerun()  # Call the login page function (replace with your actual login page)
        if login_page:
            st.session_state.page = "Login"
            st.rerun()

    with col2:
        co1,co2,col3,col4,col5,col6 = st.columns(6)
        with co1:
            if st.button("🔙"):
                st.session_state.page = "Landing"
                st.rerun()

def login_page():
    image_base64 = get_base64_image("login.jpg")  # Replace with your local image path
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("{image_base64}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
    col1,col2,col3 = st.columns(3)
    with col1:
        add_custom_css()
        # Create a form for sign-up inputs
        with st.form("login_form"):

            st.markdown("<div class='Title'>Sign-in</div>", unsafe_allow_html=True)
            st.markdown(f"<div style ='color : black'>_____________________________________________________</div>",unsafe_allow_html=True)

            username = st.text_input("Username")
            password = st.text_input("Password", type="password")

            submit_button = st.form_submit_button("Login")
            st.markdown("<div class='white'>Don't have an account?</div>", unsafe_allow_html=True)
            sign_up_page = st.form_submit_button("Sign Up")

        if submit_button:
            if authenticate(username, password):
                st.session_state.authenticated = True
                st.session_state.username = username
                st.success(f"Login successful! Welcome, {username}!")
                st.session_state.page = "Home"
            elif username == "":
                st.error("Please enter a username")
            elif password == "":
                st.error("Please enter a Password.")
            else:
                st.error("Invalid username or password.")
        if sign_up_page:
            st.session_state.page = "Signup"
            st.rerun()
    with col2:
        col1,col2,col3,col4 = st.columns(4)
        with col1:
            st.markdown("   ")
        with col2:
            st.markdown("   ")
        with col1:
            if st.button("🔙"):
                st.session_state.page = "Landing"
                st.rerun()
        with col4:
            st.markdown("   ")
def add_custom_css11():
    st.markdown("""
    <style>
        body {
            text-align: center;
            background: linear-gradient(to right, #ff7e5f, #feb47b);  /* Gradient background */
            font-family: 'Poppins', sans-serif;
        }
        .stButton>button {
            text-align: center;
            background-color: linear-gradient(94.5deg, #F7A70D 0%, #FACA6E 73.52%, #F7A70D 106.59%);
            color: black;
            border-radius: 10%;
            font-size: 12px;
            transition: 0.3s ease;
            }
        .stButton>button:hover {
            text-align: center;
            background-color: lightgreen;
                
        }
        .stTextInput>div>div>input {
            text-align: center;
            background-color: #fff;
            padding: 10px;
            border-radius: 10px;
            border-color: white;
            border: 1px solid #ff6f61;
            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
            transition: 0.3s;
        }
        .stTextInput>div>div>input:focus {
            text-align: center;
            border-color: #ff6f61;
            box-shadow: 0 0 0 2px rgba(255, 111, 97, 0.4);
        }
        .stTextInput>div>div>input:hover {
            text-align: center;
            border-color: #ff6f61;
            box-shadow: 0 0 0 2px rgba(255, 111, 97, 0.4);
        }
        .Title {
            text-align: center;
            text-align: center;
            font-size: 38px;
            color: black;
            font-weight: bold;
            padding-top: 50px;
        }
        .para{
            text-align: center;
            text-align: center;
            font-size: 15px;
            color: black;
            font-weight: bold;
            padding-top: 10px;
        }
        div[data-testid="stTextInput"] label {
            color: white !important;
        }
        .Subheader {
            text-align: center;
            display: center;
            font-size: 18px;
            color: black;
            font-weight: 500;
            margin-bottom: 30px;
        }
        .form-container {
            display: center;
            text-align: center;
            background-color: black;
            padding: 40px;
            border-radius: 5px;
            box-shadow: 0px 10px 20px rgba(0, 0, 0, 0.1);
            max-width: 500px;
            margin: 0 auto;
        }
        .stForm {
            text-align: center;
            display: flex;
            border-color : black;
            left-padding : 20px;
            flex-direction: row;
            align-items: center;
        }
        .stForm:hover {
            background-color: #e0f7fa; /* Light cyan color */
            box-shadow: 0 6px 12px rgba(0, 0, 0, 0.2); /* Darker shadow */
        }
        div[data-testid="stForm"] {
            border: 2px solid black !important;  /* Change 'red' to any color */
            border-radius: 40px;  /* Optional: Round the corners */
            box-shadow: 0px 10px 20px rgba(0, 0, 0, 0.1);
        }
    </style>
    """, unsafe_allow_html=True)
def decor():
    """
    <style>
    .card {
        background-color: #f0f0f5;
        padding: 20px;
        margin: 10px;
        border-radius: 10px;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
        transition: box-shadow 0.3s ease, transform 0.3s ease;
    }

    .card:hover {
        box-shadow: 0px 10px 20px rgba(0, 0, 0, 0.2);
        transform: translateY(-5px);
    }

    .card-header {
        font-size: 1.5em;
        font-weight: bold;
        margin-bottom: 10px;
    }

    .card-body {
        font-size: 1em;
        color: #555;
    }
    .card-header1 {
        font-size: 1.5em;
        font-weight: bold;
        margin-bottom: 10px;
    }

    .card-body1 {
        font-size: 1em;
        color: #555;
        padding : 20px;
    }

    </style>
    """
def depress():
    with st.container():
        col1,col2,col3,col4,col5,col6 = st.columns(6)
        with col6:
            col7,col8,col9 = st.columns(3)
            with col9:
                if st.button("🔙"):
                    st.session_state.page = "Home"
                    st.rerun()
    st.header("Depression Test")

    c1,c2,col1,c4,col2,c5,col3  = st.columns(7)

    with col1:
        if st.button("Student"):
            st.session_state.show_student_test = not st.session_state.show_student_test
            st.session_state.show_working_test = False
            st.session_state.show_common_for_all_test = False
            st.rerun()
            
    with col2:
        if st.button("Working/Business"):
            st.session_state.show_working_test = not st.session_state.show_working_test
            st.session_state.show_student_test = False
            st.session_state.show_common_for_all_test = False
            st.rerun()
            
    
    with col3:
        if st.button("Common_for_all"):
                st.session_state.show_common_for_all_test = not st.session_state.show_common_for_all_test
                st.session_state.show_student_test = False
                st.session_state.show_working_test = False
                st.rerun()

    if st.session_state.show_student_test:
        std_depression_test()

    if st.session_state.show_working_test:
        Work_depression_test()
    
    if st.session_state.show_common_for_all_test:
        Common_for_all_depression_test()

def meditate():
    with st.container():
        col1,col2,col3,col4,col5,col6 = st.columns(6)
        with col6:
            col7,col8,col9 = st.columns(3)
            with col9:
                add_custom_css11()
                if st.button("🔙"):
                    st.session_state.page = "Home"
                    st.rerun()
    st.header("Meditation")
    def initialize_timer(minutes):
        st.session_state['remaining_time'] = minutes * 60
        st.session_state['is_running'] = True

    def countdown_timer(timer_placeholder):
        while st.session_state['remaining_time'] > 0 and st.session_state['is_running']:
            mins_left, secs_left = divmod(st.session_state['remaining_time'], 60)
            timer_placeholder.markdown(f"<h1 style='font-size: 100px; text-align: center;'> 🧘🏻 {mins_left:02}:{secs_left:02} 🧘🏻</h1>", unsafe_allow_html=True)
            time.sleep(1)
            st.session_state['remaining_time'] -= 1
            if st.session_state['remaining_time'] <= 0:
                st.session_state['is_running'] = False
                timer_placeholder.markdown(f"<h1 style='font-size: 60px; text-align: center;'>You did a Great Job! 🥳", unsafe_allow_html=True)

    
    st.markdown(
            """
            <div style="text-align: center; font-size: 40px; color:rgb(2, 135, 252); font-weight: bold">
                🧘🏻 Meditation Timer 🧘🏻
            </div>
            """, unsafe_allow_html=True)


    minutes = st.number_input("Enter the number of minutes for your meditation session:", min_value=1, max_value=120, value=5)

    if 'remaining_time' not in st.session_state:
        st.session_state['remaining_time'] = 0
        st.session_state['is_running'] = False

    timer_placeholder = st.empty()
    btn1, btn2 = st.columns(2)
    audio_files = [ "audio/audio_1.mp3",
                    "audio/audio_2.mp3",
                    "audio/audio_3.mp3",
                    "audio/audio_4.mp3",
                    "audio/audio_5.mp3"
                    ]
    with btn1:
        if st.button("Start Meditation Timer") and not st.session_state['is_running']:
            st.session_state['is_running'] = True   
            
            selected_audio = audio_files[1]
            audio_folder = "audio"
            selected = os.path.join(audio_folder, "audio_1.mp3")
            
            st.audio(selected_audio, format="audio/mp3", start_time=0, autoplay=True, loop=True)  # Plays the audio
            st.markdown(f"""
                <audio autoplay loop>
                    <source src="{selected}" type="audio/mp3">
                </audio>
            """, unsafe_allow_html=True)
            audio_player = st.empty()
            html_string = f"""<audio autoplay loop src="{selected}" type="audio/mp3"></audio>"""
            audio_player.markdown(html_string, unsafe_allow_html=True) 
            st.html(f"""<audio autoplay loop src="{selected}" type="audio/mp3"></audio>""")
            initialize_timer(minutes)
            countdown_timer(timer_placeholder)
            st.balloons()

    with btn2:
        if st.button("Reset Timer") and st.session_state['is_running']:
            st.session_state['is_running'] = False
            st.session_state['remaining_time'] = 0
            timer_placeholder.markdown(f"<h1 style='font-size: 70px; text-align: center;'>Timer Reset⏳", unsafe_allow_html=True)

def breath():
    with st.container():
        col1,col2,col3,col4,col5,col6 = st.columns(6)
        with col6:
            col7,col8,col9 = st.columns(3)
            with col9:
                add_custom_css11()
                if st.button("🔙"):
                    st.session_state.page = "Home"
                    st.rerun()

    def countdown(emoji_display):
        for i in range(3, 0, -1):
            emoji_display.markdown(f"""
            <div style="text-align: center; font-size: 80px; font-weight: bold;">
            {i}
            </div>
            """, unsafe_allow_html=True)
            time.sleep(1)
            
    def countdown4(emoji_display):
        for i in range(4, 0, -1):
            emoji_display.markdown(f"""
            <div style="text-align: center; font-size: 80px; font-weight: bold;">
            {i}
            </div>
            """, unsafe_allow_html=True)
            time.sleep(1) 
            
    def countdown7(emoji_display):
        for i in range(7, 0, -1):
            emoji_display.markdown(f"""
            <div style="text-align: center; font-size: 80px; font-weight: bold;">
            {i}
            </div>
            """, unsafe_allow_html=True)
            time.sleep(1) 
            
    def countdown8(emoji_display):
        for i in range(8, 0, -1):
            emoji_display.markdown(f"""
            <div style="text-align: center; font-size: 80px; font-weight: bold;">
            {i}
            </div>
            """, unsafe_allow_html=True)
            time.sleep(1)  

    def breathing_phase(phase, duration, emoji_display):
        phase_names = {
            "breathe_in": "Breathe In",
            "hold": "Hold",
            "exhale": "Exhale"
        }
        colors = {
            "breathe_in": "#4CAF50",  # Green for breathing in
            "hold": "#FF9800",         # Orange for holding
            "exhale": "#F44336"        # Red for exhaling
        }

        
        emoji_display.markdown(f"""
        <div style="text-align: center; font-size: 60px; font-weight: bold; color: {colors[phase]};">
        {phase_names[phase]}
        
        </div>
        """, unsafe_allow_html=True)
        
        time.sleep(1)
        
        if duration == 4:
            countdown4(emoji_display)
        elif duration == 7:
            countdown7(emoji_display)
        elif duration == 8:
            countdown8(emoji_display)
        
        time.sleep(1)
        

    def start_breathing_cycle():
        status_text = st.empty()
        emoji_display = st.empty()  

        for cycle in range(1, 6):  # 5 cycles
            status_text.markdown(f"""
            <div style="text-align: center; font-size: 30px; font-weight: bold;">
            Cycle {cycle}
            </div>
            """, unsafe_allow_html=True)
            countdown(emoji_display)
            breathing_phase("breathe_in", 4, emoji_display)
            breathing_phase("hold", 7, emoji_display)
            breathing_phase("exhale", 8, emoji_display)
        st.markdown("<h3 class='celebration' style='text-align: center'>🎉 Great Job! You completed the exercise! 🎉</h3>", unsafe_allow_html=True)
        st.balloons()

    # Streamlit page layout
    st.markdown(
        """
        <div style="text-align: center; font-size: 36px; color: #4CAF50; font-weight: bold;">
        🌿 Breathing Exercise for Stress Relief 🌿
        </div>
        """, unsafe_allow_html=True)

    st.markdown(
        """
        <div style="text-align: center; font-size: 20px; color: #6C757D;">
            Follow along with this simple exercise to reduce stress and feel more relaxed.
        </div>
        """, unsafe_allow_html=True)

    st.markdown(
        """
        <div style="display:flex; justify-content:center">
        <div style=" font-size: 18px; color: #007BFF;">
            1. Breathe in for 4 seconds.<br>
            2. Hold your breath for 7 seconds.<br>
            3. Exhale slowly for 8 seconds.<br>
            Repeat for 5 cycles. Let's begin... <br><br>
        </div>
        </div>
        """, unsafe_allow_html=True)

    if st.button("Start Exercise", key="start_exercise", use_container_width=True):
        start_breathing_cycle()

def st_form():
    st.markdown("""
    <style>
        body {
            text-align: center;
            background: linear-gradient(to right, #ff7e5f, #feb47b);  /* Gradient background */
            font-family: 'Poppins', sans-serif;
        }
        .stButton>button {
            text-align: center;
            background-color: linear-gradient(94.5deg, #F7A70D 0%, #FACA6E 73.52%, #F7A70D 106.59%);
            color: black;
            border-radius: 10%;
            font-size: 12px;
            transition: 0.3s ease;
            }
        .stButton>button:hover {
            text-align: center;
            background-color: #e0f7fa; /* Light cyan color */
        }
        .stForm {
            text-align: center;
            display: flex;
            border-color : black;
            left-padding : 20px;
            flex-direction: row;
            align-items: center;
        }
        div[data-testid="stForm"] {
            background-color: #e0f7fa; /* Light cyan color */
            padding: 20px;
            margin: 10px;
            border-radius: 10px;
            box-shadow: 0px 10px 20px rgba(0, 0, 0, 0.2);
            transition: box-shadow 0.3s ease, transform 0.3s ease;
            transform: translateY(-10px);
            border: 2px solid black !important;  /* Change 'red' to any color */
            border-radius: 40px;  /* Optional: Round the corners */
            box-shadow: 0px 10px 20px rgba(0, 0, 0, 0.1);
            transform: translateY(-10px);
        }
        div[data-testid="stForm"]:hover {
            background-color: #f0f0f5;
            border: 2px solid black !important;  /* Change 'red' to any color */
            border-radius: 40px;  /* Optional: Round the corners */
            box-shadow: 50px 40px 60px rgba(225, 225, 225, 225);
            transform: translateY(-1px);
        }
    </style>
    """,unsafe_allow_html=True)

    

def footer():
    st.markdown("""
        <style>
        .footer-fixed {
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            height: 60px;
            background-color: #f0f0f5;
            color: #333;
            text-align: center;
            line-height: 60px;
            font-size: 14px;
            border-top: 1px solid #ccc;
            z-index: 100;
        }

        .footer-fixed a {
            color: #007B55;
            text-decoration: none;
            font-weight: 500;
            margin: 0 10px;
        }

        .footer-fixed a:hover {
            text-decoration: underline;
        }
        </style>

        <div class="footer-fixed">
            © 2025 CompanionMind | 
            <a href="mailto:support@companionmind.ai">Support</a> | 
            <a href="https://www.linkedin.com" target="_blank">LinkedIn</a> | 
            <a href="https://github.com/mpsslalitha/CompanionMind" target="_blank">GitHub</a>
        </div>
    """, unsafe_allow_html=True)


def app_pages():
    with st.container():
        col1,col2,col3,col4,col5,col6 = st.columns(6)
        with col6:
            col8,col9 = st.columns(2)
            with col9:
                # add_custom_css11()
                if st.button(" logout 🔚 "):
                    st.session_state.authenticated = False
                    st.session_state.username = None
                    st.session_state.page = "Landing"
                    st.rerun()
            
    # st.header(f"Welcome,{username}")
    tabs = st.tabs(["Home", "About", "Contact Us","blogs","chatBot"])
    
    with tabs[0]:
        decor()
        image = Image.open('home.jpg')
        re_image = image.resize((1600,550))
        st.image(re_image)
        with st.container():
            st.markdown(
                """
                <style>
                .card {
                    background-color: #f0f0f5;
                    padding: 20px;
                    margin: 10px;
                    border-radius: 10px;
                    box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
                    transition: box-shadow 0.3s ease, transform 0.3s ease;
                }

                .card:hover {
                    box-shadow: 0px 10px 20px rgba(0, 0, 0, 0.2);
                    transform: translateY(-5px);
                }
                .card-header {
                    color : green;
                    font-size : 32px;
                    font-weight: bold;
                    margin-bottom: 10px;
                }

                .card-body {
                    aligin-text : center;
                    font-size : 22px;
                    color: black;
                    padding : 20px;
                }

                </style>
                """, unsafe_allow_html=True
            )

            # Card 1
            with st.container():
                st.markdown(f"""<div class="card">
                            <div class="card-header">Our mission</div>
                            <div class="card-header" style="font-size: 26px; color: black; font-weight: bold; text-align: center;"> Cultivating Mental Well-being for a Balanced Life </div>  
                            <div class="card-body" ">our mission is to empower individuals—students, professionals, and everyday people—with the tools to achieve mental stability, emotional resilience, and inner peace.</div>
                            </div>""", unsafe_allow_html=True)
            
        with st.container():
            st.markdown("<div style='font-size:32px; padding : 50px; text-align: center; font-weight: bold;'>Activities</div>",unsafe_allow_html=True)
            
            col1,col2,col3 = st.columns(3)
            with col1:
                st_form()
                with st.form('meditate'):
                    st.markdown("<div style='font-size:24px; text-align: center; font-weight: bold;'>Meditation</div>",unsafe_allow_html=True)
                    image = Image.open('save.jpg')
                    re_image1 = image.resize((700,450))
                    st.image(re_image1)
                    st.markdown("<div style='font-size:18px; text-align: center; font-weight:bold; padding: 5px; '>In today’s fast-paced world, mental stability is often challenged by stress, anxiety, and constant distractions. Meditation is a simple yet powerful practice that helps bring clarity, emotional balance, and inner peace. </div>",unsafe_allow_html=True)
                    submit = st.form_submit_button('Meditation Exercise')

                if submit:
                    st_form()
                    st.session_state.page = "meditate"
                    st.rerun()

            with col2:
                st_form()
                with st.form('depress'):
                    st.markdown("<div style='font-size:24px; text-align: center; font-weight: bold;'>Depression</div>",unsafe_allow_html=True)
                    image = Image.open('depress.jpg')
                    re_image1 = image.resize((700,450))
                    st.image(re_image1)
                    st.markdown("<div style='font-size:18px; text-align: center;font-weight:bold;'>Depression can silently impact your thoughts, emotions, and daily life. But acknowledging it is the first step toward healing. A simple self-assessment can help you understand your mental health and take action toward a balanced, healthier mind.</div>",unsafe_allow_html=True)
                    submit = st.form_submit_button('Depression Test')

                if submit:
                    st.session_state.page = "depress"
                    st.rerun()

            with col3:
                st_form()
                with st.form('Breath'):
                    st.markdown("<div style='font-size:24px; text-align: center; font-weight: bold;'>Breathing</div>",unsafe_allow_html=True)
                    image = Image.open('478.jpg')
                    re_image1 = image.resize((700,450))
                    st.image(re_image1)
                    st.markdown("<div style='font-size:18px; text-align: center;font-weight:bold;'>In the fast-paced world we live in, stress, anxiety, and mental fatigue have become common challenges. Breathing exercises serve as a natural and powerful tool to regain mental stability, emotional balance, and overall well-being.</div>",unsafe_allow_html=True)
                    submit = st.form_submit_button('Breathing Exercise')

                if submit:
                    st.session_state.page = "breath"
                    st.rerun()
        footer()

    with tabs[1]:
        st.markdown("""
            <style>
            .about-title {
                font-size: 36px;
                text-align: center;
                font-weight: bold;
                color: #2E8B57;
                margin-bottom: 10px;
            }
            .about-text {
                font-size: 18px;
                color: black;
                text-align: center;
                padding: 0px 50px;
            }
            </style>
        """, unsafe_allow_html=True)

        st.markdown('<div class="about-title">About CompanionMind</div>', unsafe_allow_html=True)
        st.markdown("""
            <div class="about-text">
            CompanionMind is a compassionate AI-powered platform designed to support mental wellness.
            We aim to provide accessible tools for depression assessment, mindfulness, meditation, and emotional support.
            Our team believes in promoting emotional resilience and balance through technology.
            <br><br>
            This app was built with care to help students, professionals, and everyday individuals find peace,
            reflect on their emotions, and take proactive steps toward mental well-being.
            <br><br>
            <strong>Features include:</strong><br>
            - Depression Prediction Tests (Student, Working, Common)<br>
            - Meditation Timer with Soothing Audio<br>
            - Guided Breathing Exercises (4-7-8 method)<br>
            - Mental Health Chatbot for real-time conversation<br>
            </div>
        """, unsafe_allow_html=True)

        footer()


    with tabs[2]:
        st.markdown("""
            <style>
            .contact-title {
                font-size: 36px;
                text-align: center;
                font-weight: bold;
                color: #2E8B57;
                margin-bottom: 20px;
            }
            .contact-sub {
                text-align: center;
                font-size: 18px;
                margin-bottom: 20px;
            }
            .contact-box {
                background-color:  white;
                padding: 25px;
                border: 1px solid black
                border-radius: 12px;
                box-shadow: 0px 4px 10px rgba(0,0,0,0.1);
                width: 70%;
                margin: auto;
            }
            </style>
        """, unsafe_allow_html=True)

        st.markdown('<div class="contact-title">📞 Contact Us</div>', unsafe_allow_html=True)
        st.markdown('<div class="contact-sub">We’d love to hear from you! Feel free to reach out with any questions, feedback, or support inquiries.</div>', unsafe_allow_html=True)
        st.markdown("""
        <style>
            .stText_input {
                text-align: center;
                display: flex;
                border-color : black;
                background-color: white;
            }
        </style>
        """,unsafe_allow_html=True)
        with st.form("contact_form"):
            
            st.markdown('<div class="contact-box">Feedback Form</div>', unsafe_allow_html=True)
            name = st.text_input("Your Name")
            email = st.text_input("Your Email")
            message = st.text_area("Your Message")
            submitted = st.form_submit_button("Send")

            if submitted:
                if name and email and message:
                    st.success("✅ Thank you for contacting us! We'll get back to you soon.")
                else:
                    st.error("❗ Please fill out all fields.")
            st.markdown('</div>', unsafe_allow_html=True)

        footer()


    with tabs[3]:
        st.header("Blogs")
        photos = {
        "Tips for students to overcome stress": {
            "image_path": "images/student.png",  # Replace with actual image paths
            "info": """
            
            College life can be a whirlwind of assignments, exams, extracurricular activities, and social commitments. It’s no wonder that many students feel stressed and overwhelmed. The good news is, there are ways to manage stress effectively and maintain a healthy balance. Here are some practical tips that can help you cope with stress and improve your overall well-being.
            
            ---
            
            ### **1. Prioritize and Organize Your Tasks:**
            
            Stress often arises when students feel like they have too much to do and not enough time. One of the best ways to tackle this is through effective time management. Start by making a to-do list of all your tasks and then prioritize them. Break large projects into smaller, manageable steps and set deadlines for each. Using tools like planners, calendars, or apps can help keep you on track.
            
            ---
            
            ### **2. Learn to Say **No**:**
            
            While it’s important to get involved in campus life, saying **yes** to everything can lead to burnout. Know your limits and don’t hesitate to decline commitments that may add unnecessary stress to your plate. Focus on activities that align with your goals and passions.

            
            ---
            
            ### **3. Exercise Regularly:**
            
            Exercise is a natural stress reliever. Physical activity releases endorphins, which are the body’s natural mood boosters. Even a quick walk around campus or a short workout can improve your mood and help clear your mind. Aim for at least 30 minutes of moderate activity several times a week.
            
            ---
            
            ### **4. Practice Mindfulness and Relaxation Techniques:**
            
            Taking a few minutes each day to practice mindfulness or relaxation techniques can help you manage stress. Breathing exercises, meditation, and yoga can calm your nervous system and reduce anxiety. Apps like Headspace or Calm are great for beginners and offer guided sessions to get you started.
            
            ---
            
            ### **5. Get Enough Sleep:**
            
            Sleep is crucial for your physical and mental well-being, and it can be easy to sacrifice sleep in favor of studying. However, inadequate sleep can actually reduce your ability to focus and perform well academically. Aim for 7-9 hours of sleep each night to allow your body and brain to recharge.
            
            ---
            
            ### **6. Maintain a Healthy Diet:**
            
            What you eat plays a significant role in how you feel. A well-balanced diet rich in fruits, vegetables, whole grains, and lean proteins can boost your energy levels and improve your mood. Avoid excessive caffeine and sugary snacks, which can lead to energy crashes and irritability.
            
            ---
            
            ### **7. Seek Support:**
            
            It’s important to remember that you don’t have to go through stressful times alone. Reach out to friends, family, or a counselor for support. Talking about what’s on your mind can provide relief and new perspectives. Many schools also offer mental health resources, so take advantage of these services if you need help.
            
            ---
            
            ### **8. Take Breaks:**
            
            Don’t forget to give yourself permission to take breaks! Studying for long hours without a break can lead to burnout and decreased productivity. Try the Pomodoro Technique—work for 25 minutes, then take a 5-minute break. Use your break to stretch, take a walk, or do something you enjoy.
            
            ---
            
            ### **9. Stay Positive:**
            
            It’s easy to get caught up in negative thinking when things aren’t going well, but staying positive can help reduce stress. Instead of focusing on what you can’t control, focus on the things you can. Celebrate small wins, whether it's finishing an assignment or simply taking time to relax.
            
            ---
            
            ### **10. Be Kind to Yourself:**
            
            Lastly, don’t forget to practice self-compassion. It’s okay to make mistakes, have setbacks, or feel overwhelmed sometimes. You’re only human, and stress is a normal part of life. Treat yourself with kindness, and remember that it’s okay to ask for help when needed.
            
            ---
            
            Stress is a natural part of student life, but it doesn’t have to control you. By adopting healthy habits, managing your time effectively, and reaching out for support when needed, you can navigate stress with resilience. Remember, you’re not alone, and with the right tools, you can thrive both academically and personally.
            """,
            "btn_name": "Read Blog"
        },
        "Tips to Help Employees Overcome Stress at Work": {
            "image_path": "images/employee.png",  # Replace with actual image paths
            "info": """
        

            Stress at work is something nearly every employee faces at some point. Whether it's a looming deadline, an overwhelming workload, or challenging relationships with colleagues, workplace stress can take a toll on your health, productivity, and overall well-being. However, the good news is that there are many practical strategies employees can use to manage and reduce stress effectively. Here are eight tips to help you stay calm, focused, and more productive at work.

            ---

            ### 1. **Prioritize and Organize Your Tasks**

            One of the primary sources of stress is feeling overwhelmed by a long list of things to do. When you have too much on your plate, it’s easy to feel stuck or anxious about how to start. Prioritizing your tasks can help you tackle them one at a time, instead of feeling scattered.

            **Tip:** Use tools like to-do lists or task management apps (e.g., Todoist, Asana, or Trello) to break down your tasks. List them by urgency and importance, and don’t hesitate to ask for clarification if you're unsure about priorities. This can reduce the mental load and give you a clear direction.

            ---

            ### 2. **Take Regular Breaks**

            It’s tempting to push through long hours of work without taking a break, but working non-stop actually increases stress and reduces productivity. Breaks are essential for recharging and maintaining focus throughout the day.

            **Tip:** Follow the Pomodoro Technique: work for 25 minutes, then take a 5-minute break. After four cycles, take a longer break (15-30 minutes). It keeps your mind fresh and helps maintain energy levels.

            ---

            ### 3. **Practice Deep Breathing and Mindfulness**

            When stress strikes, your body responds with physical symptoms like rapid breathing and a racing heart. Learning to calm your body and mind with breathing techniques can help counter these effects and lower stress levels.

            **Tip:** Try deep breathing exercises like the 4-7-8 technique (inhale for 4 seconds, hold for 7 seconds, exhale for 8 seconds) or mindfulness meditation to center yourself. Even a few minutes of focused breathing can significantly reduce stress.

            ---

            ### 4. **Set Boundaries to Avoid Overwork**

            Many employees experience stress because they don’t set healthy boundaries between their work and personal lives. It's easy to slip into the habit of checking emails late into the night or working through lunch, but this can quickly lead to burnout.

            **Tip:** Define your working hours and communicate them with your team. Use “Do Not Disturb” modes on your devices outside work hours to disconnect. By setting these boundaries, you allow yourself time to recharge and focus on personal activities.

            ---

            ### 5. **Get Moving During the Day**

            Physical activity is one of the best ways to combat stress. Exercise releases endorphins, which help improve mood and relieve tension. You don’t need a full workout to experience the benefits—simple movements during the day can make a difference.

            **Tip:** Take short walks around the office or do stretching exercises at your desk. If possible, incorporate a lunchtime walk or a brief gym session into your day. These breaks will clear your mind and give you a fresh perspective when you return to work.

            ---

            ### 6. **Seek Support When Needed**

            Don’t feel like you need to handle stress alone. If you're feeling overwhelmed, talking to a colleague, mentor, or manager can help alleviate some of the pressure. A support system can offer a different perspective, advice, or even just a listening ear.

            **Tip:** Schedule regular check-ins with a trusted coworker or supervisor. Discuss your challenges and ask for feedback or help when necessary. Sometimes, just talking about what’s stressing you out can make a huge difference.

            ---

            ### 7. **Focus on What You Can Control**

            Stress often arises when we fixate on things outside of our control—like the actions of others, unforeseen circumstances, or things that didn’t go as planned. Shifting your focus to what you can control can significantly reduce stress.

            **Tip:** Identify areas in your workday where you have control (your reactions, your schedule, how you handle tasks) and focus on those. Let go of things beyond your influence. This mindset shift can help you feel more empowered and less stressed.

            ---

            ### 8. **Celebrate Small Wins**

            In a fast-paced work environment, it’s easy to focus solely on big goals or challenges. However, recognizing and celebrating small achievements can help build momentum and increase motivation.

            **Tip:** Take time to acknowledge the small wins—completing a project milestone, receiving positive feedback, or simply staying on top of your to-do list. Rewarding yourself with even small celebrations reinforces positive behaviors and helps maintain your energy.

            ---

            Stress is an inevitable part of any workplace, but how we manage it makes all the difference. By implementing some of these strategies, employees can create a healthier work environment, boost their resilience, and ultimately increase their job satisfaction. Remember, managing stress is a skill that takes practice. 
            By prioritizing self-care and finding what works best for you, you'll be better equipped to handle whatever challenges come your way. By incorporating these practices into your daily routine, you can start taking proactive steps toward a less stressful and more fulfilling work life.
            """,
            "btn_name": "Read Blog"
        },
        "Overcome Stress and Find Peace in Everyday Life": {
            "image_path": "images/people.png",  # Replace with actual image paths
            "info": """
            Stress is something we all face, whether it’s due to work, family responsibilities, financial worries, or even just the hustle and bustle of daily life. It’s easy to feel overwhelmed, but the good news is that you don’t have to let stress control your life. With a few simple changes, you can reclaim your peace and build healthier ways to cope. In this blog, we’ll explore seven practical tips that can help you reduce stress and improve your well-being.
            
            ---

            ### 1. **Breathe and Relax**
            When stress takes over, it often shows up in the body first—with shallow breathing, tension in the shoulders, or a racing heart. The simple act of deep breathing can help you reset your nervous system and bring calmness back into your day.

            **Tip:** Try the "4-7-8" breathing technique: inhale for 4 seconds, hold for 7 seconds, and exhale slowly for 8 seconds. Repeat for a few minutes. This slows your heart rate and helps you feel more relaxed in the moment.

            ---

            ### 2. **Simplify Your To-Do List**

            Many people stress out because they feel overwhelmed by a never-ending to-do list. You don’t have to get everything done at once! A little bit of structure can go a long way in helping you tackle your tasks without the anxiety of feeling like it’s all on your shoulders.

            **Tip:** Break your to-do list into smaller, more manageable chunks. Instead of focusing on everything at once, prioritize the most important or time-sensitive tasks. Once you’ve checked off a few smaller tasks, you’ll feel more in control and less stressed.

            ---

            ### 3. **Move Your Body**

            Exercise is a proven way to relieve stress, and it doesn’t have to be complicated. You don’t need a gym membership or an intense workout routine to feel the benefits of physical activity. Moving your body, even in small ways, can help release tension and improve your mood.

            **Tip:** Go for a walk around the block, do a few minutes of stretching, or even dance around your living room. These simple movements help release endorphins, the body’s natural stress relievers.

            ---

            ### 4. **Connect with Others**

            It’s easy to isolate yourself when you’re feeling stressed, but talking things out with a friend or family member can provide comfort and perspective. Social connections not only help alleviate stress but can also improve your overall well-being.

            **Tip:** Reach out to someone you trust, whether it’s for a chat over coffee or a casual text. Even a brief conversation can help shift your mindset and give you a sense of support.

            ---

            ### 5. **Practice Mindfulness or Meditation**

            Mindfulness is the practice of being present in the moment, without judgment. It’s a powerful tool for reducing stress, as it helps you let go of worries about the past or future. Meditation, even if only for a few minutes, can also be a great way to calm your mind.

            **Tip:** Try setting aside five to ten minutes a day for mindfulness or meditation. Find a quiet space, close your eyes, and focus on your breathing. If your mind starts to wander, gently bring it back to your breath. Apps like Headspace or Calm can help guide you through the process.

            ---

            ### 6. **Take Time for Yourself**

            It’s easy to forget about self-care when life gets busy. But taking time to recharge is essential to managing stress. Whether it’s enjoying a hobby, reading a book, or simply taking a nap, personal time helps you reset and gain perspective.

            **Tip:** Schedule "me-time" into your week, just like any other appointment. Even 15-30 minutes of doing something that brings you joy can make a world of difference in how you feel. It’s okay to put yourself first every now and then.

            ---

            ### 7. **Practice Gratitude**

            When you’re feeling stressed, it’s easy to focus on everything that’s going wrong. However, shifting your focus to what’s going right can help you feel more balanced and less overwhelmed.

            **Tip:** Keep a gratitude journal and write down three things you’re thankful for each day. They don’t have to be big things—sometimes, it's the little moments (a good cup of coffee, a kind word from a friend) that can make all the difference.

            ---

            Small Changes Can Make a Big Difference. Stress is an inevitable part of life, but it doesn’t have to consume you. By incorporating small changes into your daily routine, you can learn to manage stress more effectively and live a more balanced, peaceful life. The key is to recognize that you have the power to make positive shifts, no matter how hectic things may seem.


            Taking one step at a time, even in the midst of a busy life, can help you find moments of calm. And remember, it’s okay to ask for help or take a break when you need it. Your well-being matters!
            """,
            "btn_name": "Read Blog"
        },
        "Mastering Time Management": {
            "image_path": "images/time.png",  # Replace with actual image paths
            "info": """
            Time—it’s one of the most valuable resources we have, yet so many of us find ourselves running short of it. Whether it's work, personal commitments, or the constant buzz of life in general, managing time effectively can feel like an uphill battle. The good news is that with a few smart strategies, you can reclaim your time and use it more efficiently.

            In this blog, we’ll dive into simple, practical tips to help you manage your time better and reduce stress, so you can make room for what matters most.

            ---

            ### 1. **Set Clear Goals**

            One of the main reasons people struggle with time management is because they don’t have a clear sense of direction. Without defined goals, it’s easy to get lost in distractions or feel like you’re just "busy" without being productive.

            **Tip:** Start by setting SMART goals (Specific, Measurable, Achievable, Relevant, Time-bound). Whether it's for the day, week, or month, having clear goals gives you something concrete to focus on, helping you prioritize your tasks and stay on track.

            ---

            ### 2. **Break Tasks Into Smaller Chunks**

            Big tasks can be overwhelming, and when we face them head-on, they often feel like too much to handle. This feeling of being overwhelmed can cause procrastination, leaving us feeling like we don’t have enough time.

            **Tip:** Break down large tasks into smaller, more manageable steps. For example, if you need to write a report, break it into sections (research, outline, draft, edit). Completing smaller tasks gives you a sense of accomplishment and keeps you moving forward without feeling weighed down.

            ---

            ### 3. **Use the Pomodoro Technique**

            The Pomodoro Technique is a time management method that’s been proven to boost focus and productivity. It works by breaking your work into intervals, typically 25 minutes long, followed by a short break.

            **Tip:** Set a timer for 25 minutes and focus solely on the task at hand. Once the timer goes off, take a 5-minute break to recharge. After four "Pomodoros," take a longer break (15–30 minutes). This method helps you maintain focus, prevents burnout, and allows you to track your progress.

            ---

            ### 4. **Prioritize Your Tasks**

            Not all tasks are created equal, and not all of them need to be done immediately. Learning to prioritize your tasks based on urgency and importance is key to managing time effectively.

            **Tip:** Use the **Eisenhower Matrix** to categorize tasks into four quadrants:
            - **Urgent and Important:** Do these first (e.g., deadlines, emergencies).
            - **Important, Not Urgent:** Schedule these for later (e.g., long-term projects, personal growth).
            - **Urgent, Not Important:** Delegate these if possible (e.g., distractions, certain emails).
            - **Not Urgent, Not Important:** Eliminate or minimize these (e.g., social media scrolling, unnecessary meetings).

            By prioritizing tasks, you focus on what truly matters and avoid wasting time on less critical activities.

            ---

            ### 5. **Learn to Say No**

            In an age of constant requests and demands, it’s easy to spread yourself too thin. Saying yes to everything leads to overcommitment, burnout, and feeling like you’re constantly running out of time.

            **Tip:** Be mindful of what you agree to. It’s okay to say no when you’re already at capacity. Politely but firmly turn down requests or delegate tasks when appropriate. Remember, your time is precious, and it’s better to do a few things well than many things poorly.

            ---

            ### 6. **Eliminate Distractions**

            We live in a world full of distractions—social media, emails, phone notifications, and more. These constant interruptions can break your focus and eat away at your precious time.

            **Tip:** Turn off non-essential notifications on your phone and computer when you're working on important tasks. Use apps like **Forest** (to stay off your phone) or **Freedom** (to block distracting websites). If you're working in a noisy environment, consider using noise-canceling headphones or listening to instrumental music to maintain focus.

            ---

            ### 7. **Use a Calendar or Planner**

            A physical or digital calendar is a simple yet powerful tool for time management. It helps you keep track of deadlines, meetings, and personal commitments, ensuring that nothing falls through the cracks.

            **Tip:** Schedule your day in advance—either the night before or first thing in the morning. Block off time for specific tasks and use reminders to stay on top of your commitments. If you prefer a digital tool, apps like Google Calendar, Microsoft Outlook, or Trello can keep you organized and on track.

            ---

            ### 8. **Batch Similar Tasks**

            Task-switching—constantly jumping from one type of activity to another—can be mentally draining and inefficient. Instead, try batching similar tasks together to streamline your workflow.

            **Tip:** For example, group all of your emails into one block of time rather than checking them sporadically throughout the day. Similarly, schedule a specific time for phone calls, errands, or meetings. This minimizes distractions and helps you stay focused on a single type of work at a time.

            ---

            ### 9. **Take Care of Yourself**

            It may sound counterintuitive, but the more you take care of your physical and mental well-being, the more effective you'll be with your time. When you’re well-rested, hydrated, and mentally sharp, you can complete tasks more efficiently.

            **Tip:** Make time for exercise, eat nutritious foods, get adequate sleep, and practice mindfulness. A balanced body and mind are essential for productivity and will help you make the most of the time you have.

            ---

            ### 10. **Reflect and Adjust**

            Time management isn’t a one-size-fits-all solution. What works for one person may not work for another. The key is to reflect on your methods regularly and make adjustments as needed.

            **Tip:** At the end of each week, take a moment to reflect on what worked and what didn’t. Did you meet your goals? Did you waste time on distractions? Adjust your approach based on your reflections and keep tweaking your strategy for improvement.

            ---

            ### Time Is What You Make of It

            Mastering time management isn’t about getting more hours in the day—it’s about using the time you have more wisely. By setting goals, prioritizing tasks, eliminating distractions, and caring for yourself, you can reduce stress, boost productivity, and feel more in control of your day. Time will always be limited, but with the right strategies, you can make the most of it and live a more balanced life.
            
            With just a few adjustments, you can create habits that will set you up for success in managing your time—and ultimately, in living a more fulfilling life.
            """,
            "btn_name": "Read Blog"
        },
        "The Connection Between Physical and Mental Health": {
            "image_path": "images/student.png",  # Replace with actual image paths
            "info": """
            **The Connection Between Physical and Mental Health: How Taking Care of Your Body Can Positively Impact Mental Well-Being**

            ---

            When we think of health, it's easy to focus on the physical aspects: eating well, exercising, and getting enough sleep. But what if we told you that taking care of your body isn't just about looking good—it's also about feeling good mentally? The link between physical and mental health is stronger than most of us realize, and the state of one directly impacts the other.

            In this blog, we’ll explore how taking care of your body through regular exercise, a balanced diet, proper sleep, and stress management can significantly improve your mental well-being. It’s all about understanding how the body and mind work together in a symbiotic relationship.

            ---

            ### 1. **Exercise: The Natural Mood Booster**

            Physical activity is one of the most well-researched ways to boost mental health. When you exercise, your body releases endorphins—those feel-good chemicals that help reduce stress, anxiety, and even symptoms of depression. Regular exercise also improves sleep quality, boosts self-esteem, and sharpens your thinking.

            **The Science:**  
            Exercise increases the production of neurotransmitters like serotonin and dopamine. These chemicals play a crucial role in regulating mood, motivation, and happiness. It's not just about running a marathon; even a simple 30-minute walk can make a significant difference in how you feel.

            **Tip:** Aim for at least 150 minutes of moderate-intensity exercise each week (that’s just 30 minutes five days a week!). This can include walking, jogging, yoga, dancing, or even gardening. Anything that gets your body moving counts!

            <img src="images/jogging.png" width="800" style="display: block; margin: 20px auto;">
            
            ---

            ### 2. **Sleep: The Foundation of Mental Health**

            Good sleep is essential for both physical and mental health. When you’re sleep-deprived, it’s harder to concentrate, regulate emotions, or cope with stress. Chronic sleep problems are linked to a higher risk of developing mental health issues such as anxiety and depression.

            **The Science:**  
            During deep sleep, your brain clears out waste products that build up throughout the day. It’s also a time for emotional processing. Without enough sleep, your brain struggles to regulate emotional responses, making it more difficult to handle stress and emotions effectively.

            **Tip:** Aim for 7–9 hours of quality sleep each night. Establish a bedtime routine, reduce caffeine intake later in the day, and avoid looking at screens before bed to improve your sleep hygiene.

            **Image Suggestion:** A peaceful bedroom scene with soft lighting and a cozy bed, promoting relaxation.

            ---

            ### 3. **Healthy Eating: Fuel for Your Mind**

            The foods you eat have a direct impact on your brain health. A balanced diet rich in nutrients—especially omega-3 fatty acids, antioxidants, and vitamins—helps maintain cognitive function, regulate mood, and even combat inflammation that can lead to mental health issues.

            **The Science:**  
            Research shows that diets high in processed foods and sugar can lead to fluctuations in blood sugar levels, which in turn affects mood stability. On the other hand, eating nutrient-dense foods like leafy greens, fish, nuts, and whole grains supports brain health and can even reduce the risk of depression.

            **Tip:** Try to include more whole, nutrient-dense foods in your diet. Focus on eating a variety of fruits, vegetables, lean proteins, and healthy fats. Foods like salmon, avocados, and dark chocolate are known to have mental health benefits!

            **Image Suggestion:** A colorful salad bowl with mixed greens, salmon, avocado, and nuts, symbolizing a balanced, brain-boosting meal.

            ---

            ### 4. **Stress Management: Keep Your Mind in Balance**

            Chronic stress is a major contributor to both physical and mental health problems. When stress is prolonged, it can lead to anxiety, depression, and even physical conditions like heart disease. Managing stress is essential for maintaining both a healthy body and mind.

            **The Science:**  
            Stress triggers the release of cortisol, the "stress hormone." While small amounts of cortisol are natural and even necessary, chronic stress can lead to imbalances, affecting your immune system, digestion, and mental health. Regular physical activity, mindfulness, and relaxation techniques can help reduce cortisol levels and promote a sense of calm.

            **Tip:** Incorporate stress-reducing activities into your daily routine. Practices like yoga, meditation, deep breathing exercises, and even simple hobbies like knitting or painting can lower stress levels and improve mental clarity.

            **Image Suggestion:** A person practicing yoga or deep breathing in a peaceful, serene environment.

            ---

            ### 5. **The Power of Social Connections**

            Strong social relationships and support systems are crucial for mental well-being. The physical act of hugging, talking, or laughing with friends or family not only strengthens your emotional bonds but also has a positive effect on your physical health by reducing stress and boosting immune function.

            **The Science:**  
            When you spend time with loved ones or engage in social activities, your brain releases oxytocin—also known as the "love hormone." Oxytocin promotes feelings of happiness, reduces stress, and even helps lower blood pressure. Having a supportive social network also reduces feelings of isolation, which is a key risk factor for depression and anxiety.

            **Tip:** Make time for social connections, even if it’s just a quick call with a friend or a weekend hangout. Positive, nurturing relationships play a big role in your mental health.

            **Image Suggestion:** A group of friends laughing and enjoying each other’s company at a coffee shop, illustrating the positive impact of social interactions.

            ---

            ### 6. **Hydration: Don’t Forget the Basics**

            While hydration is often overlooked in conversations about health, it plays a critical role in your mental state. Dehydration can lead to fatigue, headaches, and poor concentration, all of which can negatively affect your mood and productivity.

            **The Science:**  
            Water is essential for transporting nutrients throughout your body, including to your brain. Even mild dehydration can impair cognitive function and lead to feelings of irritability, fatigue, and low mood.

            **Tip:** Make sure to drink plenty of water throughout the day. Aim for at least eight 8-ounce glasses, but more may be necessary depending on activity level and climate.

            **Image Suggestion:** A refreshing glass of water with a slice of lemon, symbolizing hydration and mental clarity.

            ---

            ###  Mind and Body Are Inseparable

            Taking care of your body isn’t just about physical appearance or performance—it’s about nurturing your mind and emotional well-being too. Regular exercise, balanced nutrition, quality sleep, stress management, and social connections all work together to support mental health. When you treat your body with care, you create the foundation for a healthier, happier, and more resilient mind.

            Remember, the mind and body are deeply interconnected. By making small changes to improve your physical health, you’re not just enhancing your body—you’re also supporting your mental well-being.


            Taking care of yourself is a holistic process—one that nourishes both the body and the mind for long-term health and happiness. 🌿✨
            """,
            "btn_name": "Read Blog"
            }
        }
        
        if "page1" not in st.session_state:
            st.session_state.page1 = "land"
        def main1():
            if st.session_state.page1 == "land":
                landing()
            elif st.session_state.page1 == "view":
                view()

        def landing():
            # # Session state to track which blog is currently active
            if "active_blog" not in st.session_state:
                st.session_state.active_blog = None

            # # Layout with two columns per row
            cols = st.columns(3)

            for idx, (photo_name, photo_data) in enumerate(photos.items()):
                col = cols[idx % 3]  # Arrange blogs in two columns
                with col:
                    image = Image.open(photo_data['image_path'])
                    re_image2 = image.resize((300,300))
                    st.image(re_image2)
                    st.write(photo_name)
                    
                    if st.button(f"{photo_data['btn_name']}", key=photo_name):
                        # Toggle active blog
                        if st.session_state.active_blog == photo_name:
                            st.session_state.active_blog = None  # Hide content when clicked again
                        else:
                            st.session_state.active_blog = photo_name  # Show the clicked blog

            st.markdown("------")
            # Display the selected blog content
            if st.session_state.active_blog:
                st.session_state.page1 = "view"
                st.rerun()

        def view():
            selected_blog = photos[st.session_state.active_blog]
            image = Image.open(selected_blog['image_path'])
            re_image2 = image.resize((850,500))
            col1,col2,col3 = st.columns(3)
            with col2:
                st.image(re_image2)
                # st.image(selected_blog["image_path"], use_column_width=True)
            st.write(f"### {st.session_state.active_blog}")
            st.markdown(selected_blog["info"], unsafe_allow_html=True)
            if st.button("Close"):
                st.session_state.page1 = "land"
                st.session_state.active_blog = None
                st.rerun()
        
        if __name__ == "__main__":
            main1()
        
        footer()

    with tabs[4]:

        # st.set_page_config(page_title="Mental Health Chatbot", layout="centered")

        if "conversation_history" not in st.session_state:
            st.session_state["conversation_history"] = []

        CRISIS_KEYWORDS = [
            "self-harm", "suicide", "kill myself", "want to die", "ending my life",
            "no reason to live", "can't go on", "hopeless", "hurting myself", "cutting",
            "taking my life", "life is meaningless", "death", "dying"
        ]
        MENTAL_HEALTH_TOPICS = [
            "stress", "anxiety", "depression", "mental health", "mindfulness",
            "meditation", "breathing", "therapy", "self-care", "well-being",
            "relaxation", "panic", "burnout", "insomnia", "loneliness", "overwhelmed",
            "fatigue", "trauma", "PTSD", "mental illness", "coping", "emotions",
            "grief", "sadness", "worry", "fear", "nervous", "bipolar", "OCD",
            "schizophrenia", "psychologist", "psychiatrist", "mood", "distress",
            "serotonin", "dopamine", "CBT", "DBT", "mindful", "healing", "hopeless",
            "negativity", "positivity", "calm", "soothing", "counseling",
        "therapy session",
            "emotional support", "mental wellness", "psychotherapy", "self-compassion",
            "meditative", "inner peace", "mental clarity", "unmotivated", "numb",
            "life purpose", "emotional health", "mental exhaustion", "wellness",
            "inner healing", "mental breakdown", "pressure", "guilt", "shame",
            "hypnotherapy", "affirmations", "deep breathing", "guided meditation"
        ]
        def is_mental_health_query(user_input):
            return any(topic in user_input.lower() for topic in MENTAL_HEALTH_TOPICS)
        def is_crisis_query(user_input):
            return any(keyword in user_input.lower() for keyword in CRISIS_KEYWORDS)
        def generate_response(user_input):
            st.session_state["conversation_history"].append({"role": "user",
        "content": user_input})
            if is_crisis_query(user_input):
                ai_response = ( "I'm really sorry you're feeling this way. You're notalone, and there are people who care about you. "
                    "Please reach out to a trusted friend, family member, or aprofessional for support. "
                    "If you're in crisis, I strongly encourage you to contacta mental health helpline in your country."
                    "\n\n**Resources:**\n"
                    "- 📞 [Find a Helpline](https://findahelpline.com/)"
                    "(International Suicide Prevention Hotlines)\n"
                    "- 📞 U.S.: Call or text **988** for the Suicide & Crisisn Lifeline\n"
                    "- 📞 UK: Call **Samaritans at 116 123**\n"
                    "- 📞 India: Call **Vandrevala Foundation Helpline at 1860266 2345**"
                )
            elif is_mental_health_query(user_input):
                response = ollama.chat(model="llama3.2",
        messages=st.session_state["conversation_history"])
                ai_response = response["message"]["content"]
            else:
                ai_response = "I'm here to discuss mental health topics. Let's talk about stress management, self-care, mindfulness, or anything related to well-being. 😊"
            st.session_state["conversation_history"].append({"role": "assistant", "content": ai_response})
            st.rerun()
        def generate_affirmation():
            generate_response("Provide a positive affirmation to encourage someone who is feeling stressed or overwhelmed.")

        def generate_meditation():
            generate_response("Provide a 3-minute guided meditation script to relax and reduce stress.")

        st.markdown(
            """
            <h1 style='text-align: center;'> 🤖 CompanionMind – Your AI Chat
        Companion 🤖</h1>
            <h3 style='text-align: center; color: gray;'>Your personal
        AI-powered support for mindfulness, well-being, and guidance.</h3>
            """,
            unsafe_allow_html=True
        )

        for message in st.session_state["conversation_history"]:
            with st.chat_message("user" if message["role"] == "user" else "assistant"):
                st.markdown(message["content"])

        user_message = st.chat_input("How can I help you today?")

        if user_message:
            generate_response(user_message)

        col1, col2 = st.columns(2)

        with col1:
            if st.button("🌟 Positive Affirmation"):
                generate_affirmation()

        with col2:
            if st.button("🧘 Guided Meditation"):
                generate_meditation()

        st.markdown(
            "<div style='text-align: center; margin-top: 50px; font-size:14px; color: gray;'>"
            "This is an AI chatbot focused on mental health. It may make mistakes."
            "</div>",
            unsafe_allow_html=True
        )

        footer()
    
    


def std_depression_test():
    st.title("Student Depression Test")

    name = st.text_input('Patient Name')
    mobile_no = st.number_input('Mobile Number',min_value = 100000000, max_value = 9999999999,step = 1)
    landline_no = st.text_input('Landline Number')
    self_employed = st.selectbox('self_employed', model2['self_employed'].unique())
    Country = st.selectbox('select your country', model2['Country'].unique())
    Ocupation = st.text_input('Occupation')
    age = st.number_input("Student Age",min_value=0, max_value=25, step=1)
    academic_pressure = st.selectbox('Academic Pressure',  [1,2,3,4,5])
    cgpa = st.number_input('CGPA', min_value=0.0, max_value=10.0, step=1.0)
    study_satisfaction = st.number_input('Study Satisfaction (0 - Very Dissatisfied, 10 - Very Satisfied)', min_value=0, max_value=10, step=1)
    study_hours = st.number_input('Studying in hours per day', min_value=0, max_value=24, step=1)
    gender = st.selectbox('Gender', ['Male', 'Female'])
    sleep_duration = st.selectbox('Sleep Duration per day', model.get('sleep Duration', [4, 5, 6, 7, 8, 9]))
    dietary = st.selectbox('Dietary Habits', ['Healthy', 'Moderate', 'Unhealthy', 'Others'])
    suicide = st.selectbox('Have you ever had suicidal thoughts? (Yes/No)', ['Yes', 'No'])
    family_illness = st.selectbox('Family History of Mental Illness? (Yes/No)', ['Yes', 'No'])

    if st.button("Predict"):
        try:
            input_data = preprocess_input(name, Country, mobile_no,landline_no,Ocupation,self_employed, age, academic_pressure, cgpa, study_satisfaction, study_hours, gender, sleep_duration, dietary, suicide, family_illness)
            prediction = svc.predict(input_data)

            if prediction[0] == 1:
                ans = "The model predicts a high likelihood of depression. Please seek professional help."
                st.error("The model predicts a high likelihood of depression. Please seek professional help.")
            else:
                ans = "The model predicts no significant signs of depression."
                st.success("The model predicts no significant signs of depression.")
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

        data_to_store = {
            "name": name,
            "mobile_no": mobile_no,
            "landline_no": landline_no,
            "Country": Country,
            "Ocupation": Ocupation,
            "self_employed": self_employed,
            "Age": age,
            "academic_pressure": academic_pressure,
            "cgpa": cgpa,
            "study_satisfaction": study_satisfaction,
            "study_hours": study_hours,
            "gender": gender,
            "sleep_duration": sleep_duration,
            "dietary": dietary,
            "suicide": suicide,
            "family_illness": family_illness,
            "ans": ans
        }
        with open("Report1.pkl", "wb") as f:
            pickle.dump(data_to_store,f)

    submit = st.button('submit')
    if submit:
        st.session_state.page = "Report"
        st.rerun()

def Work_depression_test():
    st.title("Working/Business Depression Test")

    # Input features
    name = st.text_input("Name")
    mobile_no = st.text_input("Mobile Number")
    landline_no = st.text_input("Landline Number")
    Country = st.selectbox('select your country', model2['Country'].unique())
    Ocupation = st.selectbox('Occupation',['Employee/Business'])
    self_employed = st.selectbox('self_employed', model2['self_employed'].unique())
    Age = st.number_input('Employee Age', min_value=0, max_value=80, step=1)
    Work_Pressure = st.selectbox('Work Pressure', model1['Work Pressure'].unique())
    Job_satisfaction = st.number_input('Job Satisfaction (0 - Very Dissatisfied, 5 - Very Satisfied)', min_value=0, max_value=10, step=1)
    work_hours = st.number_input('Working in hours per day',min_value=0, max_value=24, step=1)
    Financial_stress = st.number_input('Financial Stress', min_value=0, max_value=10, step=1)
    gender = st.selectbox('Employee Gender', ['Male', 'Female'])
    sleep_duration = st.selectbox('Sleep Duration per day of Employee', [4, 6, 8, 9])
    dietary = st.selectbox('Dietary Habits of Employee', ['Healthy', 'Moderate', 'Unhealthy'])
    suicide = st.selectbox('Have you ever had suicidal thoughts or not? (Yes/No)', ['Yes', 'No'])
    family_illness = st.selectbox('Family History of Mental Illness? [Yes/No]', ['Yes', 'No'])

    if st.button("Predicts"):
        try:
            input_data = preprocess_input(name, Country, mobile_no,landline_no,Ocupation,self_employed,Age, Work_Pressure, Job_satisfaction, work_hours, Financial_stress, gender, sleep_duration, dietary, suicide, family_illness)
            prediction = svc1.predict(input_data)

            if prediction[0] == 1:
                ans = "The model predicts a high likelihood of depression. Please seek professional help."
                st.error("The model predicts a high likelihood of depression. Please seek professional help.")
            else:
                ans = "The model predicts no significant signs of depression."
                st.success("The model predicts no significant signs of depression.")
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
        
        data_to_store = {
            "name": name,
            "mobile_no": mobile_no,
            "landline_no": landline_no,
            "Country": Country,
            "Ocupation": Ocupation,
            "self_employed": self_employed,
            "Age": Age,
            "Work_Pressure": Work_Pressure,
            "Job_satisfaction": Job_satisfaction,
            "work_hours": work_hours,
            "Financial_stress": Financial_stress,
            "gender": gender,
            "sleep_duration": sleep_duration,
            "dietary": dietary,
            "suicide": suicide,
            "family_illness": family_illness,
            "ans": ans
        }
        with open("Report2.pkl", "wb") as f:
            pickle.dump(data_to_store,f)

    submit1 = st.button('submit')
    if submit1:
        st.session_state.page = "Report1"
        st.rerun()


def Common_for_all_depression_test():
    st.title("Common For All Depression Test")

    name = st.text_input("Name")
    age = st.number_input("Age",min_value = 1,max_value = 150,step=1)
    mobile_no = st.number_input("Mobile Number",min_value=1000000000, max_value=9999999999,step=1)
    landline_no = st.text_input("Landline Number")
    Gender = st.selectbox('Gender',model2['Gender'].unique())
    Country = st.selectbox('select your country', model2['Country'].unique())
    Ocupation = st.selectbox('Occupation', model2['Occupation'].unique())
    self_employed = st.selectbox('self_employed', model2['self_employed'].unique())
    family_history = st.selectbox('family_history',model2['family_history'].unique())
    Days_Indoors = st.selectbox('Days_Indoors', model2['Days_Indoors'].unique())
    Growing_Stress = st.selectbox('Growing_Stress', model2['Growing_Stress'].unique())
    Changes_Habits = st.selectbox('Changes_Habits', model2['Changes_Habits'].unique())
    Mental_Health_History = st.selectbox('Mental_Health_History', model2['Mental_Health_History'].unique())
    Mood_Swings = st.selectbox('Mood_Swings', model2['Mood_Swings'].unique())
    Coping_Struggles = st.selectbox('Coping_Struggles', model2['Coping_Struggles'].unique())
    Work_Interest = st.selectbox('Work_Interest', model2['Work_Interest'].unique())
    Social_Weakness = st.selectbox('Social_Weakness', model2['Social_Weakness'].unique())
    mental_health_interview = st.selectbox('mental_health_interview', model2['mental_health_interview'].unique())
    care_options = st.selectbox('care_options', model2['care_options'].unique())

    if st.button("Predicts"):
        try:
            input_data = preprocess_input1(name,age,mobile_no,landline_no, Gender,Country,Ocupation,self_employed,family_history,Days_Indoors,Growing_Stress,Changes_Habits,Mental_Health_History,Mood_Swings,Coping_Struggles,Work_Interest,Social_Weakness,mental_health_interview,care_options)
            prediction = GB.predict(input_data)

            if prediction[0] == 0:
                ans = 'The model predicts a high likelihood of depression. Please seek professional help.'
                st.error("The model predicts a high likelihood of depression. Please seek professional help.")
            else:
                ans = 'The model predicts no significant signs of depression.'
                st.success("The model predicts no significant signs of depression.")
                
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

        data_to_store = {
            "name": name,
            "age" : age,
            "mobile_no" : mobile_no,
            "landline_no" : landline_no,
            "Gender" : Gender,
            "Country" : Country,
            "Ocupation" : Ocupation,
            "self_employed" : self_employed,
            "family_history" : family_history,
            "Days_Indoors" : Days_Indoors,
            "Growing_Stress" : Growing_Stress,
            "Changes_Habits" : Changes_Habits,
            "Mental_Health_History" : Mental_Health_History,
            "Mood_Swings" : Mood_Swings,
            "Coping_Struggles" : Coping_Struggles,
            "Work_Interest" : Work_Interest,
            "Social_Weakness" : Social_Weakness,
            "mental_health_interview" : mental_health_interview,
            "care_options" : care_options,
            "ans" : ans
        }
        with open("Report3.pkl", "wb") as f:
            pickle.dump(data_to_store,f)

                
    submit2 = st.button('submit2')
    if submit2:
        st.session_state.page = "Report2"
        st.rerun()
        
def Report():
    with st.container():
        col1,col2,col3,col4,col5,col6 = st.columns(6)
        with col6:
            col7,col8,col9 = st.columns(3)
            with col9:
                if st.button("🔙"):
                    st.session_state.page = "depress"
                    st.rerun()
    class PDF(FPDF):
        def header(self):
            """Draws a border on every page"""
            self.rect(5, 5, 200, 287)  # Rectangle: (x, y, width, height)

        def footer(self):
            """Adds a page number at the bottom"""
            self.set_y(-15)
            self.set_font("Arial", size=10)
            self.cell(0, 10, f"Page {self.page_no()}", align="C")

    tabs = st.tabs(['Download Report','Back to home'])            
    with tabs[0]:
        def Generate_Report(name, mobile_no, landline_no, Country, Ocupation, self_employed, Age,
                            academic_pressure, cgpa, study_satisfaction, study_hours, gender,
                            sleep_duration, dietary, suicide, family_illness,ans):
            pdf = FPDF()
            pdf.set_auto_page_break(auto=True, margin=15)
            pdf.add_page()
            pdf.set_font("Arial", size=18)
            pdf.cell(200, 10, txt="CompanionMind", ln=True, align='C')
            pdf.set_font_size(10)
            pdf.cell(200, 4, txt="mental health report", ln=True, align='C')
            pdf.set_font_size(12)
            pdf.cell(200, 10, txt="Patient Information:", ln=True, align='C')
            pdf.rect(10, 25, 190, 50)
            pdf.set_font_size(10)
            pdf.set_text_color(50, 50, 50)
            
            pdf.cell(190, 10, txt=f"Name: {name}       Age: {Age}       Gender: {gender}       Country: {Country}", ln=True, align='L')
            pdf.cell(200, 10, txt=f"Mobile Number: {mobile_no}   Landline Number: {landline_no}", ln=True, align='L')
            pdf.cell(200, 10, txt=f"Occupation: {Ocupation}  Self-Employed: {self_employed}", ln=True, align='L')
            pdf.cell(200, 10, txt="", ln=True, align='L')
            
            pdf.set_text_color(0, 0, 0)
            pdf.set_font_size(12)
            pdf.cell(200, 10, txt="Mental Health and Lifestyle Information", ln=True, align='C')
            pdf.set_font_size(10)
            
            questions = [
                ("Academmic Pressure Level:", academic_pressure),
                ("cgpa):", cgpa),
                ("Studying Hours per Day:", study_hours),
                ("Study Satisfaction Level:", study_satisfaction),
                ("Sleep Duration per Day:", sleep_duration),
                ("Dietary Habits:", dietary),
                ("History of Suicidal Thoughts:", suicide),
                ("Family History of Mental Illness:", family_illness)
            ]
            
            for question, answer in questions:
                pdf.set_text_color(0, 0, 0)
                pdf.cell(200, 10, txt=question, ln=True, align='L')
                pdf.set_text_color(50, 50, 50)
                pdf.cell(200, 10, txt=f"{answer}", ln=True, align='L')
            
            pdf.cell(200, 10, txt="   ", ln=True, align='L')
            pdf.set_font_size(12)
            pdf.cell(200, 10, txt="Summary: ", ln=True, align='L')
            pdf.set_font_size(10)
            pdf.set_text_color(50, 50, 50)
            
            if ans == "The model predicts no significant signs of depression.":
                summary = f"{name} appears to have stable mental health. There are no major concerns detected based on the provided data."
                recommendations = [
                    "1. Maintain a balanced work-life routine.",
                    "2. Engage in social and physical activities to support well-being.",
                    "3. Continue monitoring stress levels and seek help if necessary."
                ]
            else:
                summary = f"{name} may be experiencing heightened stress or other mental health concerns. Further evaluation is recommended."
                recommendations = [
                    "1. Consider professional counseling for stress management.",
                    "2. Engage in mindfulness and relaxation activities.",
                    "3. Build a strong support network and seek help if needed."
                ]
            
            pdf.write(5, summary)
            pdf.cell(200, 10, txt=" ", ln=True, align='L')
            pdf.set_font_size(12)
            pdf.set_text_color(0, 0, 0)
            pdf.cell(200, 10, txt="Recommendations: ", ln=True, align='L')
            pdf.set_font_size(10)
            pdf.set_text_color(50, 50, 50)
            
            for rec in recommendations:
                pdf.cell(200, 10, txt=rec, ln=True, align='L')
            
            pdf.cell(200, 10, txt=" ", ln=True, align='L')
            pdf.output("std_mental_health_report.pdf")
            st.subheader("Report Generated and Downloaded Successfully.")
        
        with open('Report1.pkl', 'rb') as f:
            loaded_data = pickle.load(f)

        # Streamlit UI
        name = loaded_data.get("name")
        mobile_no = loaded_data.get("mobile_no")
        landline_no = loaded_data.get("landline_no")
        Country = loaded_data.get("Country")
        Ocupation = loaded_data.get("Ocupation")
        self_employed = loaded_data.get("self_employed")
        Age = loaded_data.get("Age")
        academic_pressure = loaded_data.get("academic_pressure")
        cgpa = loaded_data.get("cgpa")
        study_satisfaction = loaded_data.get("study_satisfaction")
        study_hours = loaded_data.get("study_hours")
        gender = loaded_data.get("gender")
        sleep_duration = loaded_data.get("sleep_duration")
        dietary = loaded_data.get("dietary")
        suicide = loaded_data.get("suicide")
        family_illness = loaded_data.get("family_illness")
        ans = loaded_data.get("ans")

        download = st.button("Download Report")
        st.write('Student')
        if download:
            Generate_Report(name, mobile_no, landline_no, Country, Ocupation, self_employed, Age,
                            academic_pressure, cgpa, study_satisfaction, study_hours, gender,
                            sleep_duration, dietary, suicide, family_illness,ans)

    with tabs[1]:
        back = st.button("Back to Home")
        if back:
            st.session_state.page = "Home"
            st.rerun()

def Report1():
    with st.container():
        col1,col2,col3,col4,col5,col6 = st.columns(6)
        with col6:
            col7,col8,col9 = st.columns(3)
            with col9:
                if st.button("🔙"):
                    st.session_state.page = "depress"
                    st.rerun()
    class PDF(FPDF):
        def header(self):
            """Draws a border on every page"""
            self.rect(5, 5, 200, 287)  # Rectangle: (x, y, width, height)

        def footer(self):
            """Adds a page number at the bottom"""
            self.set_y(-15)
            self.set_font("Arial", size=10)
            self.cell(0, 10, f"Page {self.page_no()}", align="C")

    tabs = st.tabs(['Download Report','Back to home'])            
    with tabs[0]:
        def Generate_Report(name, mobile_no, landline_no, Country, Ocupation, self_employed, Age,
                            Work_Pressure, Job_satisfaction, work_hours, Financial_stress, gender,
                            sleep_duration, dietary, suicide, family_illness,ans):
            pdf = FPDF()
            pdf.set_auto_page_break(auto=True, margin=15)
            pdf.add_page()
            pdf.set_font("Arial", size=18)
            pdf.cell(200, 10, txt="CompanionMind", ln=True, align='C')
            pdf.set_font_size(10)
            pdf.cell(200, 4, txt="mental health report", ln=True, align='C')
            pdf.set_font_size(12)
            pdf.cell(200, 10, txt="Patient Information:", ln=True, align='C')
            pdf.rect(10, 25, 190, 50)
            pdf.set_font_size(10)
            pdf.set_text_color(50, 50, 50)
            
            pdf.cell(190, 10, txt=f"Name: {name}       Age: {Age}       Gender: {gender}       Country: {Country}", ln=True, align='L')
            pdf.cell(200, 10, txt=f"Mobile Number: {mobile_no}   Landline Number: {landline_no}", ln=True, align='L')
            pdf.cell(200, 10, txt=f"Occupation: {Ocupation}  Self-Employed: {self_employed}", ln=True, align='L')
            pdf.cell(200, 10, txt="", ln=True, align='L')
            
            pdf.set_text_color(0, 0, 0)
            pdf.set_font_size(12)
            pdf.cell(200, 10, txt="Mental Health and Lifestyle Information", ln=True, align='C')
            pdf.set_font_size(10)
            
            questions = [
                ("Work Pressure Level:", Work_Pressure),
                ("Job Satisfaction Level (0-10):", Job_satisfaction),
                ("Working Hours per Day:", work_hours),
                ("Financial Stress Level:", Financial_stress),
                ("Sleep Duration per Day:", sleep_duration),
                ("Dietary Habits:", dietary),
                ("History of Suicidal Thoughts:", suicide),
                ("Family History of Mental Illness:", family_illness)
            ]
            
            for question, answer in questions:
                pdf.set_text_color(0, 0, 0)
                pdf.cell(200, 10, txt=question, ln=True, align='L')
                pdf.set_text_color(50, 50, 50)
                pdf.cell(200, 10, txt=f"{answer}", ln=True, align='L')
            
            pdf.cell(200, 10, txt="   ", ln=True, align='L')
            pdf.set_font_size(12)
            pdf.cell(200, 10, txt="Summary: ", ln=True, align='L')
            pdf.set_font_size(10)
            pdf.set_text_color(50, 50, 50)
            
            if ans == "The model predicts no significant signs of depression.":
                summary = f"{name} appears to have stable mental health. There are no major concerns detected based on the provided data."
                recommendations = [
                    "1. Maintain a balanced work-life routine.",
                    "2. Engage in social and physical activities to support well-being.",
                    "3. Continue monitoring stress levels and seek help if necessary."
                ]
            else:
                summary = f"{name} may be experiencing heightened stress or other mental health concerns. Further evaluation is recommended."
                recommendations = [
                    "1. Consider professional counseling for stress management.",
                    "2. Engage in mindfulness and relaxation activities.",
                    "3. Build a strong support network and seek help if needed."
                ]
            
            pdf.write(5, summary)
            pdf.cell(200, 10, txt=" ", ln=True, align='L')
            pdf.set_font_size(12)
            pdf.set_text_color(0, 0, 0)
            pdf.cell(200, 10, txt="Recommendations: ", ln=True, align='L')
            pdf.set_font_size(10)
            pdf.set_text_color(50, 50, 50)
            
            for rec in recommendations:
                pdf.cell(200, 10, txt=rec, ln=True, align='L')
            
            pdf.cell(200, 10, txt=" ", ln=True, align='L')
            pdf.output("work_mental_health_report.pdf")
            st.subheader("Report Generated and Downloaded Successfully.")
        
        with open('Report2.pkl', 'rb') as f:
            loaded_data = pickle.load(f)

        # Streamlit UI
        name = loaded_data.get("name")
        mobile_no = loaded_data.get("mobile_no")
        landline_no = loaded_data.get("landline_no")
        Country = loaded_data.get("Country")
        Ocupation = loaded_data.get("Ocupation")
        self_employed = loaded_data.get("self_employed")
        Age = loaded_data.get("Age")
        Work_Pressure = loaded_data.get("Work_Pressure")
        Job_satisfaction = loaded_data.get("Job_satisfaction")
        work_hours = loaded_data.get("work_hours")
        Financial_stress = loaded_data.get("Financial_stress")
        gender = loaded_data.get("gender")
        sleep_duration = loaded_data.get("sleep_duration")
        dietary = loaded_data.get("dietary")
        suicide = loaded_data.get("suicide")
        family_illness = loaded_data.get("family_illness")
        ans = loaded_data.get("ans")

        download = st.button("Download Report")
        st.write('Work')
        if download:
            Generate_Report(name, mobile_no, landline_no, Country, Ocupation, self_employed, Age,
                            Work_Pressure, Job_satisfaction, work_hours, Financial_stress, gender,
                            sleep_duration, dietary, suicide, family_illness,ans)

    with tabs[1]:
        back = st.button("Back to Home")
        if back:
            st.session_state.page = "Home"
            st.rerun()
            
def Report2():
    with st.container():
        col1,col2,col3,col4,col5,col6 = st.columns(6)
        with col6:
            col7,col8,col9 = st.columns(3)
            with col9:
                if st.button("🔙"):
                    st.session_state.page = "depress"
                    st.rerun()
    class PDF(FPDF):
        def header(self):
            """Draws a border on every page"""
            self.rect(5, 5, 200, 287)  # Rectangle: (x, y, width, height)

        def footer(self):
            """Adds a page number at the bottom"""
            self.set_y(-15)
            self.set_font("Arial", size=10)
            self.cell(0, 10, f"Page {self.page_no()}", align="C")

    tabs = st.tabs(['Report1','Back to home'])
    with tabs[0]:
        # Define Generate_Report before its usage
        def Generate_Report1(name, age, mobile_no, landline_no, Gender, Country, Ocupation, self_employed,
                      family_history, Days_Indoors, Growing_Stress, Changes_Habits, Mental_Health_History,
                      Mood_Swings, Coping_Struggles, Work_Interest, Social_Weakness, mental_health_interview,
                      care_options, ans):
            print(name)
            print(landline_no)
            pdf = PDF()
            pdf.set_auto_page_break(auto=True, margin=15)  # Set auto page break
            pdf.add_page()
            pdf.set_font("Arial", size=18)  # Set font size to 18
            pdf.cell(200, 10, txt="CompanionMind", ln=True, align='C')
            pdf.set_font_size(10)
            pdf.cell(200, 4, txt="mental health report", ln=True, align='C')
            pdf.set_font_size(12)
            pdf.cell(200, 10, txt=" Patient Information:", ln=True, align='C')
            pdf.rect(10, 25, 190, 50)
            pdf.set_font_size(10)
            pdf.set_text_color(50, 50, 50)
            pdf.set_auto_page_break(auto=True, margin=15)

            pdf.cell(190, 10, txt=f"Name: {name}       Age: {age}       Sex: {Gender}       Country: {Country}", ln=True, align='L')
            pdf.cell(200, 10, txt=f"Mobile Number: {mobile_no}   Landline Number: {landline_no}", ln=True, align='L')
            pdf.cell(200, 10, txt=f"Occupation: {Ocupation}  Self-Employed: {self_employed}", ln=True, align='L')
            pdf.cell(200, 10, txt="", ln=True, align='L')
            
            pdf.set_text_color(0, 0, 0)
            pdf.set_font_size(12)
            pdf.cell(200, 10, txt="Medical And LifeStyle Information", ln=True, align='C')
            pdf.set_font_size(10)
            
            # Medical History and Lifestyle Information
            questions = [
                ("What is your Family History of Mental Illness Status:", family_history),
                ("How many days do you spend indoors per week:", Days_Indoors),
                ("How are your Growing Stress Levels:", Growing_Stress),
                ("What are the Changes in your Habits:", Changes_Habits),
                ("What is your History of Mental Health Issues:", Mental_Health_History),
                ("How are your Mood Swings:", Mood_Swings),
                ("Are you facing Coping Struggles:", Coping_Struggles),
                ("What is your Interest in Work:", Work_Interest),
                ("Are you facing Social Weakness:", Social_Weakness),
                ("Have you conducted Mental Health Interview:", mental_health_interview),
                ("Are you aware of Care Options:", care_options),
                ("The model predicts and gives:", ans),
            ]

            for question, answer in questions:
                pdf.set_text_color(0, 0, 0)
                pdf.cell(200, 10, txt=question, ln=True, align='L')
                pdf.set_text_color(50, 50, 50)
                pdf.cell(200, 10, txt=f"{answer}", ln=True, align='L')
            
            pdf.cell(200, 10, txt="   ", ln=True, align='L')
            pdf.set_font_size(12)
            pdf.cell(200, 10, txt="Summary: ", ln=True, align='L')
            pdf.set_font_size(10)
            pdf.set_text_color(50, 50, 50)

            if ans == "The model predicts no significant signs of depression.":
                summary = f"{name} does not exhibit signs of depression. Their mental health assessment suggests a stable condition with normal stress management and coping mechanisms. There are no significant mental health concerns at this time."
                recommendations = [
                    "1. Maintain a healthy work-life balance to prevent future stress.",
                    "2. Engage in regular physical activities and social interactions.",
                    "3. Continue monitoring mental health and seek help if any concerns arise.",
                    "4. Stay informed about mental health and self-care strategies.",
                ]
            else:
                summary = f"{name} exhibits signs of heightened stress, mood swings, and struggles with coping mechanisms. Their mental health history suggests past experiences with {Mental_Health_History}, which may be exacerbated by occupational stress and limited social interaction. A mental health interview was conducted, revealing a need for structured intervention and support."
                recommendations = [
                    "1. Professional Counseling: Seeking regular therapy sessions to address stress management and coping strategies.",
                    "2. Lifestyle Changes: Encouragement to engage in physical activities and hobbies to reduce stress.",
                    "3. Support Network: Encouraging social interactions and community engagement.",
                    "4. Mental Health Resources: Providing information on available care options and support groups.",
                ]
            
            pdf.write(5, summary)
            pdf.cell(200, 10, txt=" ", ln=True, align='L')
            pdf.set_font_size(12)
            pdf.set_text_color(0, 0, 0)
            pdf.cell(200, 10, txt="Recommendations: ", ln=True, align='L')
            pdf.set_font_size(10)
            pdf.set_text_color(50, 50, 50)
            
            for rec in recommendations:
                pdf.cell(200, 10, txt=rec, ln=True, align='L')
            
            pdf.cell(200, 10, txt=" ", ln=True, align='L')
            pdf.set_font_size(12)
            pdf.set_text_color(0, 0, 0)
            pdf.cell(200, 10, txt="Next Steps: ", ln=True, align='L')
            pdf.set_font_size(10)
            pdf.set_text_color(50, 50, 50)
            pdf.write(5, "A follow-up session is recommended in four weeks to monitor progress and adjust the treatment plan if necessary.")
            
            pdf.cell(200, 10, txt="", ln=True, align='L')
            pdf.write(5, "This report is confidential and intended for the patient's personal and medical reference.")
            
            pdf.output("common_mental_health_report.pdf")
            st.subheader("Report Generated and Downloaded Successfully.")

        # Load the pickle file data
        with open('Report3.pkl', 'rb') as f:
            loaded_data = pickle.load(f)

        name = loaded_data.get("name")
        age = loaded_data.get("age")
        mobile_no = loaded_data.get("mobile_no")
        landline_no = loaded_data.get("landline_no")
        Gender = loaded_data.get("Gender")
        Country = loaded_data.get("Country")
        Ocupation = loaded_data.get("Ocupation")
        self_employed = loaded_data.get("self_employed")
        family_history = loaded_data.get("family_history")
        Days_Indoors = loaded_data.get("Days_Indoors")
        Growing_Stress = loaded_data.get("Growing_Stress")
        Changes_Habits = loaded_data.get("Changes_Habits")
        Mental_Health_History = loaded_data.get("Mental_Health_History")
        Mood_Swings = loaded_data.get("Mood_Swings")
        Coping_Struggles = loaded_data.get("Coping_Struggles")
        Work_Interest = loaded_data.get("Work_Interest")
        Social_Weakness = loaded_data.get("Social_Weakness")
        mental_health_interview = loaded_data.get("mental_health_interview")
        care_options = loaded_data.get("care_options")
        ans = loaded_data.get("ans")

        st.write(name)
        download = st.button("Download")
        st.write('Common')
        if download:
            Generate_Report1(name, age, mobile_no, landline_no, Gender, Country, Ocupation, self_employed,
                            family_history, Days_Indoors, Growing_Stress, Changes_Habits, Mental_Health_History,
                            Mood_Swings, Coping_Struggles, Work_Interest, Social_Weakness, mental_health_interview,
                            care_options, ans)
    
    with tabs[1]:
        back = st.button("Back to Home")
        if back:
            st.session_state.page = "Home"
            st.rerun()


if __name__ == "__main__":
    main()
