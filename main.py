import base64
import streamlit as st
import joblib  # Use joblib for scikit-learn models
import pandas as pd
import matplotlib.pyplot as plt

@st.cache_data
def get_img_as_base64(file):
    try:
        with open(file, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except FileNotFoundError:
        st.error(f"Background image file not found: {file}")
        return None

def add_bg_from_local(image_file):
    img_base64 = get_img_as_base64(image_file)
    if img_base64:
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url("data:image/jpg;base64,{img_base64}");
                background-size: cover;
                background-repeat: no-repeat;
                background-attachment: fixed;
            }}
            .stHeader, .stToolbar {{
                background: rgba(0,0,0,0) !important;
            }}
            .stSidebar > div:first-child {{
                background: rgba(0,0,0,0.2);
            }}
            </style>
            """,
            unsafe_allow_html=True
        )

add_bg_from_local('background.jpg')

teams = ["--select--", "Mumbai Indians", "Chennai Super Kings", "Royal Challengers Bangalore",
         "Delhi Capitals", "Kolkata Knight Riders", "Sunrisers Hyderabad", "Punjab Kings", "Rajasthan Royals"]

venues = ['--select--', 'Wankhede Stadium', 'M Chinnaswamy Stadium', 'Arun Jaitley Stadium', 'Eden Gardens',
          'M. A. Chidambaram Stadium', 'Sawai Mansingh Stadium', 'Feroz Shah Kotla', 'Punjab Cricket Association Stadium, Mohali',
          'Himachal Pradesh Cricket Association Stadium', 'Dr DY Patil Sports Academy', 'Maharashtra Cricket Association Stadium',
          'Holkar Cricket Stadium', 'Rajiv Gandhi Intl. Cricket Stadium', 'Vidarbha Cricket Association Stadium, Jamtha',
          'Brabourne Stadium', 'Sheikh Zayed Stadium', 'Sharjah Cricket Stadium', 'Dubai International Cricket Stadium',
          'Kingsmead', 'Newlands', "St George's Park", 'SuperSport Park', 'Buffalo Park', 'OUTsurance Oval',
          'New Wanderers Stadium', 'De Beers Diamond Oval', 'JSCA International Stadium Complex',
          'Barabati Stadium', 'ACA-VDCA Stadium', 'Sardar Patel Stadium, Motera']

# --- Load the correct model file with joblib ---
try:
    # Loading the RandomForest model (pipe.pkl) which performs better than decision_tree_model.pkl
    pipe = joblib.load('pipe.pkl')
except FileNotFoundError:
    st.error("Model file ('pipe.pkl') not found. Please ensure it's in the same directory and you have run the cleaning_data.py script.")
    st.stop()
except Exception as e:
    st.error(f"Error loading the model: {e}")
    st.stop()

st.markdown(""" # **CRICKET WIN PREDICTOR** """)

col1, col2 = st.columns(2)
with col1:
    batting_team = st.selectbox("Select Batting Team", teams)

with col2:
    if batting_team == "--select--":
        bowling_team = st.selectbox("Select Bowling Team", teams, disabled=True)
    else:
        filtered_teams = [team for team in teams if team != batting_team and team != "--select--"]
        bowling_team = st.selectbox("Select Bowling Team", filtered_teams)

selected_venues = st.selectbox("Select venue", venues)
target = st.number_input("Target", min_value=0, step=1)

col1, col2, col3 = st.columns(3)

with col1:
    score = st.number_input("Score", min_value=0, step=1)

with col2:
    overs = st.number_input("Overs Completed", min_value=0.0, max_value=20.0, step=0.1, format="%.1f")

with col3:
    wickets = st.number_input("Wickets Down", min_value=0, max_value=10, step=1)

if st.button("PREDICT WINNING PROBABILITY"):
    # --- Input Validation ---
    if batting_team == "--select--" or bowling_team == "--select--" or selected_venues == "--select--":
        st.error("Please select all teams and a venue.")
    elif target <= 0:
        st.error("Target must be greater than 0.")
    elif score > target:
        st.error("Score cannot be greater than the target.")
    elif overs == 0 and score > 0:
        st.error("Score cannot be greater than 0 if no overs have been completed.")
    elif round((overs - int(overs)) * 10) > 5:
        st.error("Invalid overs format. The number of balls (after the decimal) cannot be more than 5.")
    else:
        win_prob, loss_prob = 0, 0
        try:
            # --- Data Processing ---
            runs_left = int(target - score)
            
            completed_overs = int(overs)
            completed_balls = round((overs - completed_overs) * 10) # More robust ball calculation
            balls_left = 120 - (completed_overs * 6 + completed_balls)
            
            wickets_remaining = 10 - int(wickets)
            
            total_balls_bowled = 120 - balls_left
            crr = (score * 6 / total_balls_bowled) if total_balls_bowled > 0 else 0
            
            # --- Handle Edge Case: No Balls Left ---
            if balls_left == 0:
                if runs_left > 0:
                    win_prob = 0.0 # Batting team loses
                else:
                    win_prob = 1.0 # Batting team wins (or ties)
                loss_prob = 1.0 - win_prob
            else:
                rrr = (runs_left * 6 / balls_left)

                # --- Create DataFrame for prediction ---
                input_data = pd.DataFrame({
                    'batting_team': [batting_team],
                    'bowling_team': [bowling_team],
                    'venue': [selected_venues],
                    'runs_left': [runs_left],
                    'balls_left': [balls_left],
                    'wickets_remaining': [wickets_remaining],
                    'total_runs_x': [int(target)],
                    'CRR': [crr],
                    'RRR': [rrr]
                })

                # --- Prediction ---
                result = pipe.predict_proba(input_data)
                loss_prob = result[0][0]
                win_prob = result[0][1]
            
            # --- Display Results ---
            st.header(f"{batting_team}: {round(win_prob * 100)}%")
            st.header(f"{bowling_team}: {round(loss_prob * 100)}%")

            # --- Pie Chart Visualization ---
            labels = [batting_team, bowling_team]
            sizes = [win_prob, loss_prob]
            team_colors = {
                'Mumbai Indians': 'blue', 'Chennai Super Kings': 'yellow', 'Royal Challengers Bangalore': 'red',
                'Delhi Capitals': 'cyan', 'Kolkata Knight Riders': 'purple', 'Sunrisers Hyderabad': 'orange',
                'Punjab Kings': 'gold', 'Rajasthan Royals': 'pink'
            }
            if batting_team != '--select--' and bowling_team != '--select--':
                colors = [team_colors.get(team, 'gray') for team in labels]

                fig, ax = plt.subplots()
                ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors,
                       textprops={'color':"w", 'weight':"bold"})
                ax.axis('equal')
                fig.patch.set_facecolor('none')
                ax.set_facecolor('none')

                st.pyplot(fig, use_container_width=True)

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
