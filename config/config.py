import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

MONGOBD_CONNECTION_STRING = "mongodb+srv://zayd88903:zayd202020@cluster0.mwmxpcy.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
#deepseek
DEEPSEEK_API = 'sk-96d964197f424eefbab7d578e590b335'
# Email configuration
EMAIL_HOST = os.getenv("EMAIL_HOST", "smtp.gmail.com")
EMAIL_PORT = int(os.getenv("EMAIL_PORT", "587"))
EMAIL_USER = os.getenv("EMAIL_USER", "elouaragli.zayd@etu.uae.ac.ma")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD", "fsyk rqyz axog mhpz")