import smtplib
from email.message import EmailMessage
import io
from typing import Optional, List
import matplotlib.pyplot as plt

EMAIL_USER ="elouaragli.zayd@etu.uae.ac.ma"
EMAIL_PASSWORD="fsyk rqyz axog mhpz"

message= f'test test'
email = 'annabell01@pperspe.com'
subject = f"Medical Monitoring -"

msg = EmailMessage()
msg["Subject"] = subject
msg["From"] = EMAIL_USER
msg["To"] = email
msg.set_content(message)

with smtplib.SMTP("smtp.gmail.com", 587) as smtp:
    smtp.starttls()
    smtp.login(EMAIL_USER, EMAIL_PASSWORD)  
    smtp.send_message(msg)