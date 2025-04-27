import smtplib
from email.message import EmailMessage
import io
from typing import Optional, List
import matplotlib.pyplot as plt

EMAIL_USER ="elouaragli.zayd@etu.uae.ac.ma"
EMAIL_PASSWORD="fsyk rqyz axog mhpz"
def ai_email_agent(contact: dict, message: str, type_: str = "Report", plot_figs: Optional[List[plt.Figure]] = None):
    email = contact["email"]
    subject = f"Medical Monitoring - {type_}"

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = EMAIL_USER  
    msg["To"] = email
    msg.set_content(message)

    # Attach plots if any
    if plot_figs:
        for idx, fig in enumerate(plot_figs):
            img_bytes = fig.to_image(format="png")  # <-- This generates PNG bytes directly
            msg.add_attachment(
                img_bytes,
                maintype="image",
                subtype="png",
                filename=f"plot_{idx+1}.png"
            )

    # Send email
    with smtplib.SMTP("smtp.gmail.com", 587) as smtp:
        smtp.starttls()
        smtp.login(EMAIL_USER, EMAIL_PASSWORD)  
        smtp.send_message(msg)
