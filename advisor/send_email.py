
import os
import smtplib, ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import date
from email.message import EmailMessage

from dotenv import load_dotenv

import util

load_dotenv()

def send_daily_update():
    destination = os.getenv('EMAIL_TO')
    source = os.getenv('EMAIL_FROM')
    password = os.getenv('EMAIL_PASSWORD')
    
    today = date.today()
    date_today = today.strftime("%Y-%m-%d")
    
    msg = EmailMessage()
    msg['Subject'] = f"Daily Stock Advisor Update: {date_today}"
    msg['From'] = source
    msg['To'] = destination

    # Send the message via our own SMTP server.
    smtp_server = smtplib.SMTP('smtp.gmail.com', 587)
    smtp_server.ehlo() #setting the ESMTP protocol
    smtp_server.starttls() #setting up to TLS connection
    smtp_server.ehlo() #calling the ehlo() again as encryption happens on calling startttls()

    smtp_server.login(source,password) #logging into out email id
    
    pred_fallers = util.read(f"data/stocks/predictions/fallers/{date_today}.json")
    pred_risers = util.read(f"data/stocks/predictions/risers/{date_today}.json")

    content = f'''\
    <html>
        <head></head>
        <body>
        <h3>Hello {destination}!</h3>
        <b>Today's updates are as follows:</b>
        <p>
        {make_prediction_string(pred_fallers)}
        {make_prediction_string(pred_risers)}
        Good luck, and happy trading!
        - AI Advisor
        </p>
        </body>
    </html>
    '''

    msg.set_content(MIMEText(content, 'html', 'utf-8'))


    try:
        # Send the message via our own SMTP server.
        server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
        server.login(source, password)
        server.send_message(msg)
        print('Mail sent successfully.') 
        smtp_server.quit()
    except SMTPException:
        print(" Unable to send mail")

def make_prediction_string(predictions:dict) -> str:
    output = "<h3>The projected fallers are:</h3> \n"
    for key in predictions:
        output += f"<b>{key}</b>:<br>Previous: <b>{predictions[key]['previous']}</b><br> Predicted: <b>{predictions[key]['predicted']}</b><br>"
    return output

if __name__ == "__main__":
    send_daily_update()