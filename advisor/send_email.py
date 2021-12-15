
import os
import smtplib, ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import date

from dotenv import load_dotenv

import util

load_dotenv()



def send_daily_update():
    destination = os.getenv('EMAIL_TO')
    source = os.getenv('EMAIL_FROM')
    password = os.getenv('EMAIL_PASSWORD')
    
    today = date.today()
    date_today = today.strftime("%Y-%m-%d")
    
    msg = MIMEMultipart("alternative")
    msg['Subject'] = f"Daily Update: {date_today}"
    msg['From'] = source
    msg['To'] = destination

    # Send the message via our own SMTP server.
    smtp_server = smtplib.SMTP('smtp.gmail.com', 587)
    smtp_server.ehlo() #setting the ESMTP protocol
    smtp_server.starttls() #setting up to TLS connection
    smtp_server.ehlo() #calling the ehlo() again as encryption happens on calling startttls()

    smtp_server.login(source,password) #logging into out email id
    
    pred_fallers = util.read(f"data/stocks/predictions/fallers/2021-12-13.json")
    pred_risers = util.read(f"data/stocks/predictions/risers/2021-12-13.json")

    msg_to_be_sent =f'''
    Hello {destination}!
    Today's updates are as follows:
    {make_prediction_string(pred_fallers)}
    {make_prediction_string(pred_risers)}

    Good luck, and happy trading!
    - AI Advisor
    '''

    #sending the mail by specifying the from and to address and the message 
    smtp_server.sendmail(source,destination,msg_to_be_sent)
    print('Successfully the mail is sent') #priting a message on sending the mail
    smtp_server.quit()#terminating the server

def make_prediction_string(predictions:dict) -> str:
    output = "The projected fallers are: \n"
    for key in predictions:
        output += f"ID: {key}. Previous Close: {predictions[key]['previous']} Predicted Close: {predictions[key]['predicted']} \n"
    return output

if __name__ == "__main__":
    send_daily_update()