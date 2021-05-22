from flask import Flask, render_template, request, jsonify
import aiml
import os
import mnbpy
import logpy
import pickle
import pandas as pd
import csv
import smtplib, ssl, email
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from email.mime.text import MIMEText
from itertools import count
from textblob import TextBlob
x = count(start=0, step=1)
mnb = pickle.load(open('mnb.pickle', 'rb'))
log = pickle.load(open('log.pickle', 'rb'))

def writeunique(message):

    message=message.lower()
    message = ' '.join(str(TextBlob(str(word)).correct()) for word in message.split())
    try:
        with open('data.csv') as csv_f:
            csv_f = csv.reader(csv_f)
            for row in csv_f:
                if(row[0]==message):
                    return

        with open('newdata.csv', 'a', newline="") as fi:
            csv_w = csv.writer(fi)
            csv_w.writerow([message])
            fi.close()

        with open('data.csv', 'a', newline="") as file:
            csv_w = csv.writer(file)
            csv_w.writerow([message])
            file.close()

    except :
        with open('newdata.csv', 'a', newline="") as file:
            csv_w = csv.writer(file)
            csv_w.writerow([message])
            file.close()

        with open('data.csv', 'a', newline="") as fi:
            csv_w = csv.writer(fi)
            csv_w.writerow([message])
            fi.close()


def sendmail():

    port = 465  
    smtp_server = "smtp.gmail.com"
    sender_email = "test12mail98@gmail.com"  
    receiver_email = "vidyapunjabi7@gmail.com"  
    password = "testmail!@98"

    messagemail = MIMEMultipart()
    messagemail['Subject'] = "Admission enquiry"

    with open('newdata.csv', 'r') as read_obj:
        csv_reader = csv.reader(read_obj)
        for row in csv_reader:
            body_part = MIMEText(row[0]+"\n", 'plain')
            messagemail.attach(body_part)
        read_obj.close()

    context = ssl.create_default_context()
    with smtplib.SMTP_SSL(smtp_server, port, context=context) as server:
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, messagemail.as_string())

app = Flask(__name__)

@app.route("/")
def hello():
    return render_template('chat.html')

@app.route("/ask", methods=['POST'])
def ask():

    message =request.form['messageText'].strip()

    kernel = aiml.Kernel()

    if os.path.isfile("bot_brain.brn"):
        kernel.bootstrap(brainFile = "bot_brain.brn")
    else:
        kernel.bootstrap(learnFiles = os.path.abspath("aiml/startup.xml"), commands = "load aiml b")
        kernel.saveBrain("bot_brain.brn")

    while True:
        bot_response = kernel.respond(message)
        if (bot_response == "NULL"):
            testa=mnbpy.clean_text(message)
            if(len(testa.split())>1):
                X_test=pd.Series(testa)
                y_prednb = mnb.predict(X_test)
                y_predlog = log.predict(X_test)
                logprob=log.predict_proba(X_test).max()
                nbprob=mnb.predict_proba(X_test).max()
                print(y_prednb,y_predlog,nbprob,logprob)
                if(logprob>nbprob ):
                    pred=y_predlog[0]
                    y_pred=logprob
                else:
                    pred=y_prednb[0]
                    y_pred=nbprob

                if((y_pred>0.8) or (y_prednb==y_predlog and logprob>0.5 and nbprob>0.5)):
                    csv_f = csv.reader(open('answers.csv',encoding='utf-8'))
                    for row in csv_f:
                        if(row[0]==str(y_prednb[0])):
                            gt=row[1]
                            print (row[1])

                elif(next(x)<2):                
                    gt="Please provide more information"

                else:
                    gt="Please contact admission cell:<br> 022-61082400"
                    writeunique(message)

                    with open('newdata.csv') as check:
                        csv_r = csv.reader(check)
                        if(len(list(csv_r))>=10):
                            sendmail()
                            check.close()
                            os.remove("newdata.csv")
            else:
                gt="Please provide more information. Insufficient words."
            return jsonify({'status':'OK','answer':'$'+gt})

        else:
            return jsonify({'status':'OK','answer':bot_response})


if __name__ == "__main__":
    app.run()
