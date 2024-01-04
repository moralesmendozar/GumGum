import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

#This function sends an E-mail when the code is done.
def sendEmail(you,subject,text):
    me = 'caratheodoryc@gmail.com'
    mail = smtplib.SMTP('smtp.gmail.com', 587)
    mail.ehlo()
    mail.starttls()
    mail.login('caratheodoryc', 'constantin29')
    msg = MIMEMultipart('alternative')
    msg['Subject'] = subject
    msg['From'] = me
    msg['To'] = you
    part1 = MIMEText(text, 'plain')
    msg.attach(part1)
    mail.sendmail(me, you, msg.as_string())
    mail.quit()
    print 'email sent'

#sendEmail('moralesmendozar@gmail.com','Your Subject here',"Your message Here")