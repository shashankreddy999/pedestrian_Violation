import smtplib, os
import mysql.connector
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

s = smtplib.SMTP('smtp.gmail.com', 587)
s.starttls()
sender_email = "saivenkat232001@gmail.com"
password = "wqsbebmgqqtjijwx"
s.login(sender_email, password)

body = """\
Hello,

You've violated road safety rules.
Please pay a challan of $500/-.
"""

message = MIMEMultipart()
message["From"] = sender_email
message["Subject"] = "PEDESTRIAN VIOLATION"

conn = mysql.connector.connect(
  host="localhost",
  user="root",
  password="root",
  database="test"
)
print(conn)
cursor = conn.cursor()

path='violations'

for imageName in os.listdir(path):
    id = int(imageName[0:-4])
    print(id)
    
    cursor.execute("""select id, name, email, sent from people where id=%d"""%id)
    rows = cursor.fetchall()
    print(rows)

    if(rows[0][3] == 0):
        cursor.execute("""update people set sent=1 where id=%d"""%id)
        conn.commit()
        message["To"] = rows[0][2]
        message.attach(MIMEText(body, "plain"))

        with open(path + "/" + imageName, 'rb') as f:
            
            mime = MIMEBase('image', 'png', filename=imageName)
           
            mime.add_header('Content-Disposition', 'attachment', filename=imageName)
            mime.add_header('X-Attachment-Id', '0')
            mime.add_header('Content-ID', '<0>')
            # read attachment file content into the MIMEBase object
            mime.set_payload(f.read())
            # encode with base64
            encoders.encode_base64(mime)
            # add MIMEBase object to MIMEMultipart object
            message.attach(mime)
        
        s.sendmail(sender_email, rows[0][2], message.as_string())
s.quit()