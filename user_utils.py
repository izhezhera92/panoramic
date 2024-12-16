import subprocess
import smtplib, ssl
from email.mime.text import MIMEText


def send_report_to_email(message = None, time_start = None, image = None):
    port = 465 
    # Учетные данные
    email_address = "leica.ngc@gmail.com"
    app_password = "ldxv ytms ujrc izdv"  

    # Настройка сообщения
    msg = MIMEText(message)
    msg['Subject'] = "Report"
    msg['From'] = "leica.ngc@gmail.com"
    msg['To'] = "ivanzhezhera92@gmail.com"

    # Подключение к серверу SMTP
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", port) as server:
            server.login(email_address, app_password)
            server.sendmail(email_address, "ivanzhezhera92@gmail.com", msg.as_string())

    except smtplib.SMTPAuthenticationError as e:
        print(f"Authentification error: {e}")
