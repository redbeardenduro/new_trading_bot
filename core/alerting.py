import smtplib
from email.mime.text import MIMEText

from core.config import BotConfig


class Alerter:

    def __init__(self, config: BotConfig) -> None:
        self.config = config
        self.email_config = config.get("alerting.email", {})

    def send_alert(self, subject: str, message: str) -> None:
        if self.email_config.get("enabled"):
            self.send_email_alert(subject, message)

    def send_email_alert(self, subject: str, message: str) -> None:
        try:
            msg = MIMEText(message)
            msg["Subject"] = subject
            msg["From"] = self.email_config["from_email"]
            msg["To"] = ", ".join(self.email_config["to_emails"])
            with smtplib.SMTP(
                self.email_config["smtp_server"], self.email_config["smtp_port"]
            ) as server:
                server.starttls()
                server.login(self.email_config["smtp_user"], self.email_config["smtp_password"])
                server.send_message(msg)
                print(f"Sent email alert: {subject}")
        except Exception as e:
            print(f"Failed to send email alert: {e}")
