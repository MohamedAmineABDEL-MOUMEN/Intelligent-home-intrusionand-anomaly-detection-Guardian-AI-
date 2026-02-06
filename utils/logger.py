"""Logging and notification utilities."""
import datetime
from colorama import Fore, Style, init

init(autoreset=True)

class NotificationLogger:
    """Real-time notification and logging system."""
    
    def __init__(self, log_file="logs/system.log"):
        self.log_file = log_file
        self.status = "HOME SECURED"
        
    def _timestamp(self):
        return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def _write_to_file(self, message):
        try:
            with open(self.log_file, "a") as f:
                f.write(f"[{self._timestamp()}] {message}\n")
        except:
            pass
    
    def info(self, message):
        """Log info message."""
        formatted = f"{Fore.BLUE}[INFO]{Style.RESET_ALL} [{self._timestamp()}] {message}"
        print(formatted)
        self._write_to_file(f"[INFO] {message}")
    
    def warning(self, message):
        """Log warning message."""
        formatted = f"{Fore.YELLOW}[WARNING]{Style.RESET_ALL} [{self._timestamp()}] {message}"
        print(formatted)
        self._write_to_file(f"[WARNING] {message}")
    
    def alert(self, message):
        """Log critical alert."""
        formatted = f"{Fore.RED}[ALERT]{Style.RESET_ALL} [{self._timestamp()}] {message}"
        print(formatted)
        self._write_to_file(f"[ALERT] {message}")
    
    def success(self, message):
        """Log success message."""
        formatted = f"{Fore.GREEN}[SUCCESS]{Style.RESET_ALL} [{self._timestamp()}] {message}"
        print(formatted)
        self._write_to_file(f"[SUCCESS] {message}")
    
    def intrusion_detected(self, source, details=""):
        """Log intrusion detection event."""
        self.status = "INTRUSION DETECTED"
        formatted = f"""{Fore.RED}
{'='*60}
              !!! INTRUSION DETECTED !!!
{'='*60}
Source: {source}
Details: {details}
Time: {self._timestamp()}
Status: {self.status}
{'='*60}
{Style.RESET_ALL}"""
        print(formatted)
        self._write_to_file(f"[INTRUSION] Source: {source} | Details: {details}")
    
    def home_secured(self):
        """Reset to secure status."""
        self.status = "HOME SECURED"
        formatted = f"{Fore.GREEN}[SECURE]{Style.RESET_ALL} [{self._timestamp()}] System Status: HOME SECURED"
        print(formatted)
        self._write_to_file("[SECURE] System Status: HOME SECURED")
    
    def get_status(self):
        """Get current security status."""
        return self.status
    
    def mobile_notification(self, title, message, priority="normal"):
        """Simulated mobile notification."""
        icon = "🔔" if priority == "normal" else "🚨"
        formatted = f"""
{Fore.CYAN}┌{'─'*50}┐
│ {icon} MOBILE NOTIFICATION
├{'─'*50}┤
│ Title: {title}
│ Message: {message}
│ Priority: {priority.upper()}
│ Time: {self._timestamp()}
└{'─'*50}┘{Style.RESET_ALL}
"""
        print(formatted)
        self._write_to_file(f"[MOBILE] {title}: {message}")


logger = NotificationLogger()
