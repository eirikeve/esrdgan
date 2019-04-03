"""
loggingclass.py
Written by Eirik Vesterkjær, 2019
Apache License

Implements a base class for storing logs without a logging.Logger
"""


class GlobalLoggingClass:
    """
    LoggingClass has a list of status logs which can be fetched & cleared 
    upon calling get_new_status_logs()
    To make logs, just append to the list
    Logs are global, so all subclasses log to the same list
    """
    status_logs = []

    def get_new_status_logs(self):
        logs = [log for log in self.status_logs]
        self.status_logs.clear()
        return logs        
