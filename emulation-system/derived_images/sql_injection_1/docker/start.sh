#!/bin/bash

/usr/sbin/sshd -D &
service rsyslog restart
/main.sh &
nohup /usr/sbin/inspircd --runasroot --debug --nopid & > irc.log
sleep 5
/setup_db.sh
sleep 10
# Retry
/setup_db.sh
sleep 10
# Retry
/setup_db.sh
tail -f /dev/null
