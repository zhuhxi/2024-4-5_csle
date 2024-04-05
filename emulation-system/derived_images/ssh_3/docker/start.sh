#!/bin/bash

/var/ossec/bin/ossec-control start
service rsyslog restart
nohup /usr/sbin/inspircd --runasroot --debug --nopid & > irc.log
rethinkdb --bind all --bind-http all --bind-emulation all &
/usr/sbin/sshd -D &
tail -f /dev/null
