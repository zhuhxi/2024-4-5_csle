#!/bin/bash

/var/ossec/bin/ossec-control start
service named start
service ntp restart
service rsyslog restart
/usr/sbin/sshd -D &
tail -f /dev/null
