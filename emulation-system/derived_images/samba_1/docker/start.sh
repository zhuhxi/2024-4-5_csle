#!/bin/bash

/usr/sbin/sshd -D &
service ntp restart
service rsyslog restart
/usr/local/samba/sbin/smbd -F -S -s /smb.conf --debuglevel=10
tail -f /dev/null
