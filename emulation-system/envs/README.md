# Emulation environments

This folder contains emulation environments.

- Version: **030** [./030](040):
  - *Level 1* [level_1](040/level_1): A simple infrastructure with 6 nodes and weak-password-vulnerabilities
  - *Level 2* [level_2](040/level_2): An infrastructure with 12 nodes and weak-password-vulnerabilities
  - *Level 3* [level_3](040/level_3): A complex infrastructure with 33 nodes and weak-password-vulnerabilities
  - *Level 4* [level_4](040/level_4): A simple infrastructure with 6 nodes and weak-password-vulnerabilities and IDS monitoring
  - *Level 5* [level_5](040/level_5): An infrastructure with 12 nodes and weak-password-vulnerabilities and IDS monitoring
  - *Level 6* [level_6](040/level_6): A complex infrastructure with 33 nodes and weak-password-vulnerabilities and IDS monitoring
  - *Level 7* [level_7](040/level_7): An infrastructure with 15 nodes and several vulnerabilities: SambaCry, Shellshock, CVE-2015-1427, CVE-2015-3306, CVE-2016-100033_1,and SQL injection., as well as SSH, FTP, Telnet servers that can be compromised using dictionary attacks
  - *Level 8* [level_8](040/level_8): An infrastructure with 26 nodes and several vulnerabilities: SambaCry, Shellshock, CVE-2015-1427, CVE-2015-3306, CVE-2016-100033_1,and SQL injection., as well as SSH, FTP, Telnet servers that can be compromised using dictionary attacks
  - *Level 9* [./level_9](040/level_9): A complex infrastructure with 36 nodes and and several vulnerabilities: SambaCry, Shellshock, CVE-2015-1427, CVE-2015-3306, CVE-2016-100033_1, SQL injection, and brute-force vulnerabilities. Further, the infrastructure has IDS monitoring
  - *Level 10* [./level_10](040/level_10): An infrastructure with 16 nodes and several vulnerabilities: SambaCry, Shellshock, CVE-2015-1427, CVE-2015-3306, CVE-2016-100033_1,and SQL injection., Pengine Server RCE Exploit, as well as SSH, FTP, Telnet servers that can be compromised using dictionary attacks
  - *Level 11* [./level_11](040/level_11): A complex infrastructure with 36 nodes and and several vulnerabilities: SambaCry, Shellshock, CVE-2015-1427, CVE-2015-3306, CVE-2016-100033_1, SQL injection, and brute-force vulnerabilities. Further, the infrastructure has IDS monitoring
  - *Level 12* [./level_12](040/level_12): A simple software-defined networking environment

## Useful commands:

- Install all emulations:
  ```bash
  make install
   ```

- Uninstall all emulations:
  ```bash
  make uninstall
   ```

- Clean the configuration of all emulations:
  ```bash
  make clean_config
   ```

## Author & Maintainer

Kim Hammar <kimham@kth.se>

## Copyright and license

[LICENSE](../LICENSE.md)

Creative Commons

(C) 2020-2024, Kim Hammar