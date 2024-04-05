#!/bin/bash

echo "Running type checker for csle-agents"
cd csle-agents; mypy src tests; cd ../
echo "Running type checker for csle-attacker"
cd csle-attacker; mypy src tests; cd ../
echo "Running type checker for csle-collector"
cd csle-collector; mypy src tests; cd ../
echo "Running type checker for csle-common"
cd csle-common; mypy src tests; cd ../
echo "Running type checker for csle-defender"
cd csle-defender; mypy src tests; cd ../
echo "Running type checker for csle-rest-api"
cd csle-rest-api; mypy src tests; cd ../
echo "Running type checker for csle-ryu"
cd csle-ryu; mypy src tests; cd ../
echo "Running type checker for csle-system-identification"
cd csle-system-identification; mypy src tests; cd ../
echo "Running type checker for gym-csle-stopping-game"
cd gym-csle-stopping-game; mypy src tests; cd ../
echo "Running type checker for gym-csle-intrusion-response-game"
cd gym-csle-intrusion-response-game; mypy src tests; cd ../
echo "Running type checker for gym-csle-apt-game"
cd gym-csle-apt-game; mypy src tests; cd ../
echo "Running type checker for csle-tolerance"
cd csle-tolerance; mypy src tests; cd ../
echo "Running type checker for gym-csle-cyborg"
cd gym-csle-cyborg; mypy src tests; cd ../
echo "Running type checker for csle-base"
cd csle-base; mypy src; cd ../
echo "Running type checker for csle-cli"
cd csle-cli; mypy src; cd ../
