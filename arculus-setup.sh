#!/bin/bash

echo "Starting setup process..."

# Grant execute permissions and run the setup script
sudo chmod +x setup_controller.sh
echo "Running initial setup script..."
./setup_controller.sh

kubectl apply -f default-deny.yaml

echo "Enter database password:"
read db_password
sed -i "s/MYSQL_ROOT_PASSWORD vimanlab/MYSQL_ROOT_PASSWORD $db_password/" arculus-gcs-mysql/Dockerfile
sed -i "s/\"password\": \"vimanlab\"/\"password\": \"$db_password\"/" arculus-gcs-node/configs/dbconfigs.json
sed -i "s/\"password\": \"vimanlab\"/\"password\": \"$db_password\"/" arculus-gcs-node/configs/honeypot_config.json

echo "Installing Docker and NPM..."
cd arculus-gcs-mysql
sudo apt update
sudo apt install docker.io npm -y
echo "Building Docker container for MySQL..."
sudo docker build -t arculus-gcs-mysql:latest .
echo "Starting MySQL container in detached mode..."
sudo docker run -d -p 3306:3306 --name arculus-gcs-db arculus-gcs-mysql
cd ..

echo "Enter a 256-bit encryption key:"
read encryption_key
echo "$encryption_key" > arculus-gcs-node/configs/ENCRYPTION_SECRET.txt

# Automatically fetch public and private IP addresses
public_ip=$(curl -s http://ipecho.net/plain)
private_ip=$(hostname -I | awk '{print $1}')  # Adjust based on your network setup

cd arculus-gcs-ui
sed -i "s/<public ip>/$public_ip/" src/config.js
sed -i "s/<private ip>/$private_ip/" src/config.js
cd ..

echo "Enter CHN Server URL (leave blank if not available):"
read chn_url
if [[ ! -z "$chn_url" ]]; then
  sed -i "s|https://<chn domain>|$chn_url|" arculus-gcs-ui/src/config.js
  sed -i "s|<honeypot domain>|$chn_url|" arculus-gcs-node/configs/honeypot_config.json
fi

echo "Enter CHN API key (leave blank if not available):"
read chn_api_key
if [[ ! -z "$chn_api_key" ]]; then
  sed -i "s/\"apikey\": \"36b37e50254447d7837c813fd1f4fb3d\"/\"apikey\": \"$chn_api_key\"/" arculus-gcs-node/configs/honeypot_config.json
fi

echo "Enter CHN deploy key (leave blank if not available):"
read chn_deploy_key
if [[ ! -z "$chn_deploy_key" ]]; then
  sed -i "s/\"deployKey\": \"0bKeXjBW\"/\"deployKey\": \"$chn_deploy_key\"/" arculus-gcs-node/configs/honeypot_config.json
fi

echo "Installing PM2 globally..."
sudo npm install -g pm2

echo "Setting up Node.js application..."
cd arculus-gcs-node
npm install
npm audit fix  # Optionally run audit fix
echo "Starting Node.js application with PM2..."
pm2 start index.js --name node-app --watch
cd ..

echo "Setting up UI server..."
cd arculus-gcs-ui
npm install
echo "Building UI..."
npm run build  # Builds the React application
echo "Starting UI server with PM2..."
pm2 start server.js --name ui-server --watch
cd ..

echo "Arculus successfully setup! Access the Arculus UI at http://${public_ip}:3000"
