# DDyn - Dynamic DNS Updater

DDyn is a dynamic DNS updater application designed to automatically update DNS records when the public IP address changes. The application leverages machine learning to predict IP changes and monitor connectivity, ensuring reliable updates to DNS services.

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Machine Learning Integration](#machine-learning-integration)
- [License](#license)

## Features
- **Dynamic DNS Updates**: Automatically updates DNS records when the public IP address changes using DNS-O-Matic as the provider. This allows users to manage multiple DNS records across various services from a single interface.
- **Machine Learning Predictions**: Utilizes machine learning models to predict IP changes based on historical data.
- **Anomaly Detection**: Identifies unusual patterns in IP changes to enhance reliability.
- **Command-Line Interface**: Provides a simple CLI for managing updates and models.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/iap/ddyn.git
   cd ddyn
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux/Mac
   # or
   .venv\Scripts\activate     # Windows
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Configure the environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your DNS-O-Matic credentials and other settings
   ```

## Usage
1. Start the DNS updater:
   ```bash
   python -m src.main
   ```

2. Manage machine learning models using the CLI:
   ```bash
   python -m src.cli train          # Train the ML models
   python -m src.cli status         # Check model status
   python -m src.cli predict        # Get predictions
   ```

## Configuration
- **Environment Variables**: Configure your `.env` file with the following variables:
  ```plaintext
  DNSOMATIC_USERNAME=your_username
  DNSOMATIC_PASSWORD=your_password
  DNSOMATIC_HOSTNAME=your_hostname
  ```

## Machine Learning Integration
- The application uses machine learning models to predict IP changes and enhance connection reliability.
- Current models include:
  - **Random Forest Classifier**: For predicting IP changes based on historical data.
  - **Isolation Forest**: For detecting anomalies in IP change patterns.

## License
This project is licensed under the GNU General Public License (GPL) Version 3. See the [LICENSE](./LICENSE) file for details.