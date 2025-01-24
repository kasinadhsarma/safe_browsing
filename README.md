# Safe Browsing Project

## Overview

The Safe Browsing Project is a comprehensive solution designed to enhance online safety by classifying and blocking inappropriate, malicious, and harmful URLs. This project consists of a backend machine learning model, a frontend web application, and a Chrome extension that integrates seamlessly with the user's browsing experience.

## Project Structure

The project is organized into several key directories:

- **app/**: Contains the frontend web application built with Next.js.
- **backend/**: Houses the backend server and machine learning models.
- **chrome-extension/**: Includes the Chrome extension that interacts with the browser.
- **components/**: Shared UI components used across the frontend and Chrome extension.
- **hooks/**: Custom React hooks used in the frontend.
- **lib/**: Utility functions and helpers.
- **ml/**: Machine learning models and related scripts.
- **public/**: Static assets like images and logos.
- **styles/**: Global CSS styles for the frontend.

## Getting Started

### Prerequisites

- Node.js (v14 or later)
- Python (v3.8 or later)
- npm or yarn
- Chrome browser

### Installation

1. **Clone the repository:**

   ```bash
   cd safe-browsing
   ```

2. **Install dependencies:**

   ```bash
   npm install
   ```

3. **Set up the backend:**

   - Navigate to the `backend` directory:

     ```bash
     cd backend
     ```

   - Create a virtual environment and activate it:

     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```

   - Install Python dependencies:

     ```bash
     pip install -r requirements.txt
     ```

   - Run the backend server:

     ```bash
     python main.py
     ```

4. **Set up the frontend:**

   - Navigate to the root directory of the project:

     ```bash
     cd ..
     ```

   - Run the development server:

     ```bash
     npm run dev
     ```

5. **Load the Chrome extension:**

   - Open Chrome and go to `chrome://extensions/`.
   - Enable "Developer mode" by toggling the switch in the top right.
   - Click "Load unpacked" and select the `chrome-extension` directory.

### Usage

1. **Backend:**

   The backend server handles URL classification and provides APIs for the frontend and Chrome extension. It uses a machine learning model trained on various types of URLs.

2. **Frontend:**

   The frontend web application allows users to authenticate, view their browsing activity, and manage settings. It communicates with the backend server to fetch and update data.

3. **Chrome Extension:**

   The Chrome extension blocks inappropriate and malicious URLs based on the classification provided by the backend. It also provides a popup interface for users to report URLs and view their browsing history.

## File Descriptions

### Backend

- **main.py**: The main entry point for the backend server.
- **requirements.txt**: Python dependencies for the backend.
- **ml/**: Machine learning models and related scripts.
  - **ai/**: Contains the URL classification model and training scripts.
    - **training.py**: Script to train the URL classification model.
    - **url_classifier_final.pth**: Trained URL classification model.
    - **url_dataset.csv**: Dataset used for training the model.
    - **lightning_logs/**: Logs and checkpoints from the training process.

### Frontend

- **app/**: Next.js application structure.
  - **auth/**: Authentication-related pages and components.
  - **dashboard/**: Dashboard pages and components.
  - **api/**: API routes for the frontend.
  - **components/**: Shared UI components.
  - **hooks/**: Custom React hooks.
  - **lib/**: Utility functions and helpers.
  - **styles/**: Global CSS styles.

### Chrome Extension

- **manifest.json**: Configuration file for the Chrome extension.
- **src/**: Source files for the Chrome extension.
  - **scripts/**: JavaScript files for the background, content, and popup scripts.
  - **styles/**: CSS files for the popup and other UI components.
  - **types/**: TypeScript type definitions.
  - **icons/**: Icon assets for the extension.

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Make your changes and commit them with descriptive messages.
4. Push your changes to your fork.
5. Create a pull request to the main repository.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For any questions or support, please contact [your email address] or open an issue on the GitHub repository.
