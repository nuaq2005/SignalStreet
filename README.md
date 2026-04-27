# SignalStreet

Small price changes in stocks can mislead prediction models into overestimating trends, causing investors to make poor decisions. SignalStreet is a full-stack web application that filters out market noise and identifies genuine signals for buy or sell actions, providing investors with reliable insights.

## Features
[*Devpost Link*](https://devpost.com/software/signalstreet)
- **Noise Filtering**: Advanced algorithms to distinguish between market noise and true signals.
- **Real-time Analysis**: Fetches live stock data and applies machine learning models for predictions.
- **Interactive Dashboard**: User-friendly React-based frontend for inputting stock tickers and viewing analysis results.
- **Robust Backend**: Flask API with data processing, ML inference, and observability.
- **Comprehensive Testing**: Automated testing suite using pytest and hypothesis for reliability.

## Architecture

SignalStreet follows a client-server architecture with a clear separation of concerns:

- **Frontend**: React application built with Vite, handling user interactions and displaying results.
- **Backend**: Flask REST API serving analysis requests, integrating data fetching, ML predictions, and evaluation.
- **Data Layer**: Handles stock data retrieval using yfinance.
- **ML Model**: Pre-trained model loaded on-demand for predictions.
- **Testing**: Separate testing suite for backend components.
- [**The Model**]

### Architecture Diagram

```
┌─────────────────┐
│   Frontend      │
│   (React/Vite)  │
└─────────────────┘
         │
         ▼
┌─────────────────┐
│   Backend       │
│   (Flask API)   │
└─────────────────┘
         │
    ┌────┼────┐
    │    │    │
    ▼    ▼    ▼
┌─────┐ ┌─────┐ ┌─────┐
│Data │ │ ML  │ │Eval │
│Layer│ │Model│ │uator│
└─────┘ └─────┘ └─────┘
```

## Technologies Used

### Backend
- **Python 3.11.9**: Core programming language.
- **Flask 3.1.3**: Web framework for the REST API.
- **pandas & numpy**: Data manipulation and numerical computing.
- **yfinance**: Library for fetching stock market data.
- **joblib**: Model serialization and loading.
- **pytest & hypothesis**: Testing frameworks for unit and property-based tests.

### Frontend
- **React 19.2.5**: JavaScript library for building user interfaces.
- **Vite**: Build tool and development server for fast development.

### Development Tools
- **Git**: Version control.
- **npm**: Package manager for frontend dependencies.

## Setup and Installation

### Prerequisites
- Python 3.11.9 or later
- Node.js and npm
- Git

### Backend Setup
1. Navigate to the backend directory:
   ```bash
   cd signal-street/backend
   ```

2. Install Python dependencies:
   ```bash
   pip install flask pandas numpy yfinance joblib pytest hypothesis
   ```

3. (Optional) Install Flask-CORS if not included:
   ```bash
   pip install flask-cors
   ```

### Frontend Setup
1. Navigate to the frontend directory:
   ```bash
   cd signal-street/frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

## Running the Application

### Backend
1. From the backend directory, run the API server:
   ```bash
   python api.py
   ```
   The server will start on port 5000 (or the next available port).

### Frontend
1. From the frontend directory, start the development server:
   ```bash
   npm run dev
   ```
   The app will be available at `http://localhost:5173` (or similar).

2. Open your browser and navigate to the provided URL to access the dashboard.

### Running Tests
1. From the backend directory, run the test suite:
   ```bash
   python -m pytest Testing/
   ```

## Usage

1. Open the frontend in your browser.
2. Enter a valid stock ticker (e.g., "AAPL").
3. Click "Analyze" to fetch data and get predictions.
4. View the results on the dashboard, which will indicate buy/sell signals based on filtered analysis.

## Key Design Decisions

- **Lazy Loading for ML Model**: The ML model is loaded on-demand to reduce startup time and memory usage, improving application responsiveness.
- **Input Validation and Security**: Implemented regex-based validation for stock tickers and security headers (CSP, XSS protection) to prevent injection attacks and ensure data integrity.
- **CORS Restrictions**: Limited to localhost origins during development to enhance security.
- **Error Handling and Logging**: Comprehensive try-except blocks with logging to handle failures gracefully and aid debugging without exposing sensitive information.
- **Timeout Handling**: Frontend API calls include timeouts to prevent hanging requests and improve user experience.
- **Separation of Concerns**: Modular backend components (data layer, ML model, evaluator) for maintainability and scalability.
- **Testing Strategy**: Property-based testing with hypothesis to ensure robustness against edge cases.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your changes. Ensure all tests pass before submitting.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
