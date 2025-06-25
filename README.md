# Fire Prediction Application

A machine learning application that uses LSTM and FNN hybrid models to predict fire risks based on electrical data.

## Features

- **Hybrid Model Architecture**: Combines LSTM and Feedforward Neural Networks
- **Web Interface**: Flask-based web application with user authentication
- **Model Training**: Train models on electrical data
- **Prediction**: Real-time fire risk prediction
- **Evaluation**: Model performance metrics and visualization
- **User Management**: Registration, login, and profile management

## Installation

### Prerequisites

1. **Python 3.9+**
2. **Graphviz (Optional - for model visualization)**
   ```bash
   # macOS
   brew install graphviz
   
   # Ubuntu/Debian
   sudo apt-get install graphviz
   
   # Windows
   # Download from https://graphviz.org/download/
   ```

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd fire-prediction
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   python app.py
   ```

5. **Access the application**
   Open your browser and navigate to `http://localhost:5000`

## Project Structure

```
fire-prediction/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── database.db           # SQLite database
├── libs/
│   ├── Database.py       # Database operations
│   ├── LSTM_FNN_Model.py # Machine learning model
│   └── data/             # Training data (CSV files)
├── static/               # CSS, JS, and images
├── templates/            # HTML templates
└── *.pkl, *.keras       # Saved models and encoders
```

## Usage

### 1. User Registration/Login
- Register a new account or login with existing credentials
- Manage your profile information

### 2. Model Training
- Navigate to the "Train" page
- Click "Start Training" to train the hybrid model
- Training uses electrical data from CSV files in `libs/data/`

### 3. Model Evaluation
- Go to the "Evaluate" page
- View model performance metrics (loss and accuracy)
- See evaluation plots and charts

### 4. Fire Risk Prediction
- Access the "Predict" page
- Input electrical parameters
- Get real-time fire risk assessment

## Troubleshooting

### Common Issues

1. **Graphviz Error ("dot" not found)**
   ```
   FileNotFoundError: [Errno 2] "dot" not found in path
   ```
   **Solution**: Install Graphviz system package (see Prerequisites)
   
   **Alternative**: The application will continue to work without model visualization

2. **Database Connection Issues**
   - Ensure `database.db` has proper permissions
   - Check if SQLite3 is installed

3. **Model Training Fails**
   - Verify CSV data files exist in `libs/data/`
   - Check data format and completeness
   - Ensure sufficient memory for training

4. **Import Errors**
   - Verify virtual environment is activated
   - Reinstall requirements: `pip install -r requirements.txt`

### Performance Optimization

1. **Memory Usage**
   - Monitor RAM usage during training
   - Reduce batch size if needed
   - Close other applications during training

2. **Training Speed**
   - Use GPU if available (install tensorflow-gpu)
   - Adjust model parameters in `LSTM_FNN_Model.py`

## Development

### Code Quality Enhancements

1. **Error Handling**
   - All model visualization calls are wrapped in try-catch blocks
   - Graceful degradation when Graphviz is unavailable

2. **Logging**
   - Consider adding proper logging instead of print statements
   - Use Python's `logging` module for better control

3. **Configuration**
   - Move hardcoded values to configuration files
   - Use environment variables for sensitive data

4. **Testing**
   - Add unit tests for model functions
   - Implement integration tests for Flask routes

### Security Considerations

1. **Session Management**
   - Use secure session keys in production
   - Implement session timeout

2. **Input Validation**
   - Validate all user inputs
   - Sanitize data before database operations

3. **Database Security**
   - Use parameterized queries (already implemented)
   - Consider encryption for sensitive data

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request


## Support

For issues and questions:
1. Check the troubleshooting section
2. Review error logs
3. Create an issue in the repository