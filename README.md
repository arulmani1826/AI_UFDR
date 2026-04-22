# UFDR AI Project

A forensic data analysis system for UFDR (Universal Forensic Device Report) device data with AI-powered insights and threat detection.

## Features

- **Data Processing**: Automated preprocessing of messages, call logs, contacts, and GPS data
- **Suspicious Activity Detection**: AI-powered detection of suspicious messages, calls, and contacts
- **Network Graph Analysis**: Visualization of communication networks and interactions
- **Risk Scoring**: Overall risk assessment from device forensic data
- **PDF Reports**: Generate forensic analysis reports
- **Chatbot Integration**: Groq-powered AI chatbot for interactive data queries

## Project Structure

```
ufdr_ai_project/
├── app/
│   ├── app.py              # Flask web application
│   ├── templates/          # HTML templates
│   └── static/             # CSS and static assets
├── data/
│   ├── *.csv               # Sample datasets
│   ├── *.json              # Sample UFDR datasets
│   └── uploads/            # User uploaded files (excluded from git)
├── utils/
│   ├── data_processing.py  # Data preprocessing & analysis
│   ├── models_ai.py        # ML models, chatbot, graph analysis
│   └── reporting.py        # Report generation utilities
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/ufdr_ai_project.git
   cd ufdr_ai_project
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   ```bash
   cp .env.example .env
   # Edit .env and add your GROQ_API_KEY
   ```

5. **Run the Flask app**
   ```bash
   python app/app.py
   ```

   The application will be available at `http://localhost:5000`

## Usage

### Web Interface
- Upload UFDR, CSV, or JSON datasets
- View interactive network graphs
- Get AI-powered insights via the chatbot
- Generate PDF forensic reports

### API Integration
Import utilities for custom analysis:

```python
from utils.data_processing import preprocess_messages, detect_dataset_type
from utils.models_ai import detect_message_suspicion, build_network_graph
from utils.reporting import summarize_messages

# Process your data
df = preprocess_messages(df)
df = detect_message_suspicion(df)

# Build network visualization
graph = build_network_graph(messages_df=df)

# Generate summary
summary = summarize_messages(df)
```

## Key Functions

### Data Processing (`data_processing.py`)
- `preprocess_messages()` - Clean and normalize message data
- `preprocess_call_logs()` - Process call logs with time analysis
- `preprocess_contacts()` - Validate and normalize contacts
- `preprocess_gps()` - Process location data
- `rule_based_contact_flags()` - Flag suspicious contacts

### AI Models & Analysis (`models_ai.py`)
- `tfidf_embeddings()` - Generate text embeddings
- `run_message_clustering()` - Cluster messages using KMeans
- `run_gps_models()` - Analyze geographic patterns
- `detect_message_suspicion()` - Find suspicious messages
- `detect_call_suspicion()` - Identify suspicious calls
- `build_network_graph()` - Create network visualization data
- `calculate_overall_risk_score()` - Compute risk metrics
- `setup_model()` - Initialize Groq chatbot

### Reporting (`reporting.py`)
- `summarize_call_logs()` - Generate call statistics
- `summarize_messages()` - Generate message statistics
- `summarize_contacts()` - Generate contact statistics
- `summarize_gps()` - Generate location statistics
- `create_pdf()` - Generate PDF reports

## Requirements

- Python 3.8+
- Flask - Web framework
- Pandas - Data manipulation
- NumPy - Numerical computing
- scikit-learn - Machine learning
- ReportLab - PDF generation
- Groq - AI chatbot API

See `requirements.txt` for specific versions.

## Environment Variables

Create a `.env` file with:

```env
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=openai/gpt-oss-20b
MAX_DATASET_ROWS=2000
FLASK_ENV=development
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For issues, questions, or suggestions, please open an GitHub issue.

## Disclaimer

This tool is designed for forensic analysis purposes. Ensure you have proper authorization before analyzing device data.
