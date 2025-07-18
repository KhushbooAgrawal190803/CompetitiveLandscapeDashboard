# Sunglasses Market Analysis Dashboard

A comprehensive Streamlit dashboard for analyzing sunglasses data across brands, retailers, SKUs, silhouettes, prices, and lens technologies.

## Features

- **Market Analysis**: Overview of market-wide trends and patterns
- **Retailer Drill-Down**: Detailed analysis by individual retailer
- **Brand Drill-Down**: Brand-specific insights and performance
- **Internal Analysis**: Internal data analysis with pricing insights
- **Interactive Visualizations**: Charts, graphs, and insights throughout

## Data Files Required

- `internDashbaordData.xlsx` - Main market data with multiple retailer sheets
- `InternalData.xlsx` - Internal analysis data with Shape and Pricing sheets

## Local Development

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the dashboard:
```bash
streamlit run intern_dashboard.py
```

3. Open your browser to `http://localhost:8501`

## Deployment Options

### Streamlit Cloud (Recommended)
1. Create a GitHub repository
2. Upload your files (including Excel data files)
3. Go to [share.streamlit.io](https://share.streamlit.io)
4. Connect your GitHub repository
5. Deploy automatically

### Heroku
1. Create a `Procfile` with: `web: streamlit run intern_dashboard.py --server.port=$PORT --server.address=0.0.0.0`
2. Deploy using Heroku CLI or GitHub integration

### Railway
1. Connect your GitHub repository
2. Railway will automatically detect and deploy your Streamlit app

## File Structure
```
Dashboard/
├── intern_dashboard.py      # Main dashboard application
├── requirements.txt         # Python dependencies
├── README.md              # This file
├── internDashbaordData.xlsx # Market data
└── InternalData.xlsx       # Internal analysis data
```

## Author
Khushboo Agrawal (Summer 2025) 