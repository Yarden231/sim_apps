# styles.py
def get_custom_css():
    return """
    <style>
    /* RTL Support and General Layout */
    .rtl-content {
        direction: rtl !important;
        text-align: right !important;
    }
    
    .main {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        padding: 1rem;
    }
    
    /* Header Styling */
    .custom-header {
        background: linear-gradient(135deg, #0396FF, #ABDCFF);
        color: white !important;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
    }
    
    /* Card Styling */
    .custom-card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1.5rem;
    }
    
    /* Navigation Styling */
    .custom-nav {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1.5rem;
    }
    
    /* Image Container */
    .img-container {
        text-align: center;
        margin: 1rem 0;
    }
    
    /* List Styling */
    .custom-list {
        padding-right: 1.5rem;
        margin: 1rem 0;
    }
    
    .custom-list li {
        margin-bottom: 0.5rem;
    }
    
    /* Section Headers */
    .section-header {
        color: #2c3e50;
        border-bottom: 2px solid #0396FF;
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
    }
    
    /* Input Fields */
    .stTextInput > div > div > input {
        direction: rtl;
    }
    
    /* Buttons */
    .stButton > button {
        background-color: #0396FF;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        transition: background-color 0.3s;
    }
    
    .stButton > button:hover {
        background-color: #0377cc;
    }
    
    /* Metrics and KPIs */
    .metric-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        text-align: center;
    }
    
    /* Info boxes */
    .info-box {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-right: 4px solid #0396FF;
    }
    
    /* Tables */
    .styled-table {
        width: 100%;
        border-collapse: collapse;
        margin: 1rem 0;
        direction: rtl;
    }
    
    .styled-table th {
        background-color: #0396FF;
        color: white;
        padding: 0.75rem;
        text-align: right;
    }
    
    .styled-table td {
        padding: 0.75rem;
        border-bottom: 1px solid #e0e0e0;
    }
    
    /* Charts container */
    .chart-container {
        background-color: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
    }
    
    /* Responsive adjustments */
    @media (max-width: 768px) {
        .custom-header {
            padding: 1rem;
        }
        
        .custom-card {
            padding: 1rem;
        }
    }
    </style>
    """
