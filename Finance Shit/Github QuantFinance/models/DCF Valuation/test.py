# test.py
import pandas as pd
from models import Financials, DCFParameters, DCFResult

# Create test data
test_income_stmt = pd.DataFrame({'2023-12-31': [1000, 500]}, index=['Total Revenue', 'Net Income'])
test_balance_sheet = pd.DataFrame({'2023-12-31': [2000, 1000]}, index=['Total Assets', 'Total Liabilities'])
test_cash_flow = pd.DataFrame({'2023-12-31': [300, 150]}, index=['Operating Cash Flow', 'Capital Expenditure'])

# Test Financials
financials = Financials(
    ticker="TEST",
    income_stmt=test_income_stmt,
    balance_sheet=test_balance_sheet,
    cash_flow_stmt=test_cash_flow,
    period="annual"
)

# Test DCFParameters (with defaults)
params = DCFParameters()

# Test DCFResult
result = DCFResult(
    ticker="TEST",
    enterprise_value=1000000,
    equity_value=800000,
    wacc=0.08,
    historical_fcf=pd.Series([100, 110, 120]),
    projected_fcf=pd.Series([130, 140, 150]),
    terminal_value=2000000,
    fair_value=150.0,
    current_price=145.0,
    parameters=params
)

print("All models instantiated successfully!")