# Python version of the Cython file (without optimizations)
import tkinter as tk
from tkinter import ttk, messagebox
from tkinter import font
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import threading
import math

class FastMath:
    """Mathematical functions for volatility calculations (Python version)"""

    def calculate_volatility(self, returns, scaling_factor):
        """Calculate volatility using standard Python/NumPy"""
        if len(returns) < 2:
            return 0.0

        mean = np.mean(returns)
        variance = np.var(returns, ddof=1)
        return math.sqrt(variance) * scaling_factor

    def calculate_growth_adjusted_volatility(self, returns, scaling_factor, period):
        """Calculate growth-adjusted volatility using standard Python/NumPy"""
        if len(returns) < 2:
            return 0.0

        # Calculate cumulative return
        cumulative_return = np.prod(1.0 + returns)

        # Set periods per year based on period type
        if period == "Daily":
            periods_per_year = 252.0
        elif period == "Weekly":
            periods_per_year = 52.0
        elif period == "Monthly":
            periods_per_year = 12.0
        elif period == "Quarterly":
            periods_per_year = 4.0
        else:
            periods_per_year = 252.0  # Default to daily

        years = len(returns) / periods_per_year
        annualized_growth = (cumulative_return ** (1.0/years)) - 1.0

        if abs(annualized_growth) < 1e-10:
            # If annualized growth is essentially zero, use raw volatility
            return self.calculate_volatility(returns, scaling_factor)
        else:
            # Volatility to growth ratio (annualized)
            raw_volatility = self.calculate_volatility(returns, scaling_factor)
            return abs(raw_volatility / annualized_growth)

class VolatilityAnalyzer:
    def __init__(self, root):
        self.root = root
        self.root.title("📈 Stock Volatility Analyzer")
        self.root.geometry("800x600")
        self.root.configure(bg='#f0f0f0')

        # Variables
        self.tickers = []
        self.results = []
        self.is_analyzing = False

        # Initialize fast math calculator
        self.fast_math = FastMath()

        # Load tickers on startup
        self.load_tickers()

        # Create GUI
        self.create_widgets()

    def load_tickers(self):
        """Load tickers from CSV file"""
        try:
            df = pd.read_csv('Top Tickers.csv')
            all_tickers = df['Symbol'].tolist()  # Load ALL tickers from CSV

            # Filter out invalid tickers
            self.tickers = []
            for ticker in all_tickers:
                ticker = str(ticker).strip()
                if not ticker.startswith('$') and ticker.isalpha() and len(ticker) <= 5:
                    self.tickers.append(ticker)

            print(f"Loaded {len(self.tickers)} valid tickers")
        except Exception as e:
            print(f"Error loading tickers: {e}")
            # Fallback tickers
            self.tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA',
                           'BRK-B', 'JNJ', 'V', 'WMT', 'JPM', 'MA', 'PG', 'UNH',
                           'DIS', 'HD', 'PYPL', 'BAC', 'NFLX', 'ADBE', 'CMCSA']

    def create_widgets(self):
        # Main title
        title_font = font.Font(size=16, weight='bold')
        title_label = tk.Label(self.root, text="Stock Volatility Analyzer",
                              font=title_font, bg='#f0f0f0', fg='#333')
        title_label.pack(pady=10)

        # Date selection frame
        date_frame = tk.Frame(self.root, bg='#f0f0f0')
        date_frame.pack(pady=10, padx=20, fill='x')

        # Start date
        tk.Label(date_frame, text="Start Date:", bg='#f0f0f0').grid(row=0, column=0, padx=5, sticky='w')
        self.start_year = tk.StringVar(value=str(datetime.now().year - 1))
        self.start_month = tk.StringVar(value=str(datetime.now().month))
        self.start_day = tk.StringVar(value=str(datetime.now().day))

        start_frame = tk.Frame(date_frame, bg='#f0f0f0')
        start_frame.grid(row=0, column=1, padx=5, sticky='w')

        ttk.Combobox(start_frame, textvariable=self.start_year, values=list(range(2010, 2026)),
                    width=6, state='readonly').pack(side='left', padx=2)
        ttk.Combobox(start_frame, textvariable=self.start_month, values=list(range(1, 13)),
                    width=4, state='readonly').pack(side='left', padx=2)
        ttk.Combobox(start_frame, textvariable=self.start_day, values=list(range(1, 32)),
                    width=4, state='readonly').pack(side='left', padx=2)

        # End date
        tk.Label(date_frame, text="End Date:", bg='#f0f0f0').grid(row=1, column=0, padx=5, sticky='w')
        self.end_year = tk.StringVar(value=str(datetime.now().year))
        self.end_month = tk.StringVar(value=str(datetime.now().month))
        self.end_day = tk.StringVar(value=str(datetime.now().day))

        end_frame = tk.Frame(date_frame, bg='#f0f0f0')
        end_frame.grid(row=1, column=1, padx=5, sticky='w')

        ttk.Combobox(end_frame, textvariable=self.end_year, values=list(range(2010, 2026)),
                    width=6, state='readonly').pack(side='left', padx=2)
        ttk.Combobox(end_frame, textvariable=self.end_month, values=list(range(1, 13)),
                    width=4, state='readonly').pack(side='left', padx=2)
        ttk.Combobox(end_frame, textvariable=self.end_day, values=list(range(1, 32)),
                    width=4, state='readonly').pack(side='left', padx=2)

        # Stock selection mode
        tk.Label(date_frame, text="Stock Selection:", bg='#f0f0f0').grid(row=2, column=0, padx=5, sticky='w')
        self.stock_mode = tk.StringVar(value="Top Stocks")
        mode_combo = ttk.Combobox(date_frame, textvariable=self.stock_mode,
                                 values=['Top Stocks', 'Specific Stocks'],
                                 width=15, state='readonly')
        mode_combo.grid(row=2, column=1, padx=5, sticky='w')
        mode_combo.bind('<<ComboboxSelected>>', self.on_stock_mode_change)

        # Dynamic controls for Top N vs Specific stocks
        self.top_n = tk.StringVar(value="10")

        # Top N controls
        self.top_n_label = tk.Label(date_frame, text="Top N stocks:", bg='#f0f0f0')
        self.top_n_entry = tk.Entry(date_frame, textvariable=self.top_n, width=40)
        self.top_n_help = tk.Label(date_frame, text="(e.g. 10, 25, 50, 100)", bg='#f0f0f0', fg='#666', font=('Arial', 8))

        # Specific stocks controls
        self.specific_label = tk.Label(date_frame, text="Stock Tickers:", bg='#f0f0f0')
        self.specific_entry = tk.Entry(date_frame, width=40)
        self.specific_help = tk.Label(date_frame, text="(comma-separated, e.g., AAPL,MSFT,GOOGL)", bg='#f0f0f0', fg='#666', font=('Arial', 8))

        # Volatility period selection
        tk.Label(date_frame, text="Volatility Period:", bg='#f0f0f0').grid(row=4, column=0, padx=5, sticky='w')
        self.volatility_period = tk.StringVar(value="Daily")
        period_combo = ttk.Combobox(date_frame, textvariable=self.volatility_period,
                                   values=['Daily', 'Weekly', 'Monthly', 'Quarterly'],
                                   width=10, state='readonly')
        period_combo.grid(row=4, column=1, padx=5, sticky='w')

        # Volatility type selection
        tk.Label(date_frame, text="Volatility Type:", bg='#f0f0f0').grid(row=5, column=0, padx=5, sticky='w')
        self.volatility_type = tk.StringVar(value="Raw")
        type_combo = ttk.Combobox(date_frame, textvariable=self.volatility_type,
                                 values=['Raw', 'Growth-Adjusted'],
                                 width=15, state='readonly')
        type_combo.grid(row=5, column=1, padx=5, sticky='w')

        # Explanation label
        tk.Label(date_frame, text="(Growth-Adjusted: volatility relative to average growth)",
                bg='#f0f0f0', fg='#666', font=('Arial', 8)).grid(row=5, column=2, padx=5, sticky='w')

        # Initially show top N controls
        self.show_top_n_controls()

        # Control buttons
        button_frame = tk.Frame(self.root, bg='#f0f0f0')
        button_frame.pack(pady=15)

        self.analyze_btn = tk.Button(button_frame, text="Analyze Volatility",
                                   command=self.start_analysis, bg='#4CAF50',
                                   fg='white', font=('Arial', 10, 'bold'),
                                   width=15, height=2)
        self.analyze_btn.pack(side='left', padx=10)

        self.stop_btn = tk.Button(button_frame, text="Stop Analysis",
                                command=self.stop_analysis, bg='#f44336',
                                fg='white', font=('Arial', 10, 'bold'),
                                width=15, height=2, state='disabled')
        self.stop_btn.pack(side='left', padx=10)

        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(self.root, variable=self.progress_var,
                                          maximum=100, length=400)
        self.progress_bar.pack(pady=10)

        # Status label
        self.status_var = tk.StringVar(value="Ready to analyze")
        self.status_label = tk.Label(self.root, textvariable=self.status_var,
                                   bg='#f0f0f0', fg='#666')
        self.status_label.pack(pady=5)

        # Results frame
        results_frame = tk.Frame(self.root, bg='#f0f0f0')
        results_frame.pack(pady=10, padx=20, fill='both', expand=True)

        tk.Label(results_frame, text="Results:", font=('Arial', 12, 'bold'),
                bg='#f0f0f0').pack(anchor='w')

        # Results listbox with scrollbar
        listbox_frame = tk.Frame(results_frame)
        listbox_frame.pack(fill='both', expand=True)

        self.results_listbox = tk.Listbox(listbox_frame, font=('Courier', 10))
        scrollbar = tk.Scrollbar(listbox_frame, orient='vertical')

        self.results_listbox.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.results_listbox.yview)

        self.results_listbox.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')

    def on_stock_mode_change(self, event=None):
        """Handle switching between Top Stocks and Specific Stocks modes"""
        if self.stock_mode.get() == "Top Stocks":
            self.show_top_n_controls()
        else:
            self.show_specific_controls()

    def show_top_n_controls(self):
        """Show Top N stocks input controls"""
        # Hide specific controls
        self.specific_label.grid_remove()
        self.specific_entry.grid_remove()
        self.specific_help.grid_remove()

        # Show top N controls
        self.top_n_label.grid(row=3, column=0, padx=5, sticky='w')
        self.top_n_entry.grid(row=3, column=1, padx=5, sticky='w')
        self.top_n_help.grid(row=3, column=2, padx=5, sticky='w')

    def show_specific_controls(self):
        """Show specific stocks input controls"""
        # Hide top N controls
        self.top_n_label.grid_remove()
        self.top_n_entry.grid_remove()
        self.top_n_help.grid_remove()

        # Show specific controls
        self.specific_label.grid(row=3, column=0, padx=5, sticky='w')
        self.specific_entry.grid(row=3, column=1, padx=5, sticky='w')
        self.specific_help.grid(row=3, column=2, padx=5, sticky='w')

    def get_tickers_to_analyze(self):
        """Get list of tickers based on selected mode"""
        if self.stock_mode.get() == "Top Stocks":
            try:
                top_n = int(self.top_n.get())
                return self.tickers[:top_n]
            except ValueError:
                raise ValueError("Please enter a valid number for Top N stocks")
        else:
            # Specific stocks mode
            tickers_text = self.specific_entry.get().strip()
            if not tickers_text:
                raise ValueError("Please enter stock tickers (comma-separated)")

            # Parse comma-separated tickers
            tickers = [ticker.strip().upper() for ticker in tickers_text.split(',')]
            tickers = [ticker for ticker in tickers if ticker]  # Remove empty strings

            if not tickers:
                raise ValueError("Please enter valid stock tickers")

            return tickers

    def start_analysis(self):
        if self.is_analyzing:
            return

        try:
            # Validate dates
            start_date = datetime(int(self.start_year.get()),
                                int(self.start_month.get()),
                                int(self.start_day.get()))
            end_date = datetime(int(self.end_year.get()),
                              int(self.end_month.get()),
                              int(self.end_day.get()))

            if start_date >= end_date:
                messagebox.showerror("Error", "Start date must be before end date")
                return

            if end_date > datetime.now():
                messagebox.showerror("Error", "End date cannot be in the future")
                return

            # Get tickers based on selected mode
            tickers_to_analyze = self.get_tickers_to_analyze()
            volatility_period = self.volatility_period.get()
            volatility_type = self.volatility_type.get()

            # Clear previous results
            self.results_listbox.delete(0, tk.END)

            # Start analysis in separate thread
            self.is_analyzing = True
            self.analyze_btn.config(state='disabled')
            self.stop_btn.config(state='normal')

            thread = threading.Thread(target=self.analyze_volatility,
                                    args=(start_date, end_date, tickers_to_analyze, volatility_period, volatility_type))
            thread.daemon = True
            thread.start()

        except ValueError as e:
            messagebox.showerror("Error", str(e))

    def stop_analysis(self):
        self.is_analyzing = False
        self.analyze_btn.config(state='normal')
        self.stop_btn.config(state='disabled')
        self.status_var.set("Analysis stopped")
        self.progress_var.set(0)

    def analyze_volatility(self, start_date, end_date, tickers_to_analyze, volatility_period, volatility_type):
        volatility_data = []

        total_tickers = len(tickers_to_analyze)

        self.status_var.set(f"Analyzing {total_tickers} stocks ({volatility_type} {volatility_period} volatility)...")

        for i in range(total_tickers):
            if not self.is_analyzing:
                break

            ticker = tickers_to_analyze[i]
            try:
                # Update progress
                progress = (i / total_tickers) * 100.0
                self.progress_var.set(progress)
                self.status_var.set(f"Processing {ticker} ({i+1}/{total_tickers}) - {volatility_type} {volatility_period}")

                # Calculate volatility based on selected period and type
                volatility = self.calculate_period_volatility(ticker, start_date, end_date, volatility_period, volatility_type)

                if volatility is not None and not np.isnan(volatility):
                    volatility_data.append({
                        'Ticker': ticker,
                        'Volatility': volatility
                    })

            except Exception as e:
                print(f"Error processing {ticker}: {e}")
                continue

        if self.is_analyzing:
            # Sort results and show all analyzed stocks
            df = pd.DataFrame(volatility_data)
            if not df.empty:
                df = df.sort_values('Volatility', ascending=False)

                # Display all results since we already processed only the requested number
                self.root.after(0, self.display_results, df, start_date, end_date, volatility_period, volatility_type)
            else:
                self.root.after(0, self.display_no_results)

        # Reset UI
        self.root.after(0, self.reset_ui)

    def calculate_period_volatility(self, ticker, start_date, end_date, period, volatility_type):
        """Calculate volatility for different time periods and types"""
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(start=start_date, end=end_date)

            if len(data) < 2:
                return None

            if period == "Daily":
                data['Returns'] = data['Close'].pct_change()
                returns = data['Returns'].dropna()
                scaling_factor = math.sqrt(252.0)

            elif period == "Weekly":
                # Resample to weekly and calculate weekly volatility
                weekly_data = data['Close'].resample('W').last()
                weekly_returns = weekly_data.pct_change().dropna()
                if len(weekly_returns) < 2:
                    return None
                returns = weekly_returns
                scaling_factor = math.sqrt(52.0)  # Annualized weekly volatility

            elif period == "Monthly":
                # Resample to monthly and calculate monthly volatility
                monthly_data = data['Close'].resample('M').last()
                monthly_returns = monthly_data.pct_change().dropna()
                if len(monthly_returns) < 2:
                    return None
                returns = monthly_returns
                scaling_factor = math.sqrt(12.0)  # Annualized monthly volatility

            elif period == "Quarterly":
                # Resample to quarterly and calculate quarterly volatility
                quarterly_data = data['Close'].resample('Q').last()
                quarterly_returns = quarterly_data.pct_change().dropna()
                if len(quarterly_returns) < 2:
                    return None
                returns = quarterly_returns
                scaling_factor = math.sqrt(4.0)  # Annualized quarterly volatility

            # Convert to numpy array for computation
            returns_array = np.asarray(returns, dtype=np.float64)

            if volatility_type == "Raw":
                # Standard volatility calculation
                volatility = self.fast_math.calculate_volatility(returns_array, scaling_factor)

            elif volatility_type == "Growth-Adjusted":
                # Calculate volatility relative to annualized compound growth
                if len(returns) < 2:
                    return None

                # Growth-adjusted calculation
                volatility = self.fast_math.calculate_growth_adjusted_volatility(returns_array, scaling_factor, period)

            return volatility

        except Exception as e:
            print(f"Error calculating {volatility_type} {period} volatility for {ticker}: {e}")
            return None

    def display_results(self, results, start_date, end_date, volatility_period, volatility_type):
        self.results_listbox.delete(0, tk.END)

        # Header
        header = f"Top {len(results)} Most Volatile Stocks ({volatility_type} {volatility_period} Volatility)"
        period = f"Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"

        self.results_listbox.insert(tk.END, header)
        self.results_listbox.insert(tk.END, period)
        self.results_listbox.insert(tk.END, "=" * 70)

        # Results with appropriate units
        for i, (_, row) in enumerate(results.iterrows(), 1):
            if volatility_type == "Growth-Adjusted":
                line = f"{i:2d}. {row['Ticker']:6s} - {row['Volatility']:.3f} (vol/growth ratio)"
            else:
                line = f"{i:2d}. {row['Ticker']:6s} - {row['Volatility']:.4f} ({row['Volatility']*100:.2f}%)"
            self.results_listbox.insert(tk.END, line)

        self.status_var.set(f"Analysis complete. Found {len(results)} results ({volatility_type} {volatility_period}).")

    def display_no_results(self):
        self.results_listbox.delete(0, tk.END)
        self.results_listbox.insert(tk.END, "No valid results found.")
        self.status_var.set("Analysis complete. No valid data found.")

    def reset_ui(self):
        self.is_analyzing = False
        self.analyze_btn.config(state='normal')
        self.stop_btn.config(state='disabled')
        self.progress_var.set(100)

def main():
    root = tk.Tk()
    app = VolatilityAnalyzer(root)
    root.mainloop()

if __name__ == "__main__":
    main()