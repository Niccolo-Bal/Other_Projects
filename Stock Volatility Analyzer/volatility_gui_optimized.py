import tkinter as tk
from tkinter import ttk, messagebox
from tkinter import font
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import pickle
import os
from functools import lru_cache

class DataCache:
    def __init__(self, cache_dir="volatility_cache"):
        self.cache_dir = cache_dir
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

    def get_cache_path(self, ticker, start_date, end_date):
        return os.path.join(self.cache_dir, f"{ticker}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.pkl")

    def get_cached_data(self, ticker, start_date, end_date):
        cache_path = self.get_cache_path(ticker, start_date, end_date)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except:
                pass
        return None

    def cache_data(self, ticker, start_date, end_date, data):
        cache_path = self.get_cache_path(ticker, start_date, end_date)
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
        except:
            pass

class VolatilityAnalyzer:
    def __init__(self, root):
        self.root = root
        self.root.title("📈 Stock Volatility Analyzer (Optimized)")
        self.root.geometry("800x600")
        self.root.configure(bg='#f0f0f0')

        # Variables
        self.tickers = []
        self.results = []
        self.is_analyzing = False
        self.cache = DataCache()
        self.executor = None

        # Load tickers on startup
        self.load_tickers()

        # Create GUI
        self.create_widgets()

    def load_tickers(self):
        """Load tickers from CSV file"""
        try:
            import os
            script_dir = os.path.dirname(os.path.abspath(__file__))
            csv_path = os.path.join(script_dir, 'Top Tickers.csv')
            df = pd.read_csv(csv_path)
            all_tickers = df['Symbol'].tolist()

            # Filter out invalid tickers in batch
            valid_tickers = []
            for ticker in all_tickers:
                ticker = str(ticker).strip()
                if not ticker.startswith('$') and ticker.isalpha() and len(ticker) <= 5:
                    valid_tickers.append(ticker)

            self.tickers = valid_tickers
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
        title_label = tk.Label(self.root, text="Stock Volatility Analyzer (Optimized)",
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

        # Clear cache button
        self.clear_cache_btn = tk.Button(button_frame, text="Clear Cache",
                                       command=self.clear_cache, bg='#FF9800',
                                       fg='white', font=('Arial', 10, 'bold'),
                                       width=15, height=2)
        self.clear_cache_btn.pack(side='left', padx=10)

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

    def clear_cache(self):
        """Clear the data cache"""
        try:
            import shutil
            if os.path.exists(self.cache.cache_dir):
                shutil.rmtree(self.cache.cache_dir)
                os.makedirs(self.cache.cache_dir)
            messagebox.showinfo("Cache Cleared", "Data cache has been cleared.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to clear cache: {e}")

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
        if self.executor:
            self.executor.shutdown(wait=False)
        self.analyze_btn.config(state='normal')
        self.stop_btn.config(state='disabled')
        self.status_var.set("Analysis stopped")
        self.progress_var.set(0)

    def download_data_batch(self, tickers, start_date, end_date):
        """Download data for multiple tickers in batch"""
        try:
            # Use yfinance bulk download for better performance
            data = yf.download(tickers, start=start_date, end=end_date,
                             group_by='ticker', progress=False, threads=True)

            if len(tickers) == 1:
                # Single ticker case - yfinance returns different structure
                return {tickers[0]: data}

            # Multi-ticker case
            result = {}
            for ticker in tickers:
                if ticker in data.columns.levels[0]:
                    ticker_data = data[ticker].dropna()
                    if len(ticker_data) > 0:
                        result[ticker] = ticker_data
            return result
        except Exception as e:
            print(f"Batch download failed: {e}")
            return {}

    def process_ticker_volatility(self, ticker_data, volatility_period, volatility_type):
        """Process volatility for a single ticker's data"""
        ticker, data = ticker_data

        try:
            if data is None or len(data) < 2:
                return None

            volatility = self.calculate_period_volatility_from_data(
                data, volatility_period, volatility_type)

            if volatility is not None and not np.isnan(volatility):
                return {
                    'Ticker': ticker,
                    'Volatility': volatility
                }
            return None

        except Exception as e:
            print(f"Error processing {ticker}: {e}")
            return None

    def analyze_volatility(self, start_date, end_date, tickers_to_analyze, volatility_period, volatility_type):
        volatility_data = []
        total_tickers = len(tickers_to_analyze)

        self.status_var.set(f"Downloading data for {total_tickers} stocks...")

        # First, try to get data from cache or download in batches
        cached_data = {}
        uncached_tickers = []

        for ticker in tickers_to_analyze:
            if not self.is_analyzing:
                break

            cached = self.cache.get_cached_data(ticker, start_date, end_date)
            if cached is not None:
                cached_data[ticker] = cached
            else:
                uncached_tickers.append(ticker)

        # Download uncached data in batches of 20 for better performance
        batch_size = 20
        all_data = cached_data.copy()

        for i in range(0, len(uncached_tickers), batch_size):
            if not self.is_analyzing:
                break

            batch = uncached_tickers[i:i+batch_size]
            self.status_var.set(f"Downloading batch {i//batch_size + 1}/{(len(uncached_tickers)-1)//batch_size + 1}...")

            batch_data = self.download_data_batch(batch, start_date, end_date)

            # Cache the downloaded data
            for ticker, data in batch_data.items():
                all_data[ticker] = data
                self.cache.cache_data(ticker, start_date, end_date, data)

        if not self.is_analyzing:
            return

        # Process volatility calculations in parallel
        self.status_var.set("Calculating volatility...")

        # Prepare data for parallel processing
        ticker_data_pairs = [(ticker, data) for ticker, data in all_data.items()]

        # Use ThreadPoolExecutor for parallel processing
        max_workers = min(8, len(ticker_data_pairs))
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

        try:
            # Submit all tasks
            future_to_ticker = {
                self.executor.submit(self.process_ticker_volatility,
                                   (ticker, data), volatility_period, volatility_type): ticker
                for ticker, data in ticker_data_pairs
            }

            completed = 0
            for future in as_completed(future_to_ticker):
                if not self.is_analyzing:
                    break

                ticker = future_to_ticker[future]
                try:
                    result = future.result()
                    if result is not None:
                        volatility_data.append(result)
                except Exception as e:
                    print(f"Error processing {ticker}: {e}")

                completed += 1
                progress = (completed / len(ticker_data_pairs)) * 100
                self.progress_var.set(progress)
                self.status_var.set(f"Processing {ticker} ({completed}/{len(ticker_data_pairs)}) - {volatility_type} {volatility_period}")

        finally:
            self.executor.shutdown(wait=True)
            self.executor = None

        if self.is_analyzing and volatility_data:
            # Sort results and display
            df = pd.DataFrame(volatility_data)
            df = df.sort_values('Volatility', ascending=False)
            self.root.after(0, self.display_results, df, start_date, end_date, volatility_period, volatility_type)
        elif self.is_analyzing:
            self.root.after(0, self.display_no_results)

        # Reset UI
        self.root.after(0, self.reset_ui)

    def calculate_period_volatility_from_data(self, data, period, volatility_type):
        """Calculate volatility from pre-loaded data"""
        try:
            if len(data) < 2:
                return None

            if period == "Daily":
                data_copy = data.copy()
                data_copy['Returns'] = data_copy['Close'].pct_change()
                returns = data_copy['Returns'].dropna()
                scaling_factor = np.sqrt(252)

            elif period == "Weekly":
                weekly_data = data['Close'].resample('W').last()
                weekly_returns = weekly_data.pct_change().dropna()
                if len(weekly_returns) < 2:
                    return None
                returns = weekly_returns
                scaling_factor = np.sqrt(52)

            elif period == "Monthly":
                monthly_data = data['Close'].resample('M').last()
                monthly_returns = monthly_data.pct_change().dropna()
                if len(monthly_returns) < 2:
                    return None
                returns = monthly_returns
                scaling_factor = np.sqrt(12)

            elif period == "Quarterly":
                quarterly_data = data['Close'].resample('Q').last()
                quarterly_returns = quarterly_data.pct_change().dropna()
                if len(quarterly_returns) < 2:
                    return None
                returns = quarterly_returns
                scaling_factor = np.sqrt(4)

            if volatility_type == "Raw":
                volatility = returns.std() * scaling_factor

            elif volatility_type == "Growth-Adjusted":
                if len(returns) < 2:
                    return None

                cumulative_return = (1 + returns).prod()
                time_periods = len(returns)

                if period == "Daily":
                    periods_per_year = 252
                elif period == "Weekly":
                    periods_per_year = 52
                elif period == "Monthly":
                    periods_per_year = 12
                elif period == "Quarterly":
                    periods_per_year = 4

                years = time_periods / periods_per_year
                annualized_growth = (cumulative_return ** (1/years)) - 1

                if abs(annualized_growth) < 1e-10:
                    volatility = returns.std() * scaling_factor
                else:
                    raw_volatility = returns.std() * scaling_factor
                    volatility = abs(raw_volatility / annualized_growth)

            return volatility

        except Exception as e:
            print(f"Error calculating volatility: {e}")
            return None

    def calculate_period_volatility(self, ticker, start_date, end_date, period, volatility_type):
        """Legacy method for backward compatibility - uses caching"""
        try:
            # Check cache first
            cached_data = self.cache.get_cached_data(ticker, start_date, end_date)
            if cached_data is not None:
                data = cached_data
            else:
                # Download and cache
                stock = yf.Ticker(ticker)
                data = stock.history(start=start_date, end=end_date)
                self.cache.cache_data(ticker, start_date, end_date, data)

            return self.calculate_period_volatility_from_data(data, period, volatility_type)

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