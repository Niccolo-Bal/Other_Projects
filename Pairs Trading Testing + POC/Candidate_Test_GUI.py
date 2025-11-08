import tkinter as tk
from tkinter import ttk, messagebox
import yfinance as yf
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, coint
from sklearn.linear_model import LinearRegression
import threading
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

# Import profit calc helpers (expects Profit_Calc.py in same directory)
try:
    from Profit_Calc import fit_hedge_ratio, compute_spread, simulate_pairs_trade
except Exception:
    # lazy fallback if import fails; functions will be required when running simulation
    fit_hedge_ratio = None
    compute_spread = None
    simulate_pairs_trade = None


class CointegrationTestGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Cointegration Calculator 8==>")
        self.root.geometry("1400x800")

        style = ttk.Style()
        style.configure('Header.TLabel', font=('Times New Roman', 10, 'bold'))

        self.create_widgets()

        # placeholders for the last downloaded/processed data so profit simulation can reuse
        self.last_combined = None
        self.last_hedge_ratio = None
        self.last_intercept = None

    def create_widgets(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=2)

        title_label = ttk.Label(main_frame, text="Stock Cointegration Calculator",
                                style='Header.TLabel', font=('Comic Sans MS', 14, 'bold'))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))

        left_frame = ttk.Frame(main_frame)
        left_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))

        right_frame = ttk.Frame(main_frame)
        right_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))

        main_frame.rowconfigure(1, weight=1)

        ttk.Label(left_frame, text="Ticker 1:", style='Header.TLabel').grid(row=0, column=0, sticky=tk.W, pady=5)
        self.ticker1_var = tk.StringVar(value="PEP")
        ticker1_entry = ttk.Entry(left_frame, textvariable=self.ticker1_var, width=15)
        ticker1_entry.grid(row=0, column=1, sticky=tk.W, pady=5)

        ttk.Label(left_frame, text="Ticker 2:", style='Header.TLabel').grid(row=1, column=0, sticky=tk.W, pady=5)
        self.ticker2_var = tk.StringVar(value="KO")
        ticker2_entry = ttk.Entry(left_frame, textvariable=self.ticker2_var, width=15)
        ticker2_entry.grid(row=1, column=1, sticky=tk.W, pady=5)

        ttk.Label(left_frame, text="Period:", style='Header.TLabel').grid(row=2, column=0, sticky=tk.W, pady=5)
        period_frame = ttk.Frame(left_frame)
        period_frame.grid(row=2, column=1, sticky=tk.W, pady=5)

        self.period_number_var = tk.StringVar(value="18")
        period_number_entry = ttk.Entry(period_frame, textvariable=self.period_number_var, width=8)
        period_number_entry.grid(row=0, column=0, padx=(0, 5))

        self.period_unit_var = tk.StringVar(value="mo")
        period_unit_combo = ttk.Combobox(period_frame, textvariable=self.period_unit_var,
                                         values=["mo", "y"], width=5, state='readonly')
        period_unit_combo.grid(row=0, column=1)

        ttk.Label(left_frame, text="Lag:", style='Header.TLabel').grid(row=3, column=0, sticky=tk.W, pady=5)

        lag_frame = ttk.Frame(left_frame)
        lag_frame.grid(row=3, column=1, columnspan=2, sticky=tk.W, pady=5)

        self.lag_method_var = tk.StringVar(value="AIC")
        ttk.Radiobutton(lag_frame, text="Automatic",
                        variable=self.lag_method_var, value="AIC").grid(row=0, column=0, sticky=tk.W)
        ttk.Radiobutton(lag_frame, text="Conservative",
                        variable=self.lag_method_var, value="BIC").grid(row=1, column=0, sticky=tk.W)
        ttk.Radiobutton(lag_frame, text="Manual (days):",
                        variable=self.lag_method_var, value="MANUAL").grid(row=2, column=0, sticky=tk.W)

        self.manual_lag_var = tk.StringVar(value="1")
        self.manual_lag_entry = ttk.Entry(lag_frame, textvariable=self.manual_lag_var, width=8)
        self.manual_lag_entry.grid(row=2, column=1, sticky=tk.W, padx=5)

        ttk.Label(left_frame, text="Reversion Precision (σ):", style='Header.TLabel').grid(row=4, column=0, sticky=tk.W, pady=5)
        reversion_frame = ttk.Frame(left_frame)
        reversion_frame.grid(row=4, column=1, sticky=tk.W, pady=5)

        ttk.Label(reversion_frame, text="Max:").grid(row=0, column=0, sticky=tk.W)
        self.reversion_max_var = tk.StringVar(value="1.96")
        reversion_max_entry = ttk.Entry(reversion_frame, textvariable=self.reversion_max_var, width=6)
        reversion_max_entry.grid(row=0, column=1, padx=(5, 10))

        ttk.Label(reversion_frame, text="Min:").grid(row=0, column=2, sticky=tk.W)
        self.reversion_min_var = tk.StringVar(value="0.5")
        reversion_min_entry = ttk.Entry(reversion_frame, textvariable=self.reversion_min_var, width=6)
        reversion_min_entry.grid(row=0, column=3, padx=5)

        # Date inputs for optional train/trade selection (used by profit simulation)
        ttk.Label(left_frame, text="Train start (YYYY-MM-DD):", style='Header.TLabel').grid(row=5, column=0, sticky=tk.W, pady=3)
        self.train_start_var = tk.StringVar()
        ttk.Entry(left_frame, textvariable=self.train_start_var, width=15).grid(row=5, column=1, sticky=tk.W)

        ttk.Label(left_frame, text="Train end (YYYY-MM-DD):", style='Header.TLabel').grid(row=6, column=0, sticky=tk.W, pady=3)
        self.train_end_var = tk.StringVar()
        ttk.Entry(left_frame, textvariable=self.train_end_var, width=15).grid(row=6, column=1, sticky=tk.W)

        ttk.Label(left_frame, text="Trade start (YYYY-MM-DD):", style='Header.TLabel').grid(row=7, column=0, sticky=tk.W, pady=3)
        self.trade_start_var = tk.StringVar()
        ttk.Entry(left_frame, textvariable=self.trade_start_var, width=15).grid(row=7, column=1, sticky=tk.W)

        ttk.Label(left_frame, text="Trade end (YYYY-MM-DD):", style='Header.TLabel').grid(row=8, column=0, sticky=tk.W, pady=3)
        self.trade_end_var = tk.StringVar()
        ttk.Entry(left_frame, textvariable=self.trade_end_var, width=15).grid(row=8, column=1, sticky=tk.W)

        # Option: normalize prices for plotting (percent of start)
        self.normalize_prices_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(left_frame, text="Normalize prices (%)", variable=self.normalize_prices_var).grid(row=9, column=0, columnspan=2, sticky=tk.W, pady=6)

        self.run_button = ttk.Button(left_frame, text="Run Cointegration Test",
                                      command=self.run_test, style='Accent.TButton')
        self.run_button.grid(row=10, column=0, columnspan=2, pady=8)

        # Button to run profit/backtest simulation using Profit_Calc logic
        self.run_profit_button = ttk.Button(left_frame, text="Run Profit Simulation",
                                           command=self.run_profit_simulation)
        self.run_profit_button.grid(row=10, column=2, padx=(10,0), pady=8)

        self.progress = ttk.Progressbar(left_frame, mode='indeterminate', length=300)
        self.progress.grid(row=11, column=0, columnspan=2, pady=5)

        ttk.Label(left_frame, text="Results:", style='Header.TLabel').grid(row=12, column=0, sticky=tk.W, pady=(10, 5))

        self.results_text = tk.Text(left_frame, width=55, height=22,
                                     wrap=tk.WORD, font=('Courier', 12))
        self.results_text.grid(row=13, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)

        left_frame.rowconfigure(13, weight=1)

        self.figure = Figure(figsize=(8, 6))
        self.canvas = FigureCanvasTkAgg(self.figure, master=right_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def log(self, message):
        """Thread-safe logging to the text widget"""
        self.results_text.insert(tk.END, message + "\n")
        self.results_text.see(tk.END)
        self.root.update_idletasks()

    def clear_results(self):
        """Clear the results text area"""
        self.results_text.delete(1.0, tk.END)

    def plot_residuals(self, residuals, ticker1, ticker2, prices1, prices2, prices_are_percent=True):
        """Plot residuals and individual asset prices
        prices_are_percent: if True, y-labels show percent; otherwise raw price units
        """
        self.figure.clear()

        gs = self.figure.add_gridspec(2, 2, height_ratios=[2, 1], hspace=0.3, wspace=0.3)

        ax1 = self.figure.add_subplot(gs[0, :])
        mean = residuals.mean()
        std = residuals.std()

        ax1.plot(residuals.index, residuals.values, linewidth=1, label='Spread (ε, %)', color='blue')
        ax1.axhline(y=mean, color='red', linestyle='--', linewidth=1, label='Mean')
        ax1.axhline(y=mean + std, color='orange', linestyle=':', linewidth=1, label='±1σ')
        ax1.axhline(y=mean - std, color='orange', linestyle=':', linewidth=1)
        ax1.axhline(y=mean + 2*std, color='green', linestyle=':', linewidth=0.8, label='±2σ')
        ax1.axhline(y=mean - 2*std, color='green', linestyle=':', linewidth=0.8)

        ax1.set_ylabel('Residual (%)', fontsize=9)
        ax1.set_title(f'Spread (percent dev): {ticker1} - β*{ticker2}', fontsize=10)
        ax1.legend(loc='best', fontsize=7)
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='both', labelsize=8)

        ax2 = self.figure.add_subplot(gs[1, 0])
        ax2.plot(prices1.index, prices1.values, linewidth=1, color='darkblue')
        ax2.set_xlabel('Date', fontsize=9)
        ax2.set_ylabel(f'{ticker1} Price (% of start)' if prices_are_percent else f'{ticker1} Price', fontsize=9)
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='both', labelsize=8)
        ax2.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax2.tick_params(axis='x', rotation=45)

        ax3 = self.figure.add_subplot(gs[1, 1])
        ax3.plot(prices2.index, prices2.values, linewidth=1, color='darkgreen')
        ax3.set_xlabel('Date', fontsize=9)
        ax3.set_ylabel(f'{ticker2} Price (% of start)' if prices_are_percent else f'{ticker2} Price', fontsize=9)
        ax3.grid(True, alpha=0.3)
        ax3.tick_params(axis='both', labelsize=8)
        ax3.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax3.tick_params(axis='x', rotation=45)

        self.figure.tight_layout()
        self.canvas.draw()

    def run_test(self):
        """Run the cointegration test in a separate thread"""
        ticker1 = self.ticker1_var.get().strip().upper()
        ticker2 = self.ticker2_var.get().strip().upper()

        if not ticker1 or not ticker2:
            messagebox.showerror("Error", "Please enter both ticker symbols")
            return

        if ticker1 == ticker2:
            messagebox.showerror("Error", "Enter two different ticker symbols")
            return

        try:
            period_number = int(self.period_number_var.get())
            if period_number <= 0:
                messagebox.showerror("Error", "Period must be a positive integer")
                return
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid integer for period")
            return

        if self.lag_method_var.get() == "MANUAL":
            try:
                lag_value = int(self.manual_lag_var.get())
                if lag_value < 0:
                    messagebox.showerror("Error", "Lag must be a non-negative integer")
                    return
            except ValueError:
                messagebox.showerror("Error", "Please enter a valid integer for manual lag")
                return

        try:
            reversion_max = float(self.reversion_max_var.get())
            reversion_min = float(self.reversion_min_var.get())
            if reversion_max <= 0 or reversion_min < 0:
                messagebox.showerror("Error", "Reversion precision values must be positive")
                return
            if reversion_min >= reversion_max:
                messagebox.showerror("Error", "Max threshold must be greater than min threshold")
                return
        except ValueError:
            messagebox.showerror("Error", "Please enter valid numbers for reversion precision")
            return

        self.run_button.config(state='disabled')
        self.progress.start()
        self.clear_results()

        thread = threading.Thread(target=self.execute_test)
        thread.daemon = True
        thread.start()

    def execute_test(self):
        """Execute the cointegration test"""
        try:
            ticker1 = self.ticker1_var.get().strip().upper()
            ticker2 = self.ticker2_var.get().strip().upper()
            period = self.period_number_var.get() + self.period_unit_var.get()

            lag_method = self.lag_method_var.get()
            if lag_method == "AIC":
                lag_param = 'AIC'
            elif lag_method == "BIC":
                lag_param = 'BIC'
            else:
                lag_param = int(self.manual_lag_var.get())

            reversion_max = float(self.reversion_max_var.get())
            reversion_min = float(self.reversion_min_var.get())

            self.test_cointegration(ticker1, ticker2, period, lag_param, reversion_max, reversion_min)

        except Exception as e:
            self.log(f"\nERROR: {str(e)}")
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
        finally:
            self.progress.stop()
            self.run_button.config(state='normal')

    def adf_test(self, series, name, lag_param): # ADF test (of series, not residuals)

        if isinstance(lag_param, str):
            result = adfuller(series, autolag=lag_param)
        else:
            result = adfuller(series, maxlag=lag_param, autolag=None)

        p_value = result[1]

        return p_value

    def reg_and_residuals(self, y, x, ticker1_name, ticker2_name): # Calculate ls-regression and residuals

        y_reshaped = y.values.reshape(-1, 1)
        x_reshaped = x.values.reshape(-1, 1)
        model = LinearRegression()
        model.fit(x_reshaped, y_reshaped)

        beta1 = model.coef_[0][0]
        beta0 = model.intercept_[0]

        residuals = y - (beta0 + beta1 * x)

        return residuals, beta1, beta0

    def engle_granger_test(self, y, x, ticker1_name, ticker2_name):
        """Perform Engle-Granger cointegration test"""
        eg_stat, p_value, crit_values = coint(y, x)

        return p_value

    def count_reversions(self, residuals, max_threshold, min_threshold):
        """Count mean reversions from ±max_threshold σ back to ±min_threshold σ"""
        mean = residuals.mean()
        # use population std (ddof=0) to match Profit_Calc's default
        std = residuals.std(ddof=0)

        z_scores = (residuals - mean) / std

        reversions = 0
        in_extreme = False
        extreme_direction = None

        for z in z_scores:
            if not in_extreme:
                if z > max_threshold:
                    in_extreme = True
                    extreme_direction = 'positive'
                elif z < -max_threshold:
                    in_extreme = True
                    extreme_direction = 'negative'
            else:
                if abs(z) <= min_threshold:
                    reversions += 1
                    in_extreme = False
                    extreme_direction = None

        return reversions

    def test_cointegration(self, ticker1, ticker2, period, lag_param, reversion_max, reversion_min):
        """Main cointegration testing function"""
        data1 = yf.download(ticker1, period=period, progress=False, auto_adjust=False)
        data2 = yf.download(ticker2, period=period, progress=False, auto_adjust=False)

        if data1.empty or data2.empty:
            self.log("Error: Unable to fetch data for one or both tickers.")
            return

        if isinstance(data1.columns, pd.MultiIndex):
            prices1 = data1['Adj Close'].iloc[:, 0].dropna()
        else:
            prices1 = data1['Adj Close'].dropna()

        if isinstance(data2.columns, pd.MultiIndex):
            prices2 = data2['Adj Close'].iloc[:, 0].dropna()
        else:
            prices2 = data2['Adj Close'].dropna()

        combined = pd.DataFrame({ticker1: prices1, ticker2: prices2}).dropna()

        residuals, hedge_ratio, intercept = self.reg_and_residuals(
            combined[ticker1], combined[ticker2], ticker1, ticker2
        )

        # Convert residuals to percent deviation from the modeled value
        modeled_values = intercept + hedge_ratio * combined[ticker2]
        # avoid division by zero
        modeled_values = modeled_values.replace(0, np.nan)
        percent_residuals = (residuals / modeled_values) * 100.0
        percent_residuals = percent_residuals.dropna()

        # Normalize prices to percent of starting value for plotting
        percent_prices1 = combined[ticker1] / combined[ticker1].iloc[0] * 100.0
        percent_prices2 = combined[ticker2] / combined[ticker2].iloc[0] * 100.0

        # use population std (ddof=0) to match Profit_Calc
        spread_std = percent_residuals.std(ddof=0)
        reversions = self.count_reversions(percent_residuals, reversion_max, reversion_min)

        correlation = combined[ticker1].corr(combined[ticker2])
        r_squared = correlation ** 2

        current_z_score = (percent_residuals.iloc[-1] - percent_residuals.mean()) / (percent_residuals.std(ddof=0) + 1e-12)

        adf_pval = self.adf_test(percent_residuals, "Residuals (epsilon, %)", lag_param)
        eg_pval = self.engle_granger_test(combined[ticker1], combined[ticker2], ticker1, ticker2)

        # Store last combined and regression params so profit simulation can reuse them
        self.last_combined = combined.copy()
        self.last_hedge_ratio = hedge_ratio
        self.last_intercept = intercept

        self.plot_residuals(percent_residuals, ticker1, ticker2, percent_prices1, percent_prices2)

        self.log(f"Regression Equation:\n{ticker1} = {intercept:.6f} + {hedge_ratio:.6f}*{ticker2} + ε")
        self.log(f"Spread Standard Deviation (σ): {spread_std:.6f}%")
        self.log(f"Correlation (R), R squared: {correlation:.6f}, {r_squared:.6f}")
        self.log(f"\nADF Test (ε ~ I(0)) P-value: {adf_pval:.6f}")
        self.log(f"Engle-Granger Test P-value: {eg_pval:.6f}")
        self.log(f"\nReversions over period: {reversions}")
        self.log(f"Current Z-Score: {current_z_score:.6f}")

    def run_profit_simulation(self):
        """Run a simple backtest using the Profit_Calc.simulate_pairs_trade logic.
        This uses the last downloaded `combined` DataFrame (require running cointegration test first).
        For simplicity the data is split into train (first half) and trade (second half).
        """
        try:
            if simulate_pairs_trade is None or fit_hedge_ratio is None or compute_spread is None:
                messagebox.showerror("Error", "Profit simulation helpers not available (could not import Profit_Calc.py)")
                return

            if self.last_combined is None:
                messagebox.showerror("Error", "No data available. Run the cointegration test first to load data.")
                return

            combined = self.last_combined.copy()
            # rename to A/B expected by Profit_Calc helpers
            df = combined.rename(columns={self.ticker1_var.get().strip().upper(): "A",
                                           self.ticker2_var.get().strip().upper(): "B"})

            if len(df) < 10:
                messagebox.showerror("Error", "Not enough data to run a meaningful simulation.")
                return

            mid = len(df) // 2
            train = df.iloc[:mid]
            trade = df.iloc[mid:]

            # fit hedge ratio on train (A = beta * B + intercept)
            beta, intercept = fit_hedge_ratio(train)

            spread_train = compute_spread(train, beta, intercept)
            mu = spread_train.mean()
            sigma = spread_train.std(ddof=0)
            if sigma == 0 or np.isnan(sigma):
                messagebox.showerror("Error", "Train spread std is zero or NaN.")
                return

            # attach beta to trade DataFrame as expected by simulate_pairs_trade
            trade._beta = beta
            spread_trade = compute_spread(trade, beta, intercept)

            enter_sd = float(self.reversion_max_var.get())
            exit_sd = float(self.reversion_min_var.get())

            sim = simulate_pairs_trade(trade, spread_trade, mu, sigma, enter_sd, exit_sd)

            # show summary in the results text
            out_lines = []
            out_lines.append(f"Hedge ratio (beta): {beta:.6f}, intercept: {intercept:.6f}")
            out_lines.append(f"Train spread mean: {mu:.6f}, std: {sigma:.6f}")
            out_lines.append(f"Enter threshold (sd): {enter_sd}, Exit threshold (sd): {exit_sd}")
            out_lines.append(f"Number of trades: {len(sim['trades'])}")
            out_lines.append(f"Total P&L (units of price): {sim['total_pnl']:.6f}")

            # per-trade details
            for idx, t in enumerate(sim['trades']):
                ei, xi = t['entry_idx'], t['exit_idx']
                pnl = np.sum(sim['pnl_daily'][ei:xi+1])
                out_lines.append(f"Trade {idx}: {ei}->{xi} entry_z={t['entry_z']:.3f}, exit_z={t['exit_z']:.3f}, pnl={pnl:.6f}")

            for line in out_lines:
                self.log(line)

            # plot spread and equity on the embedded figure
            self.figure.clear()
            ax1 = self.figure.add_subplot(2, 1, 1)
            ax1.plot(trade.index, spread_trade, label='Spread')
            ax1.axhline(mu, color='gray', linestyle='--', label='Train mean')
            ax1.axhline(mu + enter_sd*sigma, color='red', linestyle='--', label='Enter +/-')
            ax1.axhline(mu - enter_sd*sigma, color='red', linestyle='--')
            ax1.axhline(mu + exit_sd*sigma, color='orange', linestyle=':')
            ax1.axhline(mu - exit_sd*sigma, color='orange', linestyle=':')
            ax1.set_title('Spread and thresholds')
            ax1.legend()

            ax2 = self.figure.add_subplot(2, 1, 2)
            ax2.plot(trade.index, sim['equity'], label='Equity (cumulative P&L)')
            ax2.set_title('Equity curve')
            ax2.axhline(0, color='gray', linestyle='--')
            ax2.legend()

            self.figure.tight_layout()
            self.canvas.draw()

        except Exception as e:
            messagebox.showerror('Error', str(e))


if __name__ == "__main__":
    root = tk.Tk()
    app = CointegrationTestGUI(root)
    root.mainloop()
