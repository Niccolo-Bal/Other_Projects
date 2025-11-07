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


class CointegrationTestGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Cointegration Calculator 8==>")
        self.root.geometry("1400x800")

        style = ttk.Style()
        style.configure('Header.TLabel', font=('Times New Roman', 10, 'bold'))

        self.create_widgets()

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

        self.run_button = ttk.Button(left_frame, text="Run Cointegration Test",
                                      command=self.run_test, style='Accent.TButton')
        self.run_button.grid(row=5, column=0, columnspan=2, pady=20)

        self.progress = ttk.Progressbar(left_frame, mode='indeterminate', length=300)
        self.progress.grid(row=6, column=0, columnspan=2, pady=5)

        ttk.Label(left_frame, text="Results:", style='Header.TLabel').grid(row=7, column=0, sticky=tk.W, pady=(10, 5))

        self.results_text = tk.Text(left_frame, width=55, height=22,
                                     wrap=tk.WORD, font=('Courier', 12))
        self.results_text.grid(row=8, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)

        left_frame.rowconfigure(8, weight=1)

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

    def plot_residuals(self, residuals, ticker1, ticker2, prices1, prices2):
        """Plot residuals and individual asset prices"""
        self.figure.clear()

        gs = self.figure.add_gridspec(2, 2, height_ratios=[2, 1], hspace=0.3, wspace=0.3)

        ax1 = self.figure.add_subplot(gs[0, :])
        mean = residuals.mean()
        std = residuals.std()

        ax1.plot(residuals.index, residuals.values, linewidth=1, label='Spread (ε)', color='blue')
        ax1.axhline(y=mean, color='red', linestyle='--', linewidth=1, label='Mean')
        ax1.axhline(y=mean + std, color='orange', linestyle=':', linewidth=1, label='±1σ')
        ax1.axhline(y=mean - std, color='orange', linestyle=':', linewidth=1)
        ax1.axhline(y=mean + 2*std, color='green', linestyle=':', linewidth=0.8, label='±2σ')
        ax1.axhline(y=mean - 2*std, color='green', linestyle=':', linewidth=0.8)

        ax1.set_ylabel('Residual Value ($)', fontsize=9)
        ax1.set_title(f'Spread: {ticker1} - β*{ticker2}', fontsize=10)
        ax1.legend(loc='best', fontsize=7)
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='both', labelsize=8)

        ax2 = self.figure.add_subplot(gs[1, 0])
        ax2.plot(prices1.index, prices1.values, linewidth=1, color='darkblue')
        ax2.set_xlabel('Date', fontsize=9)
        ax2.set_ylabel(f'{ticker1} Price ($)', fontsize=9)
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='both', labelsize=8)
        ax2.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax2.tick_params(axis='x', rotation=45)

        ax3 = self.figure.add_subplot(gs[1, 1])
        ax3.plot(prices2.index, prices2.values, linewidth=1, color='darkgreen')
        ax3.set_xlabel('Date', fontsize=9)
        ax3.set_ylabel(f'{ticker2} Price ($)', fontsize=9)
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
        std = residuals.std()

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

        spread_std = residuals.std()
        reversions = self.count_reversions(residuals, reversion_max, reversion_min)

        correlation = combined[ticker1].corr(combined[ticker2])
        r_squared = correlation ** 2

        current_z_score = (residuals.iloc[-1] - residuals.mean()) / residuals.std()

        adf_pval = self.adf_test(residuals, "Residuals (epsilon)", lag_param)
        eg_pval = self.engle_granger_test(combined[ticker1], combined[ticker2], ticker1, ticker2)

        self.plot_residuals(residuals, ticker1, ticker2, combined[ticker1], combined[ticker2])

        self.log(f"Regression Equation:\n{ticker1} = {intercept:.6f} + {hedge_ratio:.6f}*{ticker2} + ε")
        self.log(f"Spread Standard Deviation (σ): {spread_std:.6f}")
        self.log(f"Correlation (R), R squared: {correlation:.6f}, {r_squared:.6f}")
        self.log(f"\nADF Test (ε ~ I(0)) P-value: {adf_pval:.6f}")
        self.log(f"Engle-Granger Test P-value: {eg_pval:.6f}")
        self.log(f"\nReversions over period: {reversions}")
        self.log(f"Current Z-Score: {current_z_score:.6f}")


if __name__ == "__main__":
    root = tk.Tk()
    app = CointegrationTestGUI(root)
    root.mainloop()
