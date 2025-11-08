import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
from datetime import datetime
import os


import matplotlib.pyplot as plt

# ---------- Core logic: construct spread, compute beta on train, simulate on trade ----------
def load_prices(file_a, file_b=None, col_a="Close", col_b="Close", date_col="Date"):
    # Load one or two CSVs and return a DataFrame with Date index and columns A,B
    a = pd.read_csv(file_a, parse_dates=[date_col])
    a = a[[date_col, col_a]].rename(columns={col_a: "A"})
    if file_b:
        b = pd.read_csv(file_b, parse_dates=[date_col])
        b = b[[date_col, col_b]].rename(columns={date_col: date_col, col_b: "B"})
        df = pd.merge(a, b, on=date_col, how="inner")
    else:
        # assume single file with two columns A and B already
        if col_b in a.columns:
            a = a.rename(columns={col_b: "B"})
        df = a.rename(columns={date_col: "Date"})
    df = df.rename(columns={date_col: "Date"})
    df = df.sort_values("Date").set_index("Date")
    return df[["A", "B"]]

def fit_hedge_ratio(train):
    # linear regression A = beta * B + intercept
    x = train["B"].values
    y = train["A"].values
    beta, intercept = np.polyfit(x, y, 1)
    return beta, intercept

def compute_spread(df, beta, intercept=0.0):
    # spread = A - beta * B - intercept
    return df["A"] - beta * df["B"] - intercept

def simulate_pairs_trade(df, spread, train_mean, train_std, enter_sd, exit_sd):
    # simulate on trade period DataFrame df (aligned with spread index)
    z = (spread - train_mean) / train_std
    dates = df.index
    A = df["A"].values
    B = df["B"].values
    beta = -((np.nan) )  # placeholder for scope clarity
    # We'll assume hedge ratio already applied in spread; need beta for position sizing.
    # position sizing stored externally in caller; to keep simple, we use the hedge ratio implied
    # by spread calculation via regression saved in metadata. We'll compute daily pnl using A and B and beta from metadata.
    # But in this function we expect df has attribute 'beta' set. We'll read df._metadata if present.
    beta = getattr(df, "_beta", None)
    if beta is None:
        raise ValueError("DataFrame must have attribute _beta set to hedge ratio (df._beta = beta)")

    pos = 0  # 0: no position, +1: long spread (long A short beta*B), -1: short spread (short A long beta*B)
    pos_A = 0.0
    pos_B = 0.0
    pnl_daily = np.zeros(len(dates))
    trades = []
    entry_idx = None
    entry_z = None

    # compute daily price changes for mark-to-market P&L
    dA = np.concatenate([[0], np.diff(A)])
    dB = np.concatenate([[0], np.diff(B)])

    for i in range(len(dates)):
        zi = z.iloc[i]
        # entry logic
        if pos == 0:
            if zi > enter_sd:
                pos = -1  # short spread
                pos_A = -1.0
                pos_B = beta * 1.0
                entry_idx = i
                entry_z = zi
            elif zi < -enter_sd:
                pos = +1  # long spread
                pos_A = +1.0
                pos_B = -beta * 1.0
                entry_idx = i
                entry_z = zi
        else:
            # exit logic: symmetric thresholds using exit_sd (positive)
            if pos == -1 and zi <= exit_sd:
                # exit short
                trades.append({"entry_idx": entry_idx, "exit_idx": i, "entry_z": entry_z, "exit_z": zi})
                pos = 0
                pos_A = pos_B = 0.0
                entry_idx = None
                entry_z = None
            elif pos == +1 and zi >= -exit_sd:
                trades.append({"entry_idx": entry_idx, "exit_idx": i, "entry_z": entry_z, "exit_z": zi})
                pos = 0
                pos_A = pos_B = 0.0
                entry_idx = None
                entry_z = None

        # mark-to-market P&L for this day (pos held overnight uses price changes)
        pnl_daily[i] = pos_A * dA[i] + pos_B * dB[i]

    # If still in a position at the end, close at last price (record trade)
    if pos != 0 and entry_idx is not None:
        trades.append({"entry_idx": entry_idx, "exit_idx": len(dates)-1, "entry_z": entry_z, "exit_z": z.iloc[-1]})

    equity = np.cumsum(pnl_daily)
    total_pnl = equity[-1]
    return {
        "dates": dates,
        "z": z,
        "pnl_daily": pnl_daily,
        "equity": equity,
        "trades": trades,
        "total_pnl": total_pnl
    }

# ---------- Minimal Tkinter UI ----------
class ProfitCalcApp:
    def __init__(self, root):
        self.root = root
        root.title("Pairs Profit Calculator")
        frm = tk.Frame(root, padx=8, pady=8)
        frm.pack(fill=tk.BOTH, expand=True)

        # File selectors
        tk.Label(frm, text="File A (required)").grid(row=0, column=0, sticky="w")
        self.file_a_var = tk.StringVar()
        tk.Entry(frm, textvariable=self.file_a_var, width=50).grid(row=0, column=1)
        tk.Button(frm, text="Browse", command=self.browse_a).grid(row=0, column=2)

        tk.Label(frm, text="File B (optional)").grid(row=1, column=0, sticky="w")
        self.file_b_var = tk.StringVar()
        tk.Entry(frm, textvariable=self.file_b_var, width=50).grid(row=1, column=1)
        tk.Button(frm, text="Browse", command=self.browse_b).grid(row=1, column=2)

        # Column names
        tk.Label(frm, text="Col A (default 'Close' or column name)").grid(row=2, column=0, sticky="w")
        self.col_a_var = tk.StringVar(value="Close")
        tk.Entry(frm, textvariable=self.col_a_var).grid(row=2, column=1, sticky="w")

        tk.Label(frm, text="Col B (default 'Close' or column name)").grid(row=3, column=0, sticky="w")
        self.col_b_var = tk.StringVar(value="Close")
        tk.Entry(frm, textvariable=self.col_b_var).grid(row=3, column=1, sticky="w")

        # Date ranges
        tk.Label(frm, text="Train start (YYYY-MM-DD)").grid(row=4, column=0, sticky="w")
        self.train_start = tk.StringVar()
        tk.Entry(frm, textvariable=self.train_start).grid(row=4, column=1, sticky="w")

        tk.Label(frm, text="Train end (YYYY-MM-DD)").grid(row=5, column=0, sticky="w")
        self.train_end = tk.StringVar()
        tk.Entry(frm, textvariable=self.train_end).grid(row=5, column=1, sticky="w")

        tk.Label(frm, text="Trade start (YYYY-MM-DD)").grid(row=6, column=0, sticky="w")
        self.trade_start = tk.StringVar()
        tk.Entry(frm, textvariable=self.trade_start).grid(row=6, column=1, sticky="w")

        tk.Label(frm, text="Trade end (YYYY-MM-DD)").grid(row=7, column=0, sticky="w")
        self.trade_end = tk.StringVar()
        tk.Entry(frm, textvariable=self.trade_end).grid(row=7, column=1, sticky="w")

        # Thresholds
        tk.Label(frm, text="Enter threshold (std devs)").grid(row=8, column=0, sticky="w")
        self.enter_var = tk.DoubleVar(value=2.0)
        tk.Entry(frm, textvariable=self.enter_var).grid(row=8, column=1, sticky="w")

        tk.Label(frm, text="Exit threshold (std devs, positive)").grid(row=9, column=0, sticky="w")
        self.exit_var = tk.DoubleVar(value=0.5)
        tk.Entry(frm, textvariable=self.exit_var).grid(row=9, column=1, sticky="w")

        # Run button
        tk.Button(frm, text="Run Backtest", command=self.run).grid(row=10, column=0, columnspan=3, pady=8)

        # Output text
        self.out = tk.Text(frm, height=10, width=80)
        self.out.grid(row=11, column=0, columnspan=3, pady=8)

    def browse_a(self):
        f = filedialog.askopenfilename(title="Select File A", filetypes=[("CSV files","*.csv"),("All files","*.*")])
        if f:
            self.file_a_var.set(f)

    def browse_b(self):
        f = filedialog.askopenfilename(title="Select File B (optional)", filetypes=[("CSV files","*.csv"),("All files","*.*")])
        if f:
            self.file_b_var.set(f)

    def run(self):
        try:
            file_a = self.file_a_var.get().strip()
            file_b = self.file_b_var.get().strip() or None
            if not file_a:
                messagebox.showerror("Error", "Please select File A.")
                return
            col_a = self.col_a_var.get().strip() or "Close"
            col_b = self.col_b_var.get().strip() or "Close"

            df = load_prices(file_a, file_b, col_a, col_b, date_col="Date")

            # parse dates
            ts = lambda s: pd.to_datetime(s) if s else None
            train_s = ts(self.train_start.get().strip())
            train_e = ts(self.train_end.get().strip())
            trade_s = ts(self.trade_start.get().strip())
            trade_e = ts(self.trade_end.get().strip())

            if train_s is None or train_e is None or trade_s is None or trade_e is None:
                messagebox.showerror("Error", "Please provide all four date fields (train and trade start/end).")
                return

            train = df.loc[(df.index >= train_s) & (df.index <= train_e)]
            trade = df.loc[(df.index >= trade_s) & (df.index <= trade_e)]
            if train.empty or trade.empty:
                messagebox.showerror("Error", "Train or trade period has no overlapping data. Check dates and files.")
                return

            beta, intercept = fit_hedge_ratio(train)
            spread_train = compute_spread(train, beta, intercept)
            mu = spread_train.mean()
            sigma = spread_train.std(ddof=0)
            if sigma == 0 or np.isnan(sigma):
                messagebox.showerror("Error", "Train spread std is zero or NaN.")
                return

            # Attach beta to trade DataFrame for simulation
            trade._beta = beta
            spread_trade = compute_spread(trade, beta, intercept)

            enter_sd = float(self.enter_var.get())
            exit_sd = float(self.exit_var.get())

            sim = simulate_pairs_trade(trade, spread_trade, mu, sigma, enter_sd, exit_sd)

            # show summary
            out_lines = []
            out_lines.append(f"Hedge ratio (beta): {beta:.6f}, intercept: {intercept:.6f}")
            out_lines.append(f"Train spread mean: {mu:.6f}, std: {sigma:.6f}")
            out_lines.append(f"Enter threshold (sd): {enter_sd}, Exit threshold (sd): {exit_sd}")
            out_lines.append(f"Number of trades: {len(sim['trades'])}")
            out_lines.append(f"Total P&L (units of price): {sim['total_pnl']:.6f}")
            if len(sim['equity']) > 1:
                daily_rets = np.diff(sim['equity'], prepend=0)
                if daily_rets.std(ddof=0) > 0:
                    sharpe = np.mean(daily_rets) / (daily_rets.std(ddof=0) + 1e-12) * np.sqrt(252)
                    out_lines.append(f"Approx Sharpe (daily pnl): {sharpe:.3f}")
            # Per-trade P&L
            trades_info = []
            A = trade["A"].values
            B = trade["B"].values
            for t in sim["trades"]:
                ei, xi = t["entry_idx"], t["exit_idx"]
                # P&L from entry to exit inclusive (mark-to-market)
                pnl = np.sum(sim["pnl_daily"][ei:xi+1])
                trades_info.append(f"Trade {ei}->{xi}: entry_z={t['entry_z']:.3f}, exit_z={t['exit_z']:.3f}, pnl={pnl:.6f}")
            out_lines.extend(trades_info)
            self.out.delete("1.0", tk.END)
            self.out.insert(tk.END, "\n".join(out_lines))

            # Plot results: spread with z and thresholds, equity curve
            fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
            axes[0].plot(trade.index, spread_trade, label="Spread")
            axes[0].axhline(mu, color='gray', linestyle='--', label='Train mean')
            axes[0].axhline(mu + enter_sd*sigma, color='red', linestyle='--', label='Enter +/-')
            axes[0].axhline(mu - enter_sd*sigma, color='red', linestyle='--')
            axes[0].axhline(mu + exit_sd*sigma, color='orange', linestyle=':')
            axes[0].axhline(mu - exit_sd*sigma, color='orange', linestyle=':')
            axes[0].set_title("Spread and thresholds")
            axes[0].legend()

            axes[1].plot(trade.index, sim["equity"], label="Equity (cumulative P&L)")
            axes[1].set_title("Equity curve")
            axes[1].axhline(0, color='gray', linestyle='--')
            axes[1].legend()

            plt.tight_layout()
            plt.show()

        except Exception as e:
            messagebox.showerror("Error", str(e))


if __name__ == "__main__":
    root = tk.Tk()
    app = ProfitCalcApp(root)
    root.mainloop()