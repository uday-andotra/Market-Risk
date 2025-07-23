import tkinter as tk
from tkinter import ttk, messagebox
from tkcalendar import DateEntry
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from utils import *
from config import global_data, popular_etf, sp100_tickers, model_params
import re
import pandas as pd  # Added for pd.to_datetime

def validate_inputs(tickers, start_date, end_date, forecast_start, forecast_end, risk_free_rate):
    """Validate user inputs."""
    if not tickers:
        raise ValueError("No stocks selected.")
    if not all(re.match(r'^[A-Z.]+$', t) for t in tickers):
        raise ValueError("Invalid ticker format.")
    if start_date >= end_date:
        raise ValueError("Historical start date must be before end date.")
    if forecast_start >= forecast_end:
        raise ValueError("Forecast start date must be before end date.")
    if risk_free_rate < 0:
        raise ValueError("Risk-free rate must be non-negative.")

def embed_plot(fig):
    """Embed a matplotlib figure in the Tkinter window."""
    for widget in plot_frame.winfo_children():
        widget.destroy()
    fig.tight_layout(pad=1.0)  # Adjust padding to prevent cropping
    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

def insert_text(text_widget, content):
    """Insert text into a disabled Text widget."""
    text_widget.config(state='normal')
    text_widget.insert(tk.END, content)
    text_widget.config(state='disabled')
    text_widget.see(tk.END)  # Scroll to end

def load_and_analyze():
    insert_text(output_text, "Starting analysis...\n")
    try:
        # Reset cache
        reset_cache()
        
        # Get inputs
        selected_indices = stock_listbox.curselection()
        tickers = [stock_listbox.get(i) for i in selected_indices]
        start_date = start_date_entry.get_date()
        end_date = end_date_entry.get_date()
        forecast_start = forecast_start_entry.get_date()
        forecast_end = forecast_end_entry.get_date()
        risk_free_rate = float(risk_free_entry.get()) / 100
        method = method_var.get()

        # Convert date to datetime for validation
        start_date_v = datetime.combine(start_date, datetime.min.time())
        end_date_v = datetime.combine(end_date, datetime.min.time())
        forecast_start_v = datetime.combine(forecast_start, datetime.min.time())
        forecast_end_v = datetime.combine(forecast_end, datetime.min.time())

        # Validate inputs
        validate_inputs(tickers, start_date_v, end_date_v, forecast_start_v, forecast_end_v, risk_free_rate)

        # Convert to pd.Timestamp for data filtering
        start_date_ts = pd.to_datetime(start_date)
        end_date_ts = pd.to_datetime(end_date)
        forecast_start_ts = pd.to_datetime(forecast_start)
        forecast_end_ts = pd.to_datetime(forecast_end)

        insert_text(output_text, "Loading data, please wait... This may take time due to API rate limits.\n")

        # Load data
        returns, volume, tickers = load_data(tickers, start_date_ts, end_date_ts)
        if returns is None:
            insert_text(output_text, "No data loaded for portfolio.\n")
            return

        insert_text(output_text, "Data loaded successfully. Proceeding with analysis.\n")

        # Compute preliminary portfolio returns (equal weights) for KS test
        preliminary_weights = np.array(len(tickers) * [1. / len(tickers)])
        preliminary_portfolio_train = returns.dot(preliminary_weights)
        data_train_prelim = preliminary_portfolio_train.values
        best_dist, best_params, best_pvalue = select_best_distribution(data_train_prelim)
        insert_text(output_text, f'KS Test Stats for Method Choice (on preliminary portfolio returns):\n')
        insert_text(output_text, f'Best Distribution: {best_dist} with p-value {best_pvalue:.4f} and params {best_params}\n')
        insert_text(output_text, "These stats help in choosing the FMM method. If t-dist is favored, GMM (GaussianMixture) is used as it can approximate heavy-tailed distributions.\n\n")

        # Load and display popular ETF (SPY) metrics
        insert_text(output_text, "Loading benchmark ETF data...\n")
        etf_df = fetch_alpha_vantage_data(popular_etf)
        if not etf_df.empty:
            etf_adj_close = etf_df['4. close'][(etf_df.index >= start_date_ts) & (etf_df.index <= end_date_ts)]
            etf_returns = etf_adj_close.pct_change().dropna()
            etf_var = etf_returns.quantile(0.05) * 100
            etf_cvar = etf_returns[etf_returns <= etf_returns.quantile(0.05)].mean() * 100
            etf_expected_return = etf_returns.mean() * 252 * 100
            insert_text(output_text, f'Popular ETF ({popular_etf}) Metrics (Historical):\n')
            insert_text(output_text, f'Annualized Return: {etf_expected_return:.2f}%\n')
            insert_text(output_text, f'VaR (95%): {etf_var:.2f}%\n')
            insert_text(output_text, f'cVaR (95%): {etf_cvar:.2f}%\n\n')
        else:
            insert_text(output_text, f"Failed to load data for popular ETF ({popular_etf}).\n\n")

        # Chronological split: 60% train, 20% validation, 20% test
        n = len(returns)
        train_end = int(n * 0.6)
        val_end = int(n * 0.8)
        returns_train = returns.iloc[:train_end]
        returns_val = returns.iloc[train_end:val_end]
        returns_test = returns.iloc[val_end:]

        insert_text(output_text, "Optimizing portfolio weights...\n")
        # Optimize portfolio weights
        optimal_weights = optimize_portfolio(returns_train, risk_free_rate)
        insert_text(output_text, 'Optimal Weights: ' + str(dict(zip(tickers, optimal_weights))) + '\n\n')

        # Compute portfolio returns on test data
        portfolio_returns_test = returns_test.dot(optimal_weights)

        # Compute VaR and cVaR at 95% confidence
        var = portfolio_returns_test.quantile(0.05) * 100
        cvar = portfolio_returns_test[portfolio_returns_test <= portfolio_returns_test.quantile(0.05)].mean() * 100

        # Compute annualized return % and Sharpe Ratio
        avg_return = portfolio_returns_test.mean() * 252 * 100
        volatility = portfolio_returns_test.std() * np.sqrt(252) * 100
        sharpe_ratio = (portfolio_returns_test.mean() * 252 - risk_free_rate) / portfolio_returns_test.std() * np.sqrt(252) if portfolio_returns_test.std() > 0 else np.nan
        insert_text(output_text, f'Portfolio Sharpe Ratio (Test): {sharpe_ratio:.4f}\n')
        insert_text(output_text, f'Annualized Return (Test): {avg_return:.2f}%\n')
        insert_text(output_text, f'VaR (95%, Test): {var:.2f}%\n')
        insert_text(output_text, f'cVaR (95%, Test): {cvar:.2f}%\n\n')

        insert_text(output_text, "Performing consistency KS test...\n")
        # Run KS test again for consistency
        portfolio_returns_train = returns_train.dot(optimal_weights)
        data_train_prelim = portfolio_returns_train.values
        best_dist, best_params, best_pvalue = select_best_distribution(data_train_prelim)
        use_gmm = best_dist == 'student_t'

        insert_text(output_text, "Running selected method: " + method + "\n")
        if method == 'MST + XGBoost':
            # Compute features
            features_train, mst_train = compute_mst_features(returns_train, tickers, volume)
            features_val, _ = compute_mst_features(returns_val, tickers, volume)
            features_test, _ = compute_mst_features(returns_test, tickers, volume)

            # Per-asset VaR/cVaR labels
            y_var_train = returns_train.quantile(0.05, axis=0)
            y_cvar_train = returns_train.apply(lambda col: col[col <= col.quantile(0.05)].mean())
            y_var_val = returns_val.quantile(0.05, axis=0)
            y_cvar_val = returns_val.apply(lambda col: col[col <= col.quantile(0.05)].mean())
            y_var_test = returns_test.quantile(0.05, axis=0)
            y_cvar_test = returns_test.apply(lambda col: col[col <= col.quantile(0.05)].mean())

            # Use XGBoost for prediction
            model_var = xgb.XGBRegressor(n_estimators=model_params['xgboost_n_estimators'], random_state=model_params['xgboost_random_state'], objective='reg:squarederror')
            model_cvar = xgb.XGBRegressor(n_estimators=model_params['xgboost_n_estimators'], random_state=model_params['xgboost_random_state'], objective='reg:squarederror')

            model_var.fit(features_train, y_var_train)
            var_pred_test = model_var.predict(features_test)
            mse_var = np.mean((var_pred_test - y_var_test) ** 2)

            model_cvar.fit(features_train, y_cvar_train)
            cvar_pred_test = model_cvar.predict(features_test)
            mse_cvar = np.mean((cvar_pred_test - y_cvar_test) ** 2)

            # Plot feature importance for VaR
            var_importance = pd.Series(model_var.feature_importances_, index=features_train.columns)
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.barplot(x=var_importance.values, y=var_importance.index, ax=ax)
            ax.set_title('Feature Importance (VaR)')
            embed_plot(fig)

            # Plot feature importance for cVaR
            cvar_importance = pd.Series(model_cvar.feature_importances_, index=features_train.columns)
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.barplot(x=cvar_importance.values, y=cvar_importance.index, ax=ax)
            ax.set_title('Feature Importance (cVaR)')
            embed_plot(fig)

            # Enhanced MST visualization with scalable layout
            fig, ax = plt.subplots(figsize=(10 + len(tickers)*0.2, 8 + len(tickers)*0.2))  # Scale size with number of stocks
            pos = nx.kamada_kawai_layout(mst_train) if len(tickers) > 20 else nx.spring_layout(mst_train)  # Kamada-Kawai for better spacing in large graphs
            nx.draw(mst_train, pos, with_labels=True, node_color='lightblue', node_size=500 * (20 / max(20, len(tickers))), font_size=10 * (20 / max(20, len(tickers))), ax=ax)
            edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in mst_train.edges(data=True)}
            nx.draw_networkx_edge_labels(mst_train, pos, edge_labels=edge_labels, font_color='red', font_size=8 * (20 / max(20, len(tickers))), ax=ax)
            ax.set_title('MST of Stock Correlations (Edge Weights Represent Correlation Distances)')
            embed_plot(fig)
            output_text.insert(tk.END, "MST Diagram Information: Nodes represent stocks. Edges connect stocks with the strongest correlations (lowest distance). Edge labels show the distance weight (lower = stronger correlation). The tree structure minimizes total edge weight while connecting all nodes, highlighting key market dependencies.\n\n")

            # Forward forecast for MST
            simulated_paths = simulate_future_returns('MST', portfolio_returns_train.values, (forecast_end - forecast_start).days)
            forecasted_return, forecasted_sharpe, forecasted_var, forecasted_cvar = compute_forecast_metrics(simulated_paths, risk_free_rate)
            output_text.insert(tk.END, f'Forecasted Annualized Return: {forecasted_return:.2f}%\n')
            output_text.insert(tk.END, f'Forecasted Sharpe Ratio: {forecasted_sharpe:.4f}\n')
            output_text.insert(tk.END, f'Forecasted VaR (95%): {forecasted_var:.2f}%\n')
            output_text.insert(tk.END, f'Forecasted cVaR (95%): {forecasted_cvar:.2f}%\n\n')
        else:  # FMM
            portfolio_returns_train = returns_train.dot(optimal_weights)
            data_train = portfolio_returns_train.values.reshape(-1, 1) if use_gmm else portfolio_returns_train.values
            if use_gmm:
                output_text.insert(tk.END, "KS favors t-dist, so using GMM (GaussianMixture) as approximation for heavy-tailed data.\n")
                gmm = GaussianMixture(n_components=model_params['fmm_n_components'], max_iter=100, random_state=model_params['xgboost_random_state'])
                gmm.fit(data_train)
                means = gmm.means_.flatten()
                weights = gmm.weights_
                covars = np.sqrt(gmm.covariances_.flatten())
                output_text.insert(tk.END, 'Gaussian FMM Parameters (Fit on Train):\n')
                output_text.insert(tk.END, f'Means: {means}\n')
                output_text.insert(tk.END, f'Scales: {covars}\n')
                output_text.insert(tk.END, f'Weights: {weights}\n')
                var_pred = np.sum([w * norm.ppf(0.05, loc=m, scale=s) for w, m, s in zip(weights, means, covars)])
                cvar_components = []
                for w, m, s in zip(weights, means, covars):
                    samples = norm.rvs(loc=m, scale=s, size=1000)
                    threshold = norm.ppf(0.05, loc=m, scale=s)
                    tail_samples = samples[samples <= threshold]
                    cvar_components.append(w * np.mean(tail_samples) if len(tail_samples) > 0 else w * threshold)
                cvar_pred = np.sum(cvar_components)
                simulated_paths = simulate_future_returns('FMM', None, (forecast_end - forecast_start).days, model_params=(means, covars, [0]*len(means), weights))
                forecasted_return, forecasted_sharpe, forecasted_var, forecasted_cvar = compute_forecast_metrics(simulated_paths, risk_free_rate)
                output_text.insert(tk.END, f'Forecasted Annualized Return: {forecasted_return:.2f}%\n')
                output_text.insert(tk.END, f'Forecasted Sharpe Ratio: {forecasted_sharpe:.4f}\n')
                output_text.insert(tk.END, f'Forecasted VaR (95%): {forecasted_var:.2f}%\n')
                output_text.insert(tk.END, f'Forecasted cVaR (95%): {forecasted_cvar:.2f}%\n\n')
            else:
                means, scales, shapes, weights = fit_skewed_normal_fmm(data_train.flatten(), n_components=model_params['fmm_n_components'])
                output_text.insert(tk.END, 'Skewed Normal FMM Parameters (Fit on Train):\n')
                output_text.insert(tk.END, f'Means: {means}\n')
                output_text.insert(tk.END, f'Scales: {scales}\n')
                output_text.insert(tk.END, f'Shapes (Skewness): {shapes}\n')
                output_text.insert(tk.END, f'Weights: {weights}\n')
                var_pred = np.sum([w * skewnorm.ppf(0.05, a=shapes[k], loc=means[k], scale=scales[k])
                                  for k, w in enumerate(weights)])
                cvar_components = []
                for k, w in enumerate(weights):
                    samples = skewnorm.rvs(a=shapes[k], loc=means[k], scale=scales[k], size=1000)
                    threshold = skewnorm.ppf(0.05, a=shapes[k], loc=means[k], scale=scales[k])
                    tail_samples = samples[samples <= threshold]
                    if len(tail_samples) > 0:
                        cvar_components.append(w * np.mean(tail_samples))
                    else:
                        cvar_components.append(w * threshold)
                cvar_pred = np.sum(cvar_components)
                simulated_paths = simulate_future_returns('FMM', None, (forecast_end - forecast_start).days, model_params=(means, scales, shapes, weights))
                forecasted_return, forecasted_sharpe, forecasted_var, forecasted_cvar = compute_forecast_metrics(simulated_paths, risk_free_rate)
                output_text.insert(tk.END, f'Forecasted Annualized Return: {forecasted_return:.2f}%\n')
                output_text.insert(tk.END, f'Forecasted Sharpe Ratio: {forecasted_sharpe:.4f}\n')
                output_text.insert(tk.END, f'Forecasted VaR (95%): {forecasted_var:.2f}%\n')
                output_text.insert(tk.END, f'Forecasted cVaR (95%): {forecasted_cvar:.2f}%\n\n')

        mse_var = (var_pred - var / 100) ** 2
        mse_cvar = (cvar_pred - cvar / 100) ** 2

        # Plot performance comparison
        fig, ax = plt.subplots(figsize=(6, 4))
        models = ['VaR', 'cVaR']
        mse_values = [mse_var, mse_cvar]
        sns.barplot(x=models, y=mse_values, hue=models, palette=['#36A2EB', '#FF6384'], legend=False, ax=ax)
        ax.set_ylabel('Mean Squared Error')
        ax.set_title(f'{method} Performance (Test MSE)')
        embed_plot(fig)

        output_text.insert(tk.END, f'VaR (95%) Test MSE: {mse_var:.6f}\n')
        output_text.insert(tk.END, f'cVaR (95%) Test MSE: {mse_cvar:.6f}\n\n')

        # Back-testing
        total_violations, total_observations, lr, p_value = back_test_var(portfolio_returns_test)
        output_text.insert(tk.END, f'Back-Test Results (Kupiec Test for VaR):\n')
        output_text.insert(tk.END, f'Total Violations: {total_violations}\n')
        output_text.insert(tk.END, f'Total Observations: {total_observations}\n')
        if np.isnan(lr):
            output_text.insert(tk.END, 'Likelihood Ratio: N/A (insufficient data for test)\n')
            output_text.insert(tk.END, 'P-Value: N/A\n')
            output_text.insert(tk.END, 'Model could not be tested due to limited observations.\n')
        else:
            output_text.insert(tk.END, f'Likelihood Ratio: {lr:.4f}\n')
            output_text.insert(tk.END, f'P-Value: {p_value:.4f}\n')
            if p_value > 0.05:
                output_text.insert(tk.END, 'Model passes Kupiec test (violations match expected rate).\n')
            else:
                output_text.insert(tk.END, 'Model fails Kupiec test (violations do not match expected rate).\n')
    except Exception as e:
        output_text.insert(tk.END, f"Error: {str(e)}\n")

# Create the GUI window
root = tk.Tk()
root.title("Market Risk Estimation")

# Input frame to keep button visible
input_frame = tk.Frame(root)
input_frame.pack(side=tk.TOP, fill=tk.X)

# Frame for stock selection
tk.Label(input_frame, text="Select Stocks:").pack()
stock_frame = tk.Frame(input_frame)
stock_frame.pack()
stock_listbox = tk.Listbox(stock_frame, selectmode='multiple', width=50, height=10)
for ticker in sp100_tickers:
    stock_listbox.insert(tk.END, ticker)
stock_listbox.pack(side=tk.LEFT)
scrollbar = tk.Scrollbar(stock_frame, orient=tk.VERTICAL)
scrollbar.config(command=stock_listbox.yview)
stock_listbox.config(yscrollcommand=scrollbar.set)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

# Date selection with calendar widgets
tk.Label(input_frame, text="Hist Start Date:").pack()
start_date_entry = DateEntry(input_frame, width=47, date_pattern='yyyy-mm-dd')
start_date_entry.pack()
start_date_entry.set_date(datetime(2020, 1, 1))

tk.Label(input_frame, text="Hist End Date:").pack()
end_date_entry = DateEntry(input_frame, width=47, date_pattern='yyyy-mm-dd')
end_date_entry.pack()
end_date_entry.set_date(datetime(2021, 12, 31))

tk.Label(input_frame, text="Forecast Start:").pack()
forecast_start_entry = DateEntry(input_frame, width=47, date_pattern='yyyy-mm-dd')
forecast_start_entry.pack()
forecast_start_entry.set_date(datetime(2022, 1, 1))

tk.Label(input_frame, text="Forecast End:").pack()
forecast_end_entry = DateEntry(input_frame, width=47, date_pattern='yyyy-mm-dd')
forecast_end_entry.pack()
forecast_end_entry.set_date(datetime(2022, 12, 31))

tk.Label(input_frame, text="Risk-Free Rate (%):").pack()
risk_free_entry = tk.Entry(input_frame, width=50)
risk_free_entry.pack()
risk_free_entry.insert(0, "1.0")

tk.Label(input_frame, text="Method:").pack()
method_var = tk.StringVar(value="MST + XGBoost")
ttk.Radiobutton(input_frame, text="MST + XGBoost", variable=method_var, value="MST + XGBoost").pack()
ttk.Radiobutton(input_frame, text="FMM", variable=method_var, value="FMM").pack()

# Button to run analysis (in input_frame to keep visible)
analyze_button = tk.Button(input_frame, text="Load and Analyze", command=load_and_analyze)
analyze_button.pack()

output_text = tk.Text(root, height=20, width=80, state='disabled')
output_text.pack(fill=tk.BOTH, expand=True)

# Canvas for embedding plots
plot_frame = tk.Frame(root)
plot_frame.pack(fill=tk.BOTH, expand=True)


# Start the Tkinter event loop
root.mainloop()
