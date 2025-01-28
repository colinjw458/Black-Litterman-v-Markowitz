import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns

def get_stock_data():
    """Get one year of daily stock data"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    symbols = ['AAPL', 'MSFT', 'META', 'AMZN']
    all_data = pd.DataFrame()
    
    for symbol in symbols:
        stock = yf.Ticker(symbol)
        df = stock.history(start=start_date, end=end_date)
        if all_data.empty:
            all_data = df['Close'].rename(symbol)
        else:
            all_data = pd.concat([all_data, df['Close'].rename(symbol)], axis=1)
    
    return all_data

def create_views(sigma, mu):
    """Create square view matrix P and calculate Ω and q"""
    P = np.array([
        [1, -1, 0, 0],    # Relative view: AAPL vs MSFT
        [0, 1, 0, 0],     # Absolute view on MSFT
        [0, 0, 1, 0],     # Absolute view on META
        [0, 0, 0, 1]      # Absolute view on AMZN
    ])
    
    # Calculate Ω using the paper's formula: ωii = pi'Σpi
    omega = np.zeros((P.shape[0], P.shape[0]))
    for i in range(P.shape[0]):
        pi = P[i, :]
        omega[i, i] = pi.dot(sigma).dot(pi)
    
    # Calculate q using the paper's formula: qi = pi'μ
    q = np.zeros(P.shape[0])
    for i in range(P.shape[0]):
        pi = P[i, :]
        q[i] = pi.dot(mu)
    
    return P, omega, q

def calculate_bl_parameters(P, omega, q, sigma, mu):
    """Calculate Black-Litterman posterior parameters"""
    sigma_inv = np.linalg.inv(sigma)
    omega_inv = np.linalg.inv(omega)
    
    # First term: (Σ^-1 + P'Ω^-1P)^-1
    term1 = np.linalg.inv(sigma_inv + P.T @ omega_inv @ P)
    
    # Second term: (Σ^-1μ + P'Ω^-1q)
    term2 = sigma_inv @ mu + P.T @ omega_inv @ q.T
    
    # Posterior mean and covariance
    mu_bl = term1 @ term2
    sigma_bl = sigma + term1
    
    return mu_bl, sigma_bl

def negative_sharpe_ratio(weights, mu, sigma):
    """Calculate negative Sharpe ratio for minimization"""
    portfolio_return = weights @ mu
    portfolio_risk = np.sqrt(weights.T @ sigma @ weights)
    return -portfolio_return/portfolio_risk

def optimize_portfolio(mu, sigma):
    """Find optimal weights by maximizing Sharpe ratio"""
    n_assets = len(mu)
    
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    bounds = tuple((0, 1) for _ in range(n_assets))
    initial_weights = np.ones(n_assets) / n_assets
    
    result = minimize(
        negative_sharpe_ratio,
        initial_weights,
        args=(mu, sigma),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    
    return result.x

def rolling_window_analysis(returns, window_size=60):
    """Perform rolling window analysis"""
    rolling_weights = []
    rolling_sharpe = []
    
    for i in range(window_size, len(returns)):
        window_returns = returns.iloc[i-window_size:i]
        
        # Calculate parameters for this window
        mu = window_returns.mean().values
        sigma = window_returns.cov().values
        
        # Create views and calculate BL parameters
        P, omega, q = create_views(sigma, mu)
        mu_bl, sigma_bl = calculate_bl_parameters(P, omega, q, sigma, mu)
        
        # Get optimal weights
        weights = optimize_portfolio(mu_bl, sigma_bl)
        
        # Calculate Sharpe ratio
        portfolio_return = weights @ mu_bl
        portfolio_risk = np.sqrt(weights.T @ sigma_bl @ weights)
        sharpe = portfolio_return / portfolio_risk
        
        rolling_weights.append(weights)
        rolling_sharpe.append(sharpe)
    
    return np.array(rolling_weights), np.array(rolling_sharpe)

def optimize_markowitz_portfolio(mu, sigma, risk_aversion=1):
    """Optimize portfolio using traditional Markowitz approach"""
    n_assets = len(mu)
    
    def objective(weights):
        portfolio_return = weights @ mu
        portfolio_risk = np.sqrt(weights.T @ sigma @ weights)
        utility = portfolio_return - 0.5 * risk_aversion * (portfolio_risk ** 2)
        return -utility  # Minimize negative utility
    
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    bounds = tuple((0, 1) for _ in range(n_assets))
    initial_weights = np.ones(n_assets) / n_assets
    
    result = minimize(
        objective,
        initial_weights,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    
    return result.x

# Main execution
# Get data and calculate basic parameters
prices = get_stock_data()
returns = prices.pct_change().dropna()
mu = returns.mean().values
sigma = returns.cov().values

# Create views and calculate BL parameters
P, omega, q = create_views(sigma, mu)
mu_bl, sigma_bl = calculate_bl_parameters(P, omega, q, sigma, mu)

# Get optimal weights for both portfolios
bl_weights = optimize_portfolio(mu_bl, sigma_bl)
mw_weights = optimize_markowitz_portfolio(mu, sigma)

# Calculate performance metrics for both portfolios
bl_return = bl_weights @ mu_bl
bl_risk = np.sqrt(bl_weights.T @ sigma_bl @ bl_weights)
bl_sharpe = bl_return / bl_risk

mw_return = mw_weights @ mu
mw_risk = np.sqrt(mw_weights.T @ sigma @ mw_weights)
mw_sharpe = mw_return / mw_risk

# Perform rolling window analysis
rolling_weights, rolling_sharpe = rolling_window_analysis(returns)

# Plot results
assets = ['AAPL', 'MSFT', 'META', 'AMZN']

# Plot 1: Final BL Weights
plt.figure(figsize=(10, 6))
plt.bar(assets, bl_weights, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
plt.title('Final Black-Litterman Weights')
plt.xlabel('Assets')
plt.ylabel('Weight')
for i, w in enumerate(bl_weights):
    plt.text(i, w, f'{w:.2%}', ha='center', va='bottom')
plt.show()

# Plot 2: Final Markowitz Weights
plt.figure(figsize=(10, 6))
plt.bar(assets, mw_weights, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
plt.title('Markowitz Portfolio Weights')
plt.xlabel('Assets')
plt.ylabel('Weight')
for i, w in enumerate(mw_weights):
    plt.text(i, w, f'{w:.2%}', ha='center', va='bottom')
plt.show()

# Plot 3: Rolling Window Weights
plt.figure(figsize=(10, 6))
rolling_df = pd.DataFrame(rolling_weights, columns=assets)
plt.plot(rolling_df)
plt.title('Rolling Window Weights')
plt.xlabel('Window Number')
plt.ylabel('Weight')
plt.legend(assets, bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# Final output report
print("\n" + "="*50)
print("PORTFOLIO OPTIMIZATION REPORT")
print("="*50)

print("\n1. BLACK-LITTERMAN PORTFOLIO")
print("-"*20)
print("\nWeights:")
for asset, weight in zip(assets, bl_weights):
    print(f"{asset}: {weight:.4f}")
print(f"\nSharpe Ratio: {bl_sharpe:.4f}")
print(f"Expected Return: {bl_return:.4%}")
print(f"Portfolio Risk: {bl_risk:.4%}")

print("\n2. MARKOWITZ PORTFOLIO")
print("-"*20)
print("\nWeights:")
for asset, weight in zip(assets, mw_weights):
    print(f"{asset}: {weight:.4f}")
print(f"\nSharpe Ratio: {mw_sharpe:.4f}")
print(f"Expected Return: {mw_return:.4%}")
print(f"Portfolio Risk: {mw_risk:.4%}")

print("\n3. ROLLING WINDOW ANALYSIS")
print("-"*20)
print(f"Average Sharpe Ratio: {np.mean(rolling_sharpe):.4f}")
print(f"Sharpe Ratio Volatility: {np.std(rolling_sharpe):.4f}")