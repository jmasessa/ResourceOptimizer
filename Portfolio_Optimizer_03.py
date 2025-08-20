"""
Real-World Portfolio Optimizer using Evolutionary Algorithm and Vanguard ETF Data
-------------------------------------------------------------------------------
This script uses historical daily prices of five Vanguard ETFs to optimize
a portfolio using an evolutionary algorithm. The optimizer attempts to maximize
expected annual return while penalizing portfolios with high volatility (risk),
based on the real covariance between asset returns.

Key Features:
- Pulls daily adjusted close data for 5 Vanguard ETFs from Yahoo Finance
- Calculates annualized expected returns and the covariance matrix of returns
- Uses an evolutionary algorithm to evolve portfolio allocations
- Fitness function: Expected return - 0.2 * Risk (std. dev. of portfolio returns)
- Prints a table of the best portfolio per generation, showing allocations, return, and risk

Funds used:
- VOO: Vanguard 500 ETF
- VB: Vanguard Small-Cap Index ETF
- VXUS: Vanguard Total International Stock ETF
- BND: Vanguard Total Bond Market ETF
- VVNQ: Vanguard Real Estate ETF

Bonus Technical Details
    Portfolio Return = w₁*r₁ + w₂*r₂ + ... + wₙ*rₙ
    w₁, w₂, ..., wₙ = weights (allocations) to each asset
    r₁, r₂, ..., rₙ = expected returns of each asset

What's Going On Behind the Scenes
- Population Size
    - The population_size = 30 means that in every generation, the program is working with 30 different portfolio "teams".
    - These are made up of:
        - 30 children generated in the current generation
        - Selected from top 15 parents from the previous generation (the top half by fitness)
- Each child is a mix of two parents, with a little randomness added (called mutation),
- then re-balanced to make sure the weights add up to 100%.

What's Happening Over Time
- Each generation:
    - Creates 30 new portfolios (children)
    - Calculates their fitness
    - Prints the best one in that generation
    - Uses the top 15 of them to make the next 30

- So by generation 50, the idea is that the children have “evolved” into super-portfolios that are:
    - Balanced ⚖️
    - Smart 💡
    - Less risky 😌
    - And hopefully very profitable 💰
"""

# pip install yfinance pandas numpy
import yfinance as yf
import pandas as pd
import numpy as np

# Configurable parameters
tickers = ["VOO", "VB", "VXUS", "BND", "VNQ"]
population_size = 30
generations = 50

# Download historical data
raw_data = yf.download(tickers, start="2020-01-01", end="2025-01-01", auto_adjust=False)

# Select only the Adjusted Close prices
funds = raw_data['Adj Close'].dropna()

# Calculate daily returns
returns = funds.pct_change().dropna()

# Calculate annualized expected returns and covariance matrix
mean_returns = returns.mean() * 252
cov_matrix = returns.cov() * 252

num_assets = len(tickers)

# Fitness function: reward return, penalize risk
def fitness(portfolio):
    port_return = np.dot(portfolio, mean_returns)
    port_variance = np.dot(portfolio.T, np.dot(cov_matrix, portfolio))
    port_risk = np.sqrt(port_variance)
    return port_return - 0.2 * port_risk

# Initialize random portfolios
def create_population(size, num_assets):
    return np.random.dirichlet(np.ones(num_assets), size=size)

# Evolutionary process with console output per generation
def evolve(population, fitness_fn):
    print("\n📊 Portfolio Evolution by Generation")
    print(f"{'Gen':>4} | " + " | ".join(f"{name:^8}" for name in tickers) + " |  Return  |  Risk | Fitness")
    print("-" * (6 + 12 * num_assets + 25))

    for gen in range(1, generations + 1):
        scores = np.array([fitness_fn(ind) for ind in population])
        top_indices = scores.argsort()[-population_size // 2:]
        parents = population[top_indices]

        best_index = top_indices[-1]
        best_individual = population[best_index]
        portfolio_return = np.dot(best_individual, mean_returns)
        portfolio_risk = np.sqrt(np.dot(best_individual.T, np.dot(cov_matrix, best_individual)))
        fitness_score = portfolio_return - 0.2 * portfolio_risk

        print(f"{gen:>4} | " + " | ".join(f"{w:>8.2%}" for w in best_individual) +
              f" | {portfolio_return:>8.2%} | {portfolio_risk:>6.4f} | {fitness_score:>7.4f}")

        children = []
        while len(children) < population_size:
            i1, i2 = np.random.choice(len(parents), 2, replace=False)
            parent1, parent2 = parents[i1], parents[i2]
            child = (parent1 + parent2) / 2
            child += np.random.normal(0, 0.02, size=num_assets)
            child = np.abs(child) / np.sum(child)  # normalize
            children.append(child)

        population = np.array(children)

    return population

# Run the optimization
population = create_population(population_size, num_assets)
best_portfolios = evolve(population, fitness)

# Get final best portfolio
best_portfolio = max(best_portfolios, key=fitness)
portfolio_return = np.dot(best_portfolio, mean_returns)
portfolio_risk = np.sqrt(np.dot(best_portfolio.T, np.dot(cov_matrix, best_portfolio)))

# Display final result
print("\n✅ Final Optimized Portfolio:")
for name, weight in zip(tickers, best_portfolio):
    print(f"  {name}: {weight:.2%}")

print(f"\n📈 Expected Annual Return: {portfolio_return:.2%}")
print(f"⚠️  Annualized Risk (Std Dev): {portfolio_risk:.4f}")