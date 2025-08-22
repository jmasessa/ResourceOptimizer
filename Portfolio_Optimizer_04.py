"""
Real-World Portfolio Optimizer using Evolutionary Algorithm and Vanguard ETF Data
-------------------------------------------------------------------------------
This script uses historical daily prices of five Vanguard ETFs to optimize
portfolio allocations using an evolutionary algorithm. Now extended to run
through *every possible 5-ETF combination* from a master list to find the
best-performing combo based on return vs. risk (fitness score).

Key Features:
- Pulls daily adjusted close data for Vanguard ETFs from Yahoo Finance
- Calculates annualized expected returns and the covariance matrix of returns
- Uses an evolutionary algorithm to evolve portfolio allocations
- Fitness function: Expected return - 0.2 * Risk (std. dev. of portfolio returns)
- Compares all combinations of 5 ETFs from a larger list
- Displays top ETF sets based on best portfolio fitness

Time Estimate:
- 10 ETFs → 252 combos (reasonable)
- 20 ETFs → 15,504 combos (takes a while)
- 30 ETFs → 142,506 combos (🤯 overnight job)
"""

import yfinance as yf
import pandas as pd
import numpy as np
from itertools import combinations
import time
import csv

# Configurable parameters
# --- 10 ETF list ---
# master_etf_list = ["VOO", "VB", "VXUS", "BND", "VNQ", "VTI", "VEA", "VTV", "VUG", "VYM"]
# --- 20 ETF list ---
master_etf_list = [
     "VOO",   # S&P 500
     "VTI",   # Total US Stock Market
     "VB",    # Small-Cap
     "VTV",   # Value
     "VUG",   # Growth
     "VYM",   # High Dividend Yield
     "BND",   # Total Bond Market
     "BSV",   # Short-Term Bond
     "BLV",   # Long-Term Bond
     "VXUS",  # Total International Stock
     "VEA",   # Developed Markets ex-US
     "VWO",   # Emerging Markets
     "VNQ",   # Real Estate
     "VGT",   # Information Technology
     "VPU",   # Utilities
     "VFH",   # Financials
     "VIS",   # Industrials
     "VAW",   # Materials
     "VOX",   # Communication Services
     "VDC"    # Consumer Staples
 ]
# --- 30 ETF list ---
# master_etf_list = [
#     "VOO", "VTI", "VB", "VTV", "VUG", "VYM", "BND", "BSV", "BLV", "VXUS",
#     "VEA", "VWO", "VNQ", "VGT", "VPU", "VFH", "VIS", "VAW", "VOX", "VDC",
#     "VHT",   # Health Care
#     "VDE",   # Energy
#     "VCR",   # Consumer Discretionary
#     "MGK",   # Mega Cap Growth
#     "MGV",   # Mega Cap Value
#     "VIOO",  # Small-Cap 600
#     "VIOG",  # Small-Cap Growth
#     "VIOV",  # Small-Cap Value
#     "VT",    # Global Total Market
#     "ESGV"   # ESG US Stock
# ]

population_size = 30
generations = 50

# Initialize random portfolios
def create_population(size, num_assets):
    return np.random.dirichlet(np.ones(num_assets), size=size)

# Evolutionary process with console output per generation
def evolve(population, fitness_fn):
    print("\n📊 Portfolio Evolution by Generation")
    print(f"{'Gen':>4} | " + " | ".join(f"Asset {i+1:^6}" for i in range(population.shape[1])) + " |  Return  |  Risk  | Fitness")
    print("-" * (6 + 12 * population.shape[1] + 25))

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
            child += np.random.normal(0, 0.02, size=population.shape[1])
            child = np.abs(child) / np.sum(child)
            children.append(child)

        population = np.array(children)

    return population

# Wrapper to test all ETF combinations
def run_all_combinations(etf_list, combo_size=5, top_n=5):
    results = []
    combos = list(combinations(etf_list, combo_size))
    print(f"📦 Running optimizer on {len(combos)} combinations...")

    for i, combo in enumerate(combos, start=1):
        print(f"\n🔄 [{i}/{len(combos)}] Testing combo: {combo}")

        try:
            raw_data = yf.download(combo, start="2020-01-01", end="2025-01-01", auto_adjust=False)
            funds = raw_data['Adj Close'].dropna()
            returns = funds.pct_change().dropna()

            if funds.shape[0] < 100:
                print("⚠️ Skipping: Not enough historical data")
                continue

            global mean_returns, cov_matrix
            mean_returns = returns.mean() * 252
            cov_matrix = returns.cov() * 252

            def fitness(portfolio):
                port_return = np.dot(portfolio, mean_returns)
                port_variance = np.dot(portfolio.T, np.dot(cov_matrix, portfolio))
                port_risk = np.sqrt(port_variance)
                return port_return - 0.2 * port_risk

            num_assets = len(combo)
            population = create_population(population_size, num_assets)
            best_portfolios = evolve(population, fitness)
            best_portfolio = max(best_portfolios, key=fitness)

            port_return = np.dot(best_portfolio, mean_returns)
            port_risk = np.sqrt(np.dot(best_portfolio.T, np.dot(cov_matrix, best_portfolio)))
            score = fitness(best_portfolio)

            results.append((score, combo, best_portfolio, port_return, port_risk))

        except Exception as e:
            print(f"❌ Error processing combo {combo}: {e}")
            continue

    results.sort(reverse=True, key=lambda x: x[0])
    print(f"\n✅ Finished testing. Top {top_n} combos with full allocation details:\n")

    for i, (score, combo, allocs, ret, risk) in enumerate(results[:top_n], start=1):
        print(f"🔝 Combo #{i}: {combo}")
        print("   Allocations:")
        for ticker, weight in zip(combo, allocs):
            print(f"     - {ticker}: {weight:.2%}")
        print(f"   📈 Return: {ret:.2%}")
        print(f"   ⚠️  Risk:   {risk:.4f}")
        print(f"   🧠 Fitness Score: {score:.4f}")
        print("-" * 60)

    return results[:top_n]
    save_top_results_to_csv(results[:top_n], filename="top_etf_combos.csv")
    print(f"\n📁 Top {top_n} results saved to 'top_etf_combos.csv'")

def save_top_results_to_csv(results, filename="top_etf_combos.csv"):
    # Build the header
    max_assets = max(len(combo) for _, combo, *_ in results)
    headers = []
    for i in range(max_assets):
        headers.append(f"ETF_{i+1}")
        headers.append(f"Allocation_{i+1}")
    headers += ["Return", "Risk", "Fitness"]

    # Write the rows
    with open(filename, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for score, combo, allocs, ret, risk in results:
            row = []
            for ticker, alloc in zip(combo, allocs):
                row.extend([ticker, f"{alloc:.4f}"])
            # Fill in any missing columns if fewer than max_assets
            while len(row) < len(headers) - 3:
                row.extend(["", ""])
            row += [f"{ret:.4f}", f"{risk:.4f}", f"{score:.4f}"]
            writer.writerow(row)

# Run the optimizer over all combinations
top_results = run_all_combinations(master_etf_list, combo_size=5, top_n=5)