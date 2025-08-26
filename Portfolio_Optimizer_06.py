import yfinance as yf
import numpy as np
import pandas as pd
from itertools import combinations
import random
from datetime import datetime

# -----------------------------
# CONFIGURATION
# -----------------------------
start_date = "2020-01-01"
end_date = datetime.today().strftime('%Y-%m-%d')

# Define your master list of ETFs (can be 20, 30, etc.)
master_etf_list = [
    'VOO', 'VB', 'VXUS', 'BND', 'VNQ', 'VTI', 'VTV', 'VUG', 'VEA', 'VWO',
    'VYM', 'VIG', 'VT', 'VFH', 'VGT', 'VAW', 'VPU', 'VIS', 'VDC', 'VOX'
]

# -----------------------------
# STEP 1: Download all ETF data once
# -----------------------------
print("📥 Downloading all ETF data once...")
raw_data = yf.download(master_etf_list, start=start_date, end=end_date, auto_adjust=False)

# Extract just the adjusted close prices (some versions of yfinance may structure it differently)
if isinstance(raw_data.columns, pd.MultiIndex):
    all_data = raw_data['Adj Close']
else:
    all_data = raw_data

print("✅ Data download complete.")

# -----------------------------
# STEP 2: Define helper functions
# -----------------------------
def calculate_fitness(weights, returns, cov_matrix):
    portfolio_return = np.dot(weights, returns)
    portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    if portfolio_risk == 0:
        return 0  # avoid division by zero
    return portfolio_return / portfolio_risk

def generate_random_weights(n):
    weights = np.random.rand(n)
    return weights / np.sum(weights)

# -----------------------------
# STEP 3: Test all combinations of 5 ETFs
# -----------------------------
best_results = []
print("🔍 Testing ETF combinations...")

for combo in combinations(master_etf_list, 5):
    try:
        combo_data = all_data[list(combo)].dropna()

        # Calculate daily returns and mean annual return
        daily_returns = combo_data.pct_change().dropna()
        mean_returns = daily_returns.mean() * 252  # 252 trading days
        cov_matrix = daily_returns.cov() * 252

        best_score = 0
        best_weights = None
        best_return = 0
        best_risk = 0

        for _ in range(50):
            weights = generate_random_weights(5)
            score = calculate_fitness(weights, mean_returns, cov_matrix)

            if score > best_score:
                best_score = score
                best_weights = weights
                best_return = np.dot(weights, mean_returns)
                best_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

        result = {
            'combo': combo,
            'allocations': best_weights,
            'return': best_return,
            'risk': best_risk,
            'fitness': best_score
        }
        best_results.append(result)

    except Exception as e:
        print(f"⚠️ Skipping combo {combo} due to error: {e}")

print("✅ Finished testing. Top 10 combos with full allocation details:\n")

# Sort by fitness score
top_results = sorted(best_results, key=lambda x: x['fitness'], reverse=True)[:10]

# -----------------------------
# STEP 4: Output to console and CSV
# -----------------------------
output_rows = []

for i, res in enumerate(top_results):
    print(f"\U0001F4C8 Combo #{i+1}: {res['combo']}")
    print("   Allocations:")
    for etf, alloc in zip(res['combo'], res['allocations']):
        print(f"     - {etf}: {alloc*100:.2f}%")
    print(f"   \U0001F4C9 Return: {res['return']*100:.2f}%")
    print(f"   ⚠️ Risk:   {res['risk']:.4f}")
    print(f"   \U0001F4CA Fitness Score: {res['fitness']:.4f}")
    print("-"*60)

    row = {etf: f"{alloc*100:.2f}%" for etf, alloc in zip(res['combo'], res['allocations'])}
    row.update({
        'Return': res['return'],
        'Risk': res['risk'],
        'Fitness Score': res['fitness']
    })
    output_rows.append(row)

# Convert to DataFrame and save to CSV
csv_df = pd.DataFrame(output_rows)
csv_df.to_csv("top_etf_combos.csv", index=False)
print("💾 Results saved to top_etf_combos.csv")