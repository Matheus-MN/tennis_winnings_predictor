#import codecademylib3_seaborn
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load and investigate the data
# Preferir arquivo na mesma pasta do script; mostrar mensagem clara se ausente
csv_path = os.path.join(os.path.dirname(__file__), 'tennis_stats.csv')
if not os.path.exists(csv_path):
    print(f"Arquivo não encontrado: {csv_path}")
    print("Coloque 'tennis_stats.csv' na mesma pasta que o script ou informe o caminho completo em 'csv_path'.")
    sys.exit(1)

data = pd.read_csv(csv_path)

# Create plots folder
plot_dir = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(plot_dir, exist_ok=True)

print(data.head())
print(data.info())
print(data.describe())

# Perform exploratory analysis
plt.scatter(data['BreakPointsOpportunities'], data['Winnings'], alpha=0.4)
plt.title('Break Points Opportunities vs Winnings')
plt.xlabel('Break Points Opportunities')
plt.ylabel('Winnings')
fname = os.path.join(plot_dir, '01_breakpoints_vs_winnings.png')
plt.savefig(fname, dpi=150, bbox_inches='tight')
print(f"Saved: {fname}")
plt.close()

plt.scatter(data['FirstServeReturnPointsWon'], data['Winnings'], alpha=0.4)
plt.title('First Serve Return Points Won vs Winnings')
plt.xlabel('First Serve Return Points Won')
plt.ylabel('Winnings')
fname = os.path.join(plot_dir, '02_firstservereturn_vs_winnings.png')
plt.savefig(fname, dpi=150, bbox_inches='tight')
print(f"Saved: {fname}")
plt.close()

# Perform single feature linear regressions
features = data[['FirstServeReturnPointsWon']]
outcome = data[['Winnings']]

features_train, features_test, outcome_train, outcome_test = train_test_split(features, outcome, train_size=0.8)

model = LinearRegression()
model.fit(features_train, outcome_train)

print("Single Feature Model Score:", model.score(features_test, outcome_test))

prediction = model.predict(features_test)
plt.scatter(outcome_test, prediction, alpha=0.4)
plt.title('Predicted vs Actual Winnings (Single Feature)')
plt.xlabel('Actual Winnings')
plt.ylabel('Predicted Winnings')
plt.show()

# Perform two feature linear regressions
features = data[['BreakPointsOpportunities', 'FirstServeReturnPointsWon']]
outcome = data[['Winnings']]

features_train, features_test, outcome_train, outcome_test = train_test_split(features, outcome, train_size=0.8)

model = LinearRegression()
model.fit(features_train, outcome_train)

print("Two Feature Model Score:", model.score(features_test, outcome_test))

prediction = model.predict(features_test)
plt.scatter(outcome_test, prediction, alpha=0.4)
plt.title('Predicted vs Actual Winnings (Two Features)')
plt.xlabel('Actual Winnings')
plt.ylabel('Predicted Winnings')
plt.show()

# Perform multiple feature linear regressions
features = data[['FirstServe', 'FirstServePointsWon', 'FirstServeReturnPointsWon',
                 'SecondServePointsWon', 'SecondServeReturnPointsWon', 'Aces',
                 'BreakPointsConverted', 'BreakPointsFaced', 'BreakPointsOpportunities',
                 'BreakPointsSaved', 'DoubleFaults', 'ReturnGamesPlayed', 
                 'ReturnGamesWon', 'ReturnPointsWon', 'ServiceGamesPlayed', 
                 'ServiceGamesWon', 'TotalPointsWon', 'TotalServicePointsWon']]
outcome = data[['Winnings']]

features_train, features_test, outcome_train, outcome_test = train_test_split(features, outcome, train_size=0.8)

model = LinearRegression()
model.fit(features_train, outcome_train)

print("Multiple Feature Model Score:", model.score(features_test, outcome_test))

prediction = model.predict(features_test)
plt.scatter(outcome_test, prediction, alpha=0.4)
plt.title('Predicted vs Actual Winnings (Multiple Features)')
plt.xlabel('Actual Winnings')
plt.ylabel('Predicted Winnings')
plt.show()
fname = os.path.join(plot_dir, '05_multiple_features_model.png')
plt.savefig(fname, dpi=150, bbox_inches='tight')
print(f"Saved: {fname}")
plt.close