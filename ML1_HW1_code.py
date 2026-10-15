"""Machine Learning 1, Homework 1: problems 4 and 5.
Run with: python ML1_HW1_code.py
Requires: numpy, pandas, scipy, matplotlib, scikit-learn, torch.
Keep ale.csv in the same folder as this file.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_absolute_error
import torch
from torch import nn

np.random.seed(42)
torch.manual_seed(42)
torch.set_num_threads(1)

# ALE data: first four columns are features, fifth is the target.
path = Path(__file__).resolve().parent
ale = pd.read_csv(path / 'ale.csv', header=None)
X_ale = ale.iloc[:, :4].to_numpy(dtype=float)
y_ale = ale.iloc[:, 4].to_numpy(dtype=float)
Xa, Xb, ya, yb = train_test_split(X_ale, y_ale, test_size=0.2, random_state=42)

# Problem 4: normality of held-out residuals
models = {
    'Linear': LinearRegression(),
    'SVR': make_pipeline(StandardScaler(), SVR(C=2, epsilon=0.06)),
    'Random forest': RandomForestRegressor(n_estimators=250, min_samples_leaf=2, random_state=42),
}
fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
for ax, (name, model) in zip(axes, models.items()):
    model.fit(Xa, ya)
    residuals = yb - model.predict(Xb)
    stat, p = stats.normaltest(residuals)
    print(f'P4 {name:14s} test MAE={mean_absolute_error(yb, model.predict(Xb)):.4f}, normaltest p={p:.5f}')
    stats.probplot(residuals, dist='norm', plot=ax)
    ax.set_title(f'{name} (p={p:.3g})')
fig.tight_layout()
fig.savefig(path / 'ale_qqplots.png', dpi=220)
plt.close(fig)

# Problem 5, classification: exactly the same train/test split for all models
X, y = load_breast_cancer(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scale = StandardScaler().fit(Xtr)
Xtr = scale.transform(Xtr)
Xte = scale.transform(Xte)
base = LogisticRegression(max_iter=10000).fit(Xtr, ytr)
print(f'P5 class sklearn logistic accuracy={accuracy_score(yte, base.predict(Xte)):.4f}')

# Use double precision to make probit log-probabilities stable.
Xtr_t = torch.tensor(Xtr, dtype=torch.float64)
Xte_t = torch.tensor(Xte, dtype=torch.float64)
ytr_t = torch.tensor(ytr, dtype=torch.float64)

class LinearScore(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.layer = nn.Linear(n_features, 1, dtype=torch.float64)

    def forward(self, x):
        return self.layer(x).squeeze(-1)

def train(model, x, target, loss_function, steps, lr):
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    for _ in range(steps):
        indices = torch.randperm(len(x))[:32]
        xb = x[indices]
        yb = target[indices]

        optimizer.zero_grad()
        loss = loss_function(model(xb), yb)
        loss.backward()
        optimizer.step()

    return model

# 5 Classification (a): logistic; binary cross-entropy of the sigmoid.
logistic = LinearScore(Xtr.shape[1])
log_loss = nn.BCEWithLogitsLoss()  # numerically stable sigmoid + BCE
train(logistic, Xtr_t, ytr_t, log_loss, steps=2500, lr=0.05)
with torch.no_grad():
    log_pred = (torch.sigmoid(logistic(Xte_t)) >= 0.5).numpy().astype(int)
print(f'P5 class torch logistic accuracy={accuracy_score(yte, log_pred):.4f}')

# 5 Classification (b): probit; log Phi(z) computed stably.
probit = LinearScore(Xtr.shape[1])
def probit_loss(scores, labels):
    return -(labels * torch.special.log_ndtr(scores) +
             (1 - labels) * torch.special.log_ndtr(-scores)).mean()
train(probit, Xtr_t, ytr_t, probit_loss, steps=2500, lr=0.05)
with torch.no_grad():
    probit_pred = (probit(Xte_t) >= 0).numpy().astype(int)
print(f'P5 class torch probit accuracy={accuracy_score(yte, probit_pred):.4f}')

# Problem 5, regression: same ALE split; scale features only using training data.
reg_scale = StandardScaler().fit(Xa)
Xa_t = torch.tensor(reg_scale.transform(Xa), dtype=torch.float64)
Xb_t = torch.tensor(reg_scale.transform(Xb), dtype=torch.float64)
ya_t = torch.tensor(ya, dtype=torch.float64)
base_reg = LinearRegression().fit(Xa, ya)
print(f'P5 reg sklearn linear MAE={mean_absolute_error(yb, base_reg.predict(Xb)):.4f}')

# 5 Regression (a): linear predictor fitted with SGD on mean squared error.
linear = LinearScore(Xa.shape[1])
train(linear, Xa_t, ya_t, lambda pred, actual: ((pred - actual)**2).mean(),
      steps=2500, lr=0.05)
with torch.no_grad():
    lm_pred = linear(Xb_t).numpy()
print(f'P5 reg torch MSE MAE={mean_absolute_error(yb, lm_pred):.4f}')

# 5 Regression (b): Student t negative log likelihood, nu=5.
t_reg = LinearScore(Xa.shape[1])
t_dist = torch.distributions.StudentT(df=5.0)
def t_loss(pred, actual):
    return -t_dist.log_prob(actual - pred).mean()
train(t_reg, Xa_t, ya_t, t_loss, steps=2500, lr=0.05)
with torch.no_grad():
    t_pred = t_reg(Xb_t).numpy()
print(f'P5 reg torch Student-t MAE={mean_absolute_error(yb, t_pred):.4f}')
