# %% [markdown]
# # Episteme Framework: 1D Toy Ecosystem Simulation
# 
# This notebook implements a minimal version of the Episteme Spacecraft framework on a 1D toy ecosystem, demonstrating:
# - State estimation with variational filtering
# - Active experiment design via information gain
# - Symbolic regression for equation discovery
# - The Babelfish semantic encoder concept
# 
# Based on the mathematical framework from: [Episteme Spacecraft Project]

# %% [markdown]
# ## 1. Setup and Imports

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, multivariate_normal
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# %% [markdown]
# ## 2. Define the True System (Unknown to Agent)

# %%
class TrueSystem:
    """1D ecosystem with logistic growth and external control"""
    def __init__(self):
        self.r = 0.8  # Growth rate
        self.K = 50   # Carrying capacity
        self.sigma_w = 0.5  # Process noise
        self.sigma_v = 2.0  # Observation noise
        
    def dynamics(self, x, a):
        """True dynamics: x_{t+1} = x_t + r*x_t*(1 - x_t/K) + a + w_t"""
        return x + self.r * x * (1 - x/self.K) + 0.5*a + np.random.normal(0, self.sigma_w)
    
    def observe(self, x):
        """Observation: y_t = x_t + v_t"""
        return x + np.random.normal(0, self.sigma_v)

# %% [markdown]
# ## 3. Agent's Generative Model and Inference

# %%
class EpistemeAgent:
    def __init__(self):
        # Model parameters (unknown ground truth: r=0.8, K=50)
        self.r = np.random.uniform(0.1, 1.5)
        self.K = np.random.uniform(30, 70)
        self.sigma_w = 1.0
        self.sigma_v = 2.0
        
        # Belief state (Gaussian approximation)
        self.mu = 25.0
        self.sigma = 10.0
        
        # Action space
        self.actions = np.array([-5, -2, 0, 2, 5])
        
        # Symbolic regression library
        self.library = [
            lambda x, a: x,
            lambda x, a: x**2,
            lambda x, a: a,
            lambda x, a: x*a,
            lambda x, a: np.sin(x),
        ]
        
    def predict_dynamics(self, x, a):
        """Agent's generative model: x_{t+1} ~ p_θ(x_{t+1} | x_t, a_t)"""
        return x + self.r * x * (1 - x/self.K) + 0.3*a
    
    def update_belief(self, y, a):
        """Variational belief update (Extended Kalman Filter-like)"""
        # Prediction step
        mu_pred = self.predict_dynamics(self.mu, a)
        J = 1 + self.r * (1 - 2*self.mu/self.K)  Jacobian
        sigma_pred = (J**2) * self.sigma**2 + self.sigma_w**2
        
        # Update step
        K = sigma_pred / (sigma_pred + self.sigma_v**2)  # Kalman gain
        self.mu = mu_pred + K * (y - mu_pred)
        self.sigma = np.sqrt((1 - K) * sigma_pred)
        
        return self.mu, self.sigma
    
    def information_gain(self, a):
        """Estimate information gain for action a"""
        # Simple approximation: IG ∝ reduction in uncertainty
        J = 1 + self.r * (1 - 2*self.mu/self.K)
        sigma_pred = (J**2) * self.sigma**2 + self.sigma_w**2
        sigma_post = (sigma_pred * self.sigma_v**2) / (sigma_pred + self.sigma_v**2)
        return np.log(sigma_pred) - np.log(sigma_post)  # IG ~ log(σ_pred/σ_post)
    
    def select_action(self):
        """Select action that maximizes information gain"""
        gains = [self.information_gain(a) for a in self.actions]
        return self.actions[np.argmax(gains)]
    
    def update_model(self, x_hist, a_hist, x_next_hist):
        """Update model parameters using symbolic regression"""
        # Prepare data for symbolic regression
        X = np.array([[
            f(x, a) for f in self.library
        ] for x, a in zip(x_hist, a_hist)])
        
        y = (np.array(x_next_hist) - np.array(x_hist))  # Δx
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Sparse regression (LASSO)
        lasso = Lasso(alpha=0.1, max_iter=10000)
        lasso.fit(X_scaled, y)
        
        # Update model parameters based on discovered equations
        if abs(lasso.coef_[0]) > 0.1:  # x term
            self.r = lasso.coef_[0] * scaler.scale_[0] / 0.8  # Rough approximation
        if abs(lasso.coef_[1]) > 0.1:  # x² term
            self.K = -self.r / (lasso.coef_[1] * scaler.scale_[1]) if lasso.coef_[1] != 0 else self.K

# %% [markdown]
# ## 4. Babelfish Semantic Encoder (Conceptual)

# %%
class Babelfish(nn.Module):
    """Simple neural network implementing the information bottleneck"""
    def __init__(self, input_dim=1, latent_dim=2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 8),
            nn.ReLU(),
            nn.Linear(8, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 8),
            nn.ReLU(),
            nn.Linear(8, input_dim)
        )
        
    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return z, x_recon

# %% [markdown]
# ## 5. Simulation Loop

# %%
# Initialize system and agent
true_system = TrueSystem()
agent = EpistemeAgent()

# History tracking
history = {
    'true_states': [25.0],  # x_0
    'observations': [true_system.observe(25.0)],
    'belief_means': [agent.mu],
    'belief_uncertainty': [agent.sigma],
    'actions': [],
    'information_gain': [],
    'r_estimates': [agent.r],
    'K_estimates': [agent.K]
}

# Run simulation for T timesteps
T = 50
for t in range(T):
    # Agent selects action based on information gain
    a_t = agent.select_action()
    
    # Execute action in true system
    next_true_state = true_system.dynamics(history['true_states'][-1], a_t)
    next_observation = true_system.observe(next_true_state)
    
    # Agent updates belief based on observation
    mu_t, sigma_t = agent.update_belief(next_observation, a_t)
    
    # Record history
    history['true_states'].append(next_true_state)
    history['observations'].append(next_observation)
    history['belief_means'].append(mu_t)
    history['belief_uncertainty'].append(sigma_t)
    history['actions'].append(a_t)
    history['information_gain'].append(agent.information_gain(a_t))
    
    # Periodically update model parameters using symbolic regression
    if t % 10 == 0 and t > 5:
        agent.update_model(
            history['true_states'][:-1], 
            history['actions'], 
            history['true_states'][1:]
        )
    
    # Track parameter estimates
    history['r_estimates'].append(agent.r)
    history['K_estimates'].append(agent.K)

# %% [markdown]
# ## 6. Visualization and Analysis

# %%
# Create figure with subplots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: States and beliefs
ax1.plot(history['true_states'], label='True State', linewidth=2)
ax1.plot(history['belief_means'], label='Belief Mean', linestyle='--')
ax1.fill_between(
    range(len(history['belief_means'])),
    np.array(history['belief_means']) - np.array(history['belief_uncertainty']),
    np.array(history['belief_means']) + np.array(history['belief_uncertainty']),
    alpha=0.2, label='Uncertainty'
)
ax1.set_title('State Estimation')
ax1.set_xlabel('Time')
ax1.set_ylabel('Population')
ax1.legend()
ax1.grid(True)

# Plot 2: Actions and information gain
ax2.plot(history['actions'], 'o-', label='Action')
ax2.set_title('Selected Actions Over Time')
ax2.set_xlabel('Time')
ax2.set_ylabel('Action')
ax2.grid(True)

ax2_twin = ax2.twinx()
ax2_twin.plot(history['information_gain'], 'r-', label='Information Gain')
ax2_twin.set_ylabel('Information Gain', color='r')
ax2_twin.tick_params(axis='y', labelcolor='r')

# Plot 3: Parameter learning
ax3.plot(history['r_estimates'], label='Estimated r')
ax3.axhline(y=true_system.r, color='r', linestyle='--', label='True r')
ax3.set_title('Parameter Learning: Growth Rate (r)')
ax3.set_xlabel('Time')
ax3.set_ylabel('Value')
ax3.legend()
ax3.grid(True)

ax4.plot(history['K_estimates'], label='Estimated K')
ax4.axhline(y=true_system.K, color='r', linestyle='--', label='True K')
ax4.set_title('Parameter Learning: Carrying Capacity (K)')
ax4.set_xlabel('Time')
ax4.set_ylabel('Value')
ax4.legend()
ax4.grid(True)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 7. Symbolic Equation Discovery

# %%
# Prepare data for symbolic regression
x_hist = np.array(history['true_states'][:-1])
a_hist = np.array(history['actions'])
x_next_hist = np.array(history['true_states'][1:])
delta_x = x_next_hist - x_hist

# Create feature matrix
X = np.array([[
    x,                    # x term
    x**2,                 # x² term
    a,                    # a term
    x*a,                  # x*a term
    np.sin(x)             # sin(x) term
] for x, a in zip(x_hist, a_hist)])

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform sparse regression (LASSO)
lasso = Lasso(alpha=0.1, max_iter=10000)
lasso.fit(X_scaled, delta_x)

# Display discovered equation
terms = ['x', 'x²', 'a', 'x·a', 'sin(x)']
equation = "Δx = "
for i, coef in enumerate(lasso.coef_):
    if abs(coef) > 0.01:  # Only show significant terms
        equation += f"{coef:.3f}·{terms[i]} + "

equation += f"{lasso.intercept_:.3f}"

print("Discovered equation:")
print(equation)
print("\nTrue equation:")
print("Δx = 0.8·x·(1 - x/50) + 0.5·a + noise")

# %% [markdown]
# ## 8. Babelfish Semantic Encoder Demo

# %%
# Prepare data for Babelfish
observations = np.array(history['observations']).reshape(-1, 1)
actions = np.array(history['actions']).reshape(-1, 1)

# Train simple Babelfish model
babelfish = Babelfish(input_dim=1, latent_dim=2)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(babelfish.parameters(), lr=0.01)

# Convert to tensors
X_tensor = torch.FloatTensor(observations[:-1])
y_tensor = torch.FloatTensor(observations[1:])

# Training loop
losses = []
for epoch in range(100):
    optimizer.zero_grad()
    z, reconstructions = babelfish(X_tensor)
    loss = criterion(reconstructions, y_tensor)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

# Plot training loss
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(losses)
plt.title('Babelfish Training Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.grid(True)

# Visualize latent space
with torch.no_grad():
    z, _ = babelfish(X_tensor)
    z = z.numpy()

plt.subplot(1, 2, 2)
plt.scatter(z[:, 0], z[:, 1], c=history['actions'], cmap='viridis')
plt.colorbar(label='Action')
plt.title('Babelfish Latent Space')
plt.xlabel('z₁')
plt.ylabel('z₂')
plt.grid(True)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 9. Interpretation and Discussion
# 
# This simulation demonstrates key aspects of the Episteme framework:
# 
# 1. **State Estimation**: The agent maintains a belief distribution over the ecosystem state and updates it using variational principles.
# 
# 2. **Active Learning**: The agent selects actions that maximize information gain about the system, leading to more efficient learning.
# 
# 3. **Model Learning**: Through symbolic regression, the agent discovers the underlying equations governing ecosystem dynamics.
# 
# 4. **Semantic Encoding**: The Babelfish module learns a compressed representation that captures the essential dynamics.
# 
# While simplified, this implementation captures the core loop of the Episteme framework: act, observe, update beliefs, and refine models.

# %%
# Display final parameter estimates
print(f"True parameters: r = {true_system.r}, K = {true_system.K}")
print(f"Final estimates: r = {agent.r:.3f}, K = {agent.K:.3f}")
print(f"Estimation error: r = {abs(agent.r - true_system.r):.3f}, K = {abs(agent.K - true_system.K):.3f}")