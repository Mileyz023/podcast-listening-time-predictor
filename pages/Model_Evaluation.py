import streamlit as st
import matplotlib.pyplot as plt
# from utils import report

st.set_page_config(page_title="Model Evaluation", layout="wide")
st.markdown("# Model Evaluation")

# === Report ===
st.markdown("## Report")
report(y_train, y_train_pred, "TRAIN")
report(y_eval, y_eval_pred, "TEST")

# === Plotting ===
st.markdown("## Plotting")
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# Scatter: Actual vs Predicted
axs[0].scatter(y_eval, y_eval_pred, alpha=0.3)
axs[0].plot([y_eval.min(), y_eval.max()], [y_eval.min(), y_eval.max()], 'r--', lw=2)
axs[0].set_title('Actual vs Predicted (Test Set)')
axs[0].set_xlabel('Actual')
axs[0].set_ylabel('Predicted')
axs[0].grid(True)

# Histogram: Residuals
residuals = y_eval - y_eval_pred
axs[1].hist(residuals, bins=50, edgecolor='k', alpha=0.7)
axs[1].set_title('Residuals Distribution (Test Set)')
axs[1].set_xlabel('Residual')
axs[1].set_ylabel('Frequency')
axs[1].grid(True)

plt.tight_layout()
plt.show()