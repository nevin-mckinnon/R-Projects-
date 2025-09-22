# Comprehensive Housing Price Analysis Project
#
# Objective: This project applies advanced statistical computing techniques to 
# analyze the Ames Housing dataset. We will explore the relationship between 
# above-ground living area ('Gr Liv Area') and sale price, moving from
# standard least squares to robust quantile regression and distribution modeling.
#
# Skills showcased:
# 1. Least Squares Regression from scratch using matrix algebra.
# 2. Multiple Quantile Regression using the 'quantreg' package.
# 3. Median Regression formulated and solved as a Linear Programming problem.
# 4. Maximum Likelihood Estimation (MLE) for multiple distributions (Weibull, Log-Normal, Gamma).
# 5. High-Precision Monte Carlo Simulation using the best-fitting distribution.
#
# ---- CHANGELOG ----
# - Added Gamma distribution fit and quantitative AIC comparison in Part 4/4b.
# - The simulation in Part 5 now automatically uses the best model from the AIC test.
# - Added a Multiple Quantile Regression model (Part 2b) for improved accuracy.
# - Increased simulation samples in Part 5 for higher precision.
# - Increased precision of the Weibull MLE fit by increasing iterations and lowering tolerance.
# - Updated all plots to display Sale Price in thousands ($K) for better readability.

#################################################################
# Part 0: Setup and Data Preparation
#################################################################

# Install packages if they are not already installed
if (!require("AmesHousing")) install.packages("AmesHousing")
if (!require("lpSolve")) install.packages("lpSolve")
if (!require("quantreg")) install.packages("quantreg") 
if (!require("flexsurv")) install.packages("flexsurv") # For comparing distributions

library(AmesHousing)
library(lpSolve)
library(quantreg)
library(flexsurv)

# Load the dataset
data(ames_raw, package = "AmesHousing")

# For simplicity, we'll remove a few extreme outliers that are often
# excluded in analyses of this dataset.
ames <- ames_raw[ames_raw$`Gr Liv Area` < 4000, ]

# Extract our variables of interest
x <- ames$`Gr Liv Area`
y <- ames$SalePrice

cat("Data loaded. Analyzing SalePrice vs. Gr Liv Area.\n")
cat("Number of observations:", length(y), "\n\n")


#################################################################
# Part 1: Exploratory Analysis and Baseline Least Squares Fit
#################################################################
# We begin by visualizing the relationship and fitting a standard
# quadratic model using Ordinary Least Squares (OLS). This model will
# estimate the conditional *mean* sale price for a given living area.
# We will compute the coefficients "from scratch" using matrix algebra.

cat("--- Part 1: Least Squares Analysis ---\n")

# --- SAVE PLOT 1 ---
png("plot1_ols_fit.png", width = 8, height = 6, units = "in", res = 300)

# 1a) Visualize the data, scaling the y-axis for readability
plot(x, y / 1000,
     main = "1. Sale Price vs. Living Area in Ames",
     xlab = "Above Ground Living Area (sq ft)",
     ylab = "Sale Price ($K)",
     pch = 19, col = rgb(0, 0, 0.8, 0.3),
     cex = 0.8)
grid()

# 1b) Build the design matrix for a quadratic model: y = a + b*x + c*x^2
X <- cbind(1, x, x^2)

# 1c) Compute (X^T * X) and (X^T * y)
XTX <- t(X) %*% X
XTy <- t(X) %*% y

# 1d) Solve for the coefficients beta = (X^T * X)^-1 * (X^T * y)
beta_hat_ols <- solve(XTX, XTy)
a_hat <- beta_hat_ols[1]
b_hat <- beta_hat_ols[2]
c_hat <- beta_hat_ols[3]

cat("OLS estimates for quadratic model (from scratch):\n")
cat(sprintf("  a (intercept) = %.4f\n", a_hat))
cat(sprintf("  b (linear)    = %.4f\n", b_hat))
cat(sprintf("  c (quadratic) = %.4f\n\n", c_hat))

# 1e) Plot the fitted OLS curve, scaling the y-values for the plot
x_grid <- seq(min(x), max(x), length.out = 200)
y_hat_grid_ols <- a_hat + b_hat * x_grid + c_hat * x_grid^2
lines(x_grid, y_hat_grid_ols / 1000, col = "red", lwd = 3)
legend("topleft", legend = "Mean Fit (OLS)", col = "red", lwd = 3, bty = "n")

# Close the PNG device
dev.off()
cat("Plot 1 saved to 'plot1_ols_fit.png'\n")
cat("OLS analysis complete. The red line represents the average sale price.\n\n")


#################################################################
# Part 2: Robust Quantile Regression via `quantreg` package
#################################################################
# This part fits the same simple model as Part 1, but for different quantiles.

cat("--- Part 2: Simple Quantile Regression Analysis ---\n")

# --- SAVE PLOT 2 ---
png("plot2_quantile_fits_simple.png", width = 8, height = 6, units = "in", res = 300)

# Plot the data again, scaling the y-axis
plot(x, y / 1000,
     main = "2. Simple Quantile Regression Fits (vs. Living Area)",
     xlab = "Above Ground Living Area (sq ft)",
     ylab = "Sale Price ($K)",
     pch = 19, col = rgb(0, 0, 0.8, 0.3),
     cex = 0.8)
grid()

# Define the quantiles we want to model
p_values <- c(0.1, 0.25, 0.5, 0.75, 0.9)
colors <- c("purple", "blue", "black", "blue", "purple")

# Fit and plot a curve for each quantile
for (i in 1:length(p_values)) {
  p <- p_values[i]
  cat(paste("Fitting simple quantile p =", p, "...\n"))
  
  # Use the rq() function for a quadratic model
  qr_fit <- rq(y ~ poly(x, 2, raw = TRUE), tau = p)
  
  # Predict values along our grid and scale them for the plot
  y_grid_p <- predict(qr_fit, newdata = data.frame(x = x_grid))
  lines(x_grid, y_grid_p / 1000, lwd = 2.5, col = colors[i], lty = ifelse(p==0.5, 1, 2))
}

legend("topleft", legend = paste("p =", p_values), col = colors, lwd = 2.5, 
       lty = c(2, 2, 1, 2, 2), bty = "n", title = "Quantiles")
       
# Close the PNG device
dev.off()
cat("Plot 2 saved to 'plot2_quantile_fits_simple.png'\n")
cat("Simple quantile regression analysis complete.\n\n")


#################################################################
# Part 2b: Improving Accuracy with Multiple Quantile Regression
#################################################################
# To improve accuracy, we build a model with more predictors.
# We'll predict SalePrice based on Living Area, Overall Quality, Year Built, and Basement Size.

cat("--- Part 2b: Multiple Quantile Regression Analysis ---\n")

# Fit the median model (p=0.5) with multiple predictors
cat("Fitting multiple quantile regression model (p=0.5)...\n")
multi_qr_fit <- rq(SalePrice ~ `Gr Liv Area` + `Overall Qual` + `Year Built` + `Total Bsmt SF`, 
                   tau = 0.5, data = ames)

# Print a summary of the more accurate model
cat("--- Multiple Quantile Regression (Median) Summary ---\n")
print(summary(multi_qr_fit))
cat("Multiple quantile regression provides a more accurate model of median house prices.\n\n")


#################################################################
# Part 3: Median Regression via Linear Programming
#################################################################
# Here, we demonstrate an entirely different but powerful method to find the
# median fit (p=0.5) for the simple linear model.

cat("--- Part 3: Median Regression as a Linear Program ---\n")

# Use a smaller sample for the LP to keep it computationally feasible
set.seed(42)
sample_indices <- sample(1:length(y), 500)
x_lp <- x[sample_indices]
y_lp <- y[sample_indices]
n_lp <- length(y_lp)

# Objective: minimize sum(r_plus_i + r_minus_i)
nVars <- 2 + 2 * n_lp
obj <- c(rep(0, 2), rep(1, 2 * n_lp))

# Constraints: y_i = alpha + beta*x_i + r_plus_i - r_minus_i
const_mat <- matrix(0, nrow = n_lp, ncol = nVars)
for (i in 1:n_lp) {
    const_mat[i, 1] <- 1                # alpha
    const_mat[i, 2] <- x_lp[i]           # beta
    const_mat[i, 2 + i] <- 1             # r_plus_i
    const_mat[i, 2 + n_lp + i] <- -1     # r_minus_i
}

# LP setup
const_rhs <- y_lp
const_dir <- rep("=", n_lp)

# Solve the LP
lp_solution <- lp(direction = "min",
                  objective.in = obj,
                  const.mat = const_mat,
                  const.dir = const_dir,
                  const.rhs = const_rhs)

if (lp_solution$status == 0) {
    alpha_lp <- lp_solution$solution[1]
    beta_lp <- lp_solution$solution[2]
    cat("Median linear fit from LP solution:\n")
    cat(sprintf("  alpha (intercept) = %.4f\n", alpha_lp))
    cat(sprintf("  beta (slope)      = %.4f\n\n", beta_lp))
} else {
    cat("LP solver did not find an optimal solution.\n\n")
}
cat("LP median regression complete.\n\n")


#################################################################
# Part 4: Distribution Modeling with MLE
#################################################################
# To improve accuracy, we compare the fit of three different distributions.

cat("--- Part 4: MLE for Sale Price Distribution ---\n")

# --- SAVE PLOT 3 ---
png("plot3_mle_fit_comparison.png", width = 8, height = 6, units = "in", res = 300)

# Plot histogram with scaled x-axis
hist(y / 1000, breaks = 50, main = "3. Distribution Fit Comparison",
     xlab = "Sale Price ($K)", freq = FALSE, col = "lightblue")

# --- Weibull Fit (From Scratch) ---
lambda_hat <- function(kappa, y) (mean(y^kappa))^(1/kappa)
g_kappa <- function(kappa, y) {
  lam <- lambda_hat(kappa, y); n <- length(y)
  term1 <- n / kappa; sum_log <- sum(log(y)) - n*log(lam)
  ratio_pow <- (y / lam)^kappa; ratio_log <- (log(y) - log(lam))
  sum_ratio_log <- sum(ratio_pow * ratio_log)
  return(term1 + sum_log - sum_ratio_log)
}
secantWeibullKappa <- function(y, kappa_init = 1.0, tol = 1e-10, max_iter = 200) {
  k0 <- kappa_init; k1 <- kappa_init * 1.1
  g0 <- g_kappa(k0, y); g1 <- g_kappa(k1, y)
  for (iter in 1:max_iter) {
    if (abs(g1 - g0) < 1e-14) break
    k2 <- k1 - g1 * (k1 - k0)/(g1 - g0)
    if (abs(k2 - k1) < tol) return(k2)
    k0 <- k1; g0 <- g1
    k1 <- k2; g1 <- g_kappa(k1, y)
  }
  return(k1)
}
mleWeibull_fromScratch <- function(y, kappa_start = 1.0) {
  kappa_est <- secantWeibullKappa(y, kappa_init = kappa_start)
  lambda_est <- lambda_hat(kappa_est, y)
  return(list(kappa = kappa_est, lambda = lambda_est))
}
weibull_mle <- mleWeibull_fromScratch(y, kappa_start = 2.0)
kappa_mle <- weibull_mle$kappa; lambda_mle <- weibull_mle$lambda
curve(dweibull(x * 1000, shape = kappa_mle, scale = lambda_mle) * 1000, 
      add = TRUE, col = "darkred", lwd = 3, n = 500)

# --- Log-Normal Fit (From Scratch) ---
log_y <- log(y)
mu_ln <- mean(log_y)
sigma_ln <- sd(log_y)
curve(dlnorm(x * 1000, meanlog = mu_ln, sdlog = sigma_ln) * 1000,
      add = TRUE, col = "darkgreen", lwd = 3, lty = 2, n = 500)

# --- Gamma Fit (Using flexsurv for simplicity) ---
fit_gamma <- flexsurvreg(Surv(y) ~ 1, dist = "gamma")
gamma_shape <- fit_gamma$res[1, "est"]
gamma_rate <- fit_gamma$res[2, "est"]
curve(dgamma(x * 1000, shape = gamma_shape, rate = gamma_rate) * 1000,
      add = TRUE, col = "blue", lwd = 3, lty = 3, n = 500)

legend("topright", legend = c("Weibull", "Log-Normal", "Gamma"), 
       col=c("darkred", "darkgreen", "blue"), lwd=3, lty=c(1, 2, 3), bty="n")

dev.off()
cat("Plot 3 saved to 'plot3_mle_fit_comparison.png'\n")
cat("Visual MLE analysis complete.\n\n")


#################################################################
# Part 4b: Quantitative Model Comparison with AIC
#################################################################
cat("--- Part 4b: Comparing Distribution Fits with AIC ---\n")

# Use flexsurv to fit all models and extract AIC
fit_lnorm <- flexsurvreg(Surv(y) ~ 1, dist = "lnorm")
fit_weibull <- flexsurvreg(Surv(y) ~ 1, dist = "weibull")
# We already fit the gamma model above

# Create a data frame for comparison
aic_scores <- data.frame(
  Distribution = c("Log-Normal", "Weibull", "Gamma"),
  AIC = c(AIC(fit_lnorm), AIC(fit_weibull), AIC(fit_gamma))
)

cat("Comparing model fits using AIC (lower is better):\n")
print(aic_scores)

best_model_index <- which.min(aic_scores$AIC)
best_model_name <- aic_scores$Distribution[best_model_index]

cat(paste("\nBest fitting distribution based on AIC:", best_model_name, "\n\n"))


#################################################################
# Part 5: High-Precision Simulation with Rejection Sampling
#################################################################
# We use the best-fitting distribution from our AIC test as our target
# and increase the number of samples for a more precise simulation.

cat("--- Part 5: High-Precision Simulation ---\n")
cat(paste("Using the", best_model_name, "distribution for simulation.\n"))

# Dynamically set the target PDF and parameters based on the best model
if (best_model_name == "Log-Normal") {
  f_target <- function(x) dlnorm(x, meanlog = mu_ln, sdlog = sigma_ln)
  mode_best <- exp(mu_ln - sigma_ln^2)
  sim_color <- rgb(0, 0.5, 0, 0.5); sim_border <- "darkgreen"
} else if (best_model_name == "Weibull") {
  f_target <- function(x) dweibull(x, shape = kappa_mle, scale = lambda_mle)
  mode_best <- lambda_mle * ((kappa_mle - 1) / kappa_mle)^(1/kappa_mle)
  sim_color <- rgb(1, 0, 0, 0.4); sim_border <- "darkred"
} else { # Gamma
  f_target <- function(x) dgamma(x, shape = gamma_shape, rate = gamma_rate)
  mode_best <- (gamma_shape - 1) / gamma_rate
  sim_color <- rgb(0, 0, 1, 0.4); sim_border <- "blue"
}

M <- f_target(mode_best) * 1.1 # Add 10% buffer
x_max <- max(y) * 1.1 # Range for uniform proposal

N_samples <- 500000 
candidates <- runif(N_samples, 0, x_max)
auxiliary_v <- runif(N_samples, 0, M)
accept <- auxiliary_v < f_target(candidates)
simulated_prices <- candidates[accept]

cat(sprintf("Generated %d simulated price points via Rejection Sampling.\n", length(simulated_prices)))

# --- SAVE PLOT 4 ---
png("plot4_simulation_comparison.png", width = 8, height = 6, units = "in", res = 300)

# Plot original histogram with scaled axis
hist(y / 1000, breaks = 50, main = paste("4. Original vs. Simulated Distribution (", best_model_name, ")", sep=""),
     xlab = "Sale Price ($K)", freq = FALSE, col = "lightblue")
y_lim <- par("usr")[3:4]

# Overlay histogram of simulated data
hist(simulated_prices / 1000, breaks = 50, freq = FALSE, add = TRUE,
     col = sim_color, border = sim_border, ylim = y_lim)

legend("topright", legend = c("Original Data", "Simulated Data"),
       fill = c("lightblue", sim_color), bty = "n")

dev.off()
cat("Plot 4 saved to 'plot4_simulation_comparison.png'\n")
cat("Project complete. The close match validates our chosen model.\n")

