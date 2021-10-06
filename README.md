# s_distributions

Library for statistical distributions

currently supported features:
- probability distribution function and probability mass function
- cumulative distribution function 
- central moments 
    - mean 
    - median
    - mode 
- variance, skewness, kurtosis
- p value
- standard deviation
----
features to be developed:
- plotting probability distribution
    - fill-in between gradient 
    - vlines
    - annotation 
- likelihood function
    - log-likelihood function
    - Maximum likelihood estimate
    - maximum-likelihood function
- logcdf
- logpdf
- random variable generator 
- moment generating functions 
- entropy
- fisher information
- point-percentage function 

----
to clean:
- comments
- replace " ** " to pow(x,y)
- replace `np.log(x)` to `math.log(x,y)`
- add title of printing summary:
    - title of the distribution
    - list of parameters used 

# List of supported distributions 
---
## Discrete 
### Univariate 
- uniform distribution
- binomial distribution
- bernoulli distribution
- hypergeometric distribution
- geometric distribution
- poisson distribution
- zeta 
--- 
in progress 
- negative binomial 
- beta binomial
### Multivariate
- multinomial distribution
----
----
## Continuous
### Univariate 

- Uniform continuous
- Gaussian distribution
- T-distribution
- Cauchy distribution
- F distribution
- Chi-square
- Chi distribution
- Exponential distribution
- Pareto distribution
- Log-normal distribution
- Laplace distribution
- Logistic distribution
- Logit-normal distribution
- Weilbull distribution
- Weilbull inverse distribution
- Gumbell distribution
- Arcsine distribution

- staged for review
    - triangular distribution
    - trapezoidal distribution
    - beta distribution
    - beta-prime distribution
    - Erlang distribution
    - Rayleigh distribution
    - Maxwell-Boltzmann distribution
    - Wigner semicircle distribution
    - beta rectangular distribution
    - Bates distribution
    - continuous Bernoulli distribution
    - Balding-Nichols distribution

Semi-infinite class
- staged for review
    - Benini distribution
    - Folded Normal distribution
    - Half Logistic distribution
    - Half Normal distribution
    - Inverse Gaussian distribution
    - Inverse Gamma distribution
    - Dagum distribution
    - Davis distribution
    - Rayleigh distribution
    - Benktander Type 1 distribution
    - Benktander Type 2 distribution
    - hypoexponential distribution
    - log-Cauchy distribution
    - log-Laplace distribution
    - log-Logistic distribution
    - Inverse chi-squared distribution
    - Lévy distribution
    - Pareto distribution
    - Nakagami distribution
    <!-- - Rice -->
    - Lomax distribution
    - Gumbel distribution
    - Weibull distribution
    - truncated normal type 2 distribution 
    - Burr distribution 
    - Generalized Gamma distribution

Real line
- Staged for review:
    - Gumbel  distribution
    - Fisher's z-distribution
    - Asymmetric Laplace distribution
    - Generalized normal v1 distribution
    - Generalized normal v2 distribution
    - Generalized hyperbolic - resolve cdf and pvalue
    - Hyperbolic secant distribution
    - Slash distribution
    - Skew Normal distribution
    - Landau distribution
    - Johnson's SU distribution
    - variance-gamma distribution
    - generalized hyperbolic: add support for CDF and pvalue
    - Cauchy distribution
    - Laplace distribution
    - Logistic distribution
    - Normal distribution
    - T distribution
    
- change category:
    - Gumbel Type 1 distribution

Varying Type Support
- Staged for review:
    - q-Gaussian distribution
    - q-Weibull distribution
    - generalized extreme value distribution
    - generalized Pareto distribution
    - q-exponential distribution

----
in progress
- Continuous univariate 
    - bounded interval
        - ARGUS
        - non-central beta

    - semi-infinite interval
        - relativistic Breit–Wigner 
        - Exponential-logarithmic*
        - exponential F**
        - Gompertz*
        - Hotelling's T-squared - needs further reading
        - hyper-Erlang**
        - inverse chi-squared scaled 
        - Kolmogorov - needs further reading
        - matrix-exponential - needs further reading
        - Maxwell–Jüttner
        - Mittag-Leffler - needs further reading
        - noncentral chi-squared
        - noncentral F - req: infinite summation
        - phase-type - needs further reading
        - poly-Weibull - needs further reading
        - Wilks's lambda

    - supported on the whole real line 
        - exponential power
        - Gaussian q
        - geometric stable - find numerical algorithms
        - Holtsmark - hypergerometric function
        - noncentral t - needs further reading
        - normal-inverse Gaussian - req: Bassel function of the third kind
        - stable - find numerical counterparts, no analytical expression is defined
        - Tracy–Widom - needs further reading
        - Voigt -  find numerical counterparts, as analytical expression is deemed [complicated](https://en.wikipedia.org/wiki/Voigt_profile)
        
    - varying types supported
        - generalized chi-squared  - needs further reading for numerical counterparts
        - shifted log-logistic - doable
        - Tukey lambda - doable

----
### Multivariate
