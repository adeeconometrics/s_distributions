## Reminders

Efforts on maintaining class invariance
- only do type checking when necessary assumptions need be satisfied e.g. in `df` types are required to be `int`
- do value checking for prescribe domains of probability distributions
- find out if math.exp is more flexible than np.exp in terms of its limitations
- for list operations, built-in math functions should be used, for numpy.array operations, numpy.math functions should be used.


[PEP 8 style guide compliance](https://www.python.org/dev/peps/pep-0008/)

## Guidelines on editing P-distributions
1. check special functions
2. check efficient use of numpy math functions
    - for operations involving scalar values, use math functions
    - for operations involving array values, use numpy
    - Q: is it most efficient to use them together?
    - N: scipy works quite efficiently for array and most effectively on scalar values
3. check functions that may use built-in math operations
4. exception handling: change to f-strings
5. change summary and add keys function
6. check return types and mgf functions
    - if `float` is required, do not use `Union[float, int]`

## Possible Development Route
- remove support of plot in `pdf` and `cdf`; instead result list to be processed in its raw form
    - reason: incurs unecessary overhead with lesser control on plotting
- independent helper functions defined on the base class
    - reason: there is no strict rule against violating private functions, it may as well be public for everyone to see. Besides, the overhead of calling from parent class costs more than just calling a function outside of it. Alternatively, they may be implemented as `@staticmethod`
    - proposed helper functions:
        - `standardnorm_pdf`
        - `standardnorm_cdf`
        - `standard_cdf_inv`


## Features to be developed:
- [ ] likelihood function
    - [ ] log-likelihood function
    - [ ] Maximum likelihood estimate
    - [ ] maximum-likelihood function
- [ ] logcdf
- [ ] logpdf
- [ ] random variable generator 
- [ ] moment generating functions 
- [ ] random number generator
- [ ] point-percentage function 


## Distributions to be supported
### Discrete
#### Univariate

With Finite Support
- Benford
- BetaBinomial
- PoissonBinomial
- Zipf

With Infinite Support
- Borel
- Logarithmic

#### Multivariate 
- multinomial distribution

### Continuous
- staged for review
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
    - Lomax distribution
    - Gumbel distribution
    - Weibull distribution
    - truncated normal type 2 distribution 
    - Burr distribution 
    - Generalized Gamma distribution

Real line
- Staged for review:
- [ ] Gumbel  distribution
- [ ] Fisher's z-distribution
- [ ] Asymmetric Laplace distribution
- [ ] Generalized normal v1 distribution
- [ ] Generalized normal v2 distribution
- [ ] Generalized hyperbolic - resolve cdf and pvalue
- [ ] Hyperbolic secant distribution
- [ ] Slash distribution
- [ ] Skew Normal distribution
- [ ] Landau distribution
- [ ] Johnson's SU distribution
- [ ] variance-gamma distribution
- [ ] generalized hyperbolic: add support for CDF and pvalue
- [ ] Cauchy distribution
- [ ] Laplace distribution
- [ ] Logistic distribution
- [ ] Normal distribution
- [ ] T distribution
    
- change category:
    - Gumbel Type 1 distribution

Varying Type Support
- Staged for review:
- [ ] q-Gaussian distribution
- [ ] q-Weibull distribution
- [ ] generalized extreme value distribution
- [ ] generalized Pareto distribution
- [ ] q-exponential distribution

----
in progress
- Continuous univariate 
    - bounded interval
        [] ARGUS
        [] non-central beta

    - semi-infinite interval
        - [ ] relativistic Breit–Wigner 
        - [ ] Exponential-logarithmic*
        - [ ] exponential F**
        - [ ] Gompertz*
        - [ ] Hotelling's T-squared - needs further reading
        - [ ] hyper-Erlang**
        - [ ] inverse chi-squared scaled 
        - [ ] Kolmogorov - needs further reading
        - [ ] matrix-exponential - needs further reading
        - [ ] Maxwell–Jüttner
        - [ ] Mittag-Leffler - needs further reading
        - [ ] noncentral chi-squared
        - [ ] noncentral F - req: infinite summation
        - [ ] phase-type - needs further reading
        - [ ] poly-Weibull - needs further reading
        - [ ] Wilks's lambda

    - supported on the whole real line 
        - [ ] exponential power
        - [ ] Gaussian q
        - [ ] geometric stable - find numerical algorithms
        - [ ] Holtsmark - hypergerometric function
        - [ ] noncentral t - needs further reading
        - [ ] normal-inverse Gaussian - req: Bassel function of the third kind
        - [ ] stable - find numerical counterparts, no analytical expression is defined
        - [ ] Tracy–Widom - needs further reading
        - [ ] Voigt -  find numerical counterparts, as analytical expression is deemed [complicated](https://en.wikipedia.org/wiki/Voigt_profile)
        
    - varying types supported
        - [ ] generalized chi-squared  - needs further reading for numerical counterparts
        - [ ] shifted log-logistic - doable
        - [ ] Tukey lambda - doable

----
### Multivariate

