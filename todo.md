## Reminders

Efforts on maintaining class invariance
- only do type checking when necessary assumptions need be satisfied e.g. in `df` types are required to be `int`
- do value checking for prescribe domains of probability distributions
- for list operations, built-in math functions should be used, for numpy.array operations, numpy.math functions should be used.


[PEP 8 style guide compliance](https://www.python.org/dev/peps/pep-0008/)

## Guidelines on editing P-distributions

1. Check for use of special functions, import only as needed in the SciPy library.
2. Properly annotate namespace that we intent to use in private use `__dunder`s to enforce name mangling.
3. Check for the efficient use of numpy math functions
   - For operations involving scalar values, use math functions
   - For operations involving array values, use numpy
   - For an expression that involve a mixture of scalar-valued operations and array-valued operations, it is best to use numpy functions when the subexpression involves at least one array, and math functions if it involves scalar values
   - Remember that SciPy works quite efficiently for array and most effectively on scalar values
4. Properly format Exceptions and use f-strings instead of string format. 
5. Change class documentation summary and add keys function.
6. When there is a logic involved in computing `pdf` or `cdf`
   - Alternatively, see if filtering in numpy performs better over list comprehension of a function
     - Iteratively evaluating a value inside a list comprehension does not have a significant overhead on defining the function inside the list comprehension; hence calling a function inside it is preferred for maintainability.
   - When the computation of `pdf` or `cdf` does not vary in regards to positioning the numpy array e.g. the expression `1 - _gammainc(a, x / b)` in computing the `cdf` of gamma distribution, it is best to put it inside a function called `__generator([params])`.
7. Always annotate functions, for value restriction, enforce value-checking.
   - if `float` is required, do not use `Union[float, int]` because it is understood.

## Possible Development Route

- [x] Remove support of plot in `pdf ` and `cdf`; instead result list to be processed in its raw form

  - reason: incurs unnecessary overhead with lesser control on plotting

- [ ] Decide whether `np.ndarray` is returned at all times. Find the best way to map a function in numpy. 
- [x] Independent helper functions defined on the base class

  - reason: there is no strict rule against violating private functions, it may as well be public for everyone to see. Besides, the overhead of calling from parent class costs more than just calling a function outside of it. Alternatively, they may be implemented as `@staticmethod`
  - proposed helper functions:
    - [x] `standardnorm_pdf` - declared as `@staticmethod`
    - [x] `standardnorm_cdf`  - declared as `@staticmethod`
    - [x] `standard_cdf_inv`  - declared as `@staticmethod`
- [ ] impose optional value checking on random variables, implement efficient predicate value checker
- [x] guarantee that all concrete class will define a pdf/cdf
- [ ] move the need for random variables only to functions that need them `pdf`, `cdf`
- [ ] make a staticmethod for computing pdf only if it performs well on iterations in defined in the 
    likelihood functions
- [ ] implementation of generic PRNG either:
    - [MCMC implementation of Metropolis-Hatings](https://en.wikipedia.org/wiki/Metropolis%E2%80%93Hastings_algorithm)
    - [Adaptive Reject Sampling](https://en.wikipedia.org/wiki/Rejection_sampling#Adaptive_rejection_sampling)
    Note: see which one is more generic `rvs` for Univariate distributions and which one is more efficient to compute.

## Features to be developed:
- [x] likelihood function
- [x] log-likelihood function
- [x] logcdf
- [x] logpdf
- [ ] Maximum likelihood estimate
- [ ] maximum-likelihood function
- [ ] random number generator
- [ ] point-percentage function 
- [ ] string representation, pretty printing
- [ ] ppf and pvalue
- [ ] higher moments function

## New Design Proposal
Proceed only if:
- Python cost significant overhead for looking up on superclass functions
- It is possible to reference concrete class attributes without specifying it in the base
	`self.function()` / `self.attribute`
- Remove `pvalue`

```python
class ConcreteDistribution(BaseCategory):
	is_initialized:bool = True # staged for abstraction, see if it degrades performance

	def __init__(self, *params, enable_cache=True):
        # typecheck and valuecheck first
		if (is_initialized := is_cached):
			self.__initialize_moments()

	def __initialize_moments(self, *params)->None:
		"""
		initialization that is cheaper to compute, avoids code duplication
		"""
		self.__mean = ...
		self.__median = ...
		self.__mode = ...
		self.__var = ...
		self.__sk = ...
		self.__ku = ... 

	def __str__(self)->str: # can be defined as default in base, regardless of performance overhead
        pairs = self.summary()
		return tabulate([[k,v] for k,v in zip(pairs.keys(), pairs.values())],
                        tablefmt="github")

	def pdf(self, x:Union[List[numeric], 
			np.ndarray, float])->Union[np.ndarray, float]: ...

	def cdf(self, x:Union[List[numeric], 
			np.ndarray, float])->Union[np.ndarray, float]: ...

	@staticmethod
	def pdf_(x:Union[List[numeric], 
			np.ndarray, float])->Union[np.ndarray, float]: ...

	@staticmethod
	def cdf_(x:Union[List[numeric], 
			np.ndarray, float])->Union[np.ndarray, float]: ...

	def mean(self)->float: return self.__mean
	def median(self)->float: return self.__median
	def mode(self)->float: return self.__mode
	def var(self)->float: return self.__var
	def ku(self)->float: return self.__ku
	def sk(self)->float: return self.__sk
	def std(self)->Optional[float]: return sqrt(self.__var)

	def summary(self)->Dict[str, float]: ...
```

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
    - [ ] Benini distribution
    - [ ] Folded Normal distribution
    - [ ] Half Logistic distribution
    - [ ] Half Normal distribution
    - [ ] Inverse Gaussian distribution
    - [ ] Inverse Gamma distribution
    - [ ] Dagum distribution
    - [ ] Davis distribution
    - [x] Rayleigh distribution
    - [ ] Benktander Type 1 distribution
    - [ ] Benktander Type 2 distribution
    - [ ] hypoexponential distribution
    - [ ] log-Cauchy distribution
    - [ ] log-Laplace distribution
    - [ ] log-Logistic distribution
    - [ ] Inverse chi-squared distribution
    - [ ] Lévy distribution
    - [x] Pareto distribution
    - [ ] Nakagami distribution
    - [ ] Lomax distribution
    - [ ] Gumbel distribution*
    - [x] Weibull distribution
    - [ ] truncated normal type 2 distribution 
    - [ ] Burr distribution 
    - [ ] Generalized Gamma distribution

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
    - [X] Cauchy distribution
    - [X] Laplace distribution
    - [X] Logistic distribution
    - [X] Normal distribution
    - [X] T distribution
    
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
        - [ ] ARGUS
        - [ ] non-central beta

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
                
    - varying types supported
        - [ ] generalized chi-squared  - needs further reading for numerical counterparts
        - [ ] shifted log-logistic - doable
        - [ ] Tukey lambda - doable

----
### Multivariate

