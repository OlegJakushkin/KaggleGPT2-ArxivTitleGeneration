# KaggleGPT2-ArxivTitleGeneration
Having fun with GPT2 and Arxiv Title Generation competition

Steps to reproduce:
```bash
#Download data from  https://www.kaggle.com/c/title-generation
pip3 install  pandas tqdm pandarallel gensim gpt2_client  gpt_2_simple tensorflow-gpu
python3 ./train-gpt2.py
#wait for a day
```

Tricks to remember:

 - Add your special keyword tokens to encoder
 - Use `nrows=11` to test on parts of csv files

Generated Sample:

> loss=2.19 avg=2.29
> 
> ===== SAMPLE ========
> 
> an example is as simple as it is not possible with the traditional non-linearity, however, it can provide a new perspective for studying other, more general forms of the problem of simultaneous nonlinearity.<KEYS:> noise, complex, nonlinear, linearity, new, nonlinearity, form, forms, case, cases, linear, probability, given, provide new, given; <TITLE:> simultaneous nonlinear analysis of stochastic signals <END>
> 
> <START> <TEXT:> this article discusses methods for a parameter estimation in nonlinear stochastic dynamic contrast-enhanced contrast-enhancing regression (cadar). the focus is on the estimation of the time-to-event in the cadar model. in order to achieve a more parsimonious model selection, we introduce different from existing methods for modeling the time-dependent covariance of the response variable. in this context, the classical adaboost algorithm is not available to estimate the time-dependent covariance in this paper. however, this algorithm is computationally intensive because of the computation of an eigenvector solution. hence, numerical approximations are necessary to obtain an efficient solution. the aim of the current paper is to develop an accelerated version of the adaboost algorithm, i.e. the efficient adaboost algorithm by incorporating time-dependent covariance estimation via the spectral norm. in order to illustrate the results, the adaboost algorithm is applied to a multi-year longitudinal study of the quality of life of infants born to usa. a detailed empirical investigation is conducted to compare the time-dependent covariance estimation with that of the mean time to adaost treatment.<KEYS:> model, modeling, time, difference, differences, adaboost, stochastic dynamic, solution, adaboost, estimator, estimation, estimate, paper; <TITLE:> efficient adaboost methods for parameter estimation in nonlinear   stochastic contrast-enhancing contrast-enhancing regression for short data   sequences <END>
> 
> <START> <TEXT:> stochastic variational inequality (svi), and stochastic integral equations (sin), are prominent solutions for a number of important variational problems in statistics and scientific computing. however, they pose very challenging optimization problems that have not been solved by efficient learning algorithms. in this paper we propose a new algorithm that efficiently uses the recently introduced svi optimization algorithm, a stochastic variant of an approximation algorithm called the stochastic gradient ascent (sva) algorithm. using a modified version of the alternating direction method of multipliers (admm) to solve the stochastic variational problem, we provide a convergence analysis for sva, showing that it converges to a solution of sin.<KEYS:> stochastic variational, algorithms, optimal, optimization, optimization, problems, problem, sin, algorithm, introduced svi, gradient ascent, direction, learning; <TITLE:> stochastic variational inequalities with application to a   problem with application to a stochastic variation of sin (see arxiv:0905.038225v1 [math.ru]) <END>
> 
> <START> <TEXT:> we analyze the problem of maximizing expected out-of-sample utilities in continuous time with unbounded payoff functions and stochastic environment. we show that when the environment is sufficiently smooth, the optimal rate of convergence is $\mathcal{o}($n^{-2})$ for stochastic environment whose mean function depends on $\mathcal{e} \big(h \in c_{\infty}^{n}(\mathbb{e}_{n})$). we also show that when the environment is smooth, the optimal rate of convergence is $\mathcal{o}(n^{-1/\ln(n\lambda/\lambda_{\infty}))$ and we show that this is without strong convexity property in that case for all $\lambda<1$. with a slightly modified algorithm that is inspired by the deterministic-time stochastic control problem, we show that the optimal rate of convergence can be shown to hold for the stochastic-environment case too.<KEYS:> stochastic, function, time, modified, convexity, stochastic, optimal, control, environment, equations, big, mathcal, lambda; <TITLE:> stochastic environment without strong convexity property <END>

