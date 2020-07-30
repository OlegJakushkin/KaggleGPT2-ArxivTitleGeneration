# KaggleGPT2-ArxivTitleGeneration
Having fun with GPT2 and Arxiv Title Generation competition

Steps to reproduce:
```bash
#Download data from  https://www.kaggle.com/c/title-generation
pip3 install  pandas tqdm pandarallel gensim gpt2_client  gpt_2_simple tensorflow-gpu
python3 ./train-gpt2.py
#wait for a day
```

Generated Sample after first 30 minutes:

> 700 | 2324.67] loss=2.89 avg=2.92
> 
> ===== SAMPLE ========
> 
> <START> <TEXT:> a recent study on the effect of a high-density population on growth in the urban network for the metropolitan area of japanese city of japan has shown that higher density may lead to lower network growth, which may lead to lower growth of its economic activity and consequently an economy shrinking. in this paper we conduct a theoretical analysis on the effects of higher population density on average gross domestic product growth and economic activity of urban networks. a simulation study on the japanese economy has also been conducted to understand what effects of urban densities in japanese rural areas and on the rural economy can be observed in the japanese economy.<KEYS:> urban, economy, economic, densities, density, study, increase, decrease, increase, studies, rural areas, study, study, study; <TITLE:> the effect of higher level densities on the urban sector of   japanese urban networks <END>
> 
> <START> <TEXT:> our goal in this chapter is to develop and analyze a set of metrics which can, in turn, serve as a way to quantify and evaluate the "system integrity" of a system composed of interacting components. we present three such metrics which are applicable to systems that possess only a single, or few components. the first one uses the "system-on-system" structure to characterize each system, the second uses a bayesian information criterion to evaluate the integrity of each component, and the third uses state-of-the-art method to characterize the total system integrity and then to assess whether or not this integrity has been violated by the system itself. we show empirically that when the integrity can be assessed with a combination of the two best known systems-on-system metrics (or their combinations), the integrity of the system can be assessed at much lower rates. furthermore, the integrity of the best known components can be assessed at much higher rates, compared to systems which contain only a few components. we conclude with an overview of possible applications to practical models of the future to the case of the system with large interacting components. the case of the system in action is of special interest. the case in which only several inputs are observed is of general interest due to applications in field of social engineering.<KEYS:> interacting, connected, interacting, system, components, components, interact, observed, empirical, interaction, information, high, lower, model, systems, large, bayesian information; <TITLE:> measurement of the integrity of the system via a bayesian information   criterion <END>
> 
> <START> <TEXT:> we describe a class of models for estimating the variance of a dependent function from a model for estimating the density of an unknown distribution (delta or log-delta). in order to evaluate the model, a nonlinear combination of the likelihood function and the density function depends on the log-d variance of some random variables. this leads to a class of model-free estimation with a penalty term on a random vector asymptotically which is a suitable model in practice. the nonlinearity is also of practical interest in the context of signal processing. we demonstrate the applicability of our class of models by studying two real data sets, one in which the distribution of a distribution is unknown and another that contains some of the observations in the distribution which could be included in the model. two new statistical methods are presented based on maximum likelihood procedures. the first, lasso, is applied to evaluate the covariance matrix for the log-d variance in the parameter density when using maximum likelihood estimators for a sparse distribution.<KEYS:> variance, variance, likelihood, variance, model, estimators, estimator, density, distribution, empirical statistics, density, density, asymptotically, asymptotically, penality term, estimation; <TITLE:> a class of nonlinear models for estimation of the nonlinear variance of   a linearly dependent function <END>
```
