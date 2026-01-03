+++
date = '2026-01-03T19:53:15Z'
draft = true
math = true
title = 'Exploring the theory behind Variational Autoencoders (VAEs)'
summary = 'How do VAEs work?'
+++

## Theory

Consider an image, $x$. We say that $x$ has been generated from some latent variable $z$ via some non-linear function parameterised by $\theta$, written as $p_\theta(x|z)$. For each possible latent $z$, the function $p_\theta(x|z)$ specifies the distribution over all images that could be generated. If we could learn those parameters, i.e. $\theta$, then for any chosen latent variable $z$, we could generate new images from sampling from this distribution. 

However, we don't know how $z$ is distributed, nor do we know the parameters. To begin, we can assume that $z$ is drawn from a standard Gaussian distribution, $z\sim\mathcal{N}(0,I)$ - we do this because it gives us a mathematically convenient space to sample from. Then, to learn the parameters, we can take a dataset of real images and find parameters $\theta$ such that when we draw the latent variable $z\sim\mathcal{N}(0,I)$ then generate images through $p_\theta(x|z)$, the resulting samples resemble the data we actually observe. To do this, we would like to **maximise the likelihood** of seeing the real data given our model. For a single image, this likelihood, $p_\theta(x)$, is given by:
$$p_\theta(x)=\int p_\theta(x|z)\cdot p(z) dz$$
In plain English, this says that to explain an image $x$, we must consider **all possible latent variables $z$** that could have generated it, weighed by how likely each individual $z$ is under the prior distribution (i.e., the unit Gaussian). 

### Monte Carlo Approximation
In Factor Analysis, where the mapping is linear ($x=\mu+Wz+\epsilon$), this integral can be solved analytically: both the prior over $z$ and the conditional distribution $x|z$ are Gaussian, and linear transformation of a Gaussian is still Gaussian. However, in the VAE, $p_\theta(x|z)$ depends on a complicated non-linear function of $z$. This means that $x|z$ is no longer Gaussian, and the integral over all possible $z$ has no closed form solution. As a result, computing $p_\theta(x)$ directly is intractable. What can we do instead?

In principle, we could approximate the likelihood by sampling $z$, evaluating the likelihood of seeing $x$ given $z$ under the current parameters ($p_\theta(x|z)$), and then averaging across many values of $z$ (Monte Carlo technique): $$p_\theta(x)=\frac{1}{N}\sum_{n=1}^N p_\theta(x|z^{(n)}), \quad z^{(n)}\sim\mathcal{N}(0,I)$$ 
However, most samples $z$ drawn from $\mathcal{N}(0,I)$ will be poor at explaining a particular image $x$ - only a tiny fraction of samples will actually be viable for generating $x$, leading to high variance as many $p_\theta(x|z)$ will be nearly zero. As the dimensionality of $z$ grows, this makes it even less likely that randomly drawn $z$ values will explain the particular image, and so variance explodes with dimensionality. In practice, we'd need to compute this approximation separately for every single datapoint in the dataset at every single training step - so this approach isn't viable. 

### Variational Inference

What if we could instead sample viable values of $z$ *given* $x$, and then use only those in our approximation? Instead of sampling $z$ blindly from the prior, $\mathcal{N}(0,I)$, we'd instead sample from the *posterior* $p_\theta(z|x)$, which is already concentrated in the viable regions of latent space for that datapoint. This would make our Monte Carlo approximation much more efficient, since we'd only be using values of $z$ that actually explain $x$. How do we get the posterior? Let's take a look at Bayes' rule:
$$p_\theta(z|x)=\frac{p_\theta(x|z)\cdot p(z)}{p_\theta(x)}$$
We can evaluate $p_\theta(x|z)$, and we also know $p(z)$ because $z\sim\mathcal{N}(0,I)$ - but we've just seen that the denominator, $p_\theta(x)$, is intractable.

Instead, we can take an approach called **variational inference**. This means picking a family of simple, tractable distributions (e.g., Gaussians) and introducing a distribution $q_\phi(z|x)$ that *approximates* the true posterior $p_\theta(z|x)$ by adjusting its parameters $\phi$:
$$q_\phi(z|x)\approx p_\theta(z|x)$$
We already have $x$ (our data), but we need to learn the parameters $\phi$. We want $q_\phi(z|x)$ to be as *close* as possible to the true posterior - this gives us a concrete objective. We can measure 'closeness' (statistical distance) between two continuous probability distributions over the same variable using the **Kullback-Leibler (KL) divergence**:
$$D_{KL}(p||q)\doteq\int p(x)\log\frac{p(x)}{q(x)}$$
In other words, we want to minimise:
$$KL\left(q_\phi(z|x)||p_\theta(z|x)\right)$$
However - $p_\theta(z|x)=\frac{p_\theta(x|z)\cdot p(z)}{p_\theta(x)}$, and $p(x)$ is intractable. We need another approach.

### Deriving ELBO
Let's substitute in our distributions to the equation for $D_{KL}$ above:

$$
KL\left(q_\phi(z|x)||p_\theta(z|x)\right) = \int q_\phi(z|x) \cdot \log \frac{q_\phi(z|x)}{p_\theta(z|x)}$$

Rewritten in expectation notation: 

$$KL\left(q_\phi(z|x)||p_\theta(z|x)\right) = \mathbb{E}_{q_\phi}\left[\log\frac{q_\phi(z|x)}{p_\theta(z|x)}\right]$$

Using log identities:
$$KL\left(q_\phi(z|x)||p_\theta(z|x)\right) = \mathbb{E}_{q_\phi}\left[\log q_\phi(z|x) - \log p_\theta(z|x)\right]$$

Using Bayes' rule to rewrite $\log p_\theta(z|x)=\log p_\theta(x|z)+\log p(z) - \log p_\theta(x)$ and substituting in:
$$KL\left(q_\phi(z|x)||p_\theta(z|x)\right) = \mathbb{E}_{q_\phi}\left[\log q_\phi(z|x) - \log p_\theta(x|z) - \log p(z) + \log p_\theta(x) \right]$$

We can then remove $p_\theta(x)$ from the expectation because it is constant w.r.t $z$ (i.e., changing $z$ doesn't change $x$):
$$KL\left(q_\phi(z|x)||p_\theta(z|x)\right) = \mathbb{E}_{q_\phi}\left[\log q_\phi(z|x) - \log p_\theta(x|z) - \log p(z)\right] + \log p_\theta(x)$$

And then solving for $\log p_\theta(x)$ by subtracting the expectation term from both sides:
$$\log p_\theta(x) = KL\left(q_\phi(z|x)||p_\theta(z|x)\right) - \mathbb{E}_{q_\phi}\left[\log q_\phi(z|x) - \log p_\theta(x|z) - \log p(z)\right] $$

Recalling that expectation is linear (i.e., $\mathbb{E}[A-B-C]=\mathbb{E}[A]-\mathbb{E}[B]-\mathbb{E}[C]$), we can rewrite:
$$\log p_\theta(x) = KL\left(q_\phi(z|x)||p_\theta(z|x)\right) - \mathbb{E}_{q_\phi}[\log q_\phi(z|x)] + \mathbb{E}_{q_\phi}[\log p_\theta(x|z)] + \mathbb{E}_{q_\phi}[\log p(z)]$$

And that the KL divergence between two distributions of the same variable is $KL(q||p)=\mathbb{E}_q[\log q - \log p]$ - we can group $-\mathbb{E}_{q_\phi}[\log q_\phi(z|x)]$ and $\mathbb{E}_{q_\phi}[\log p(z)]$ as: $$KL\left(q_\phi(z|x)||p(z)\right)$$
To yield:
$$\log p_\theta(x) = KL\left(q_\phi(z|x)||p_\theta(z|x)\right) + \mathbb{E}_{q_\phi}[\log p_\theta(x|z)] - KL\left(q_\phi(z|x)||p(z)\right)$$

Now we have $\log p_\theta(x)$ as a function of three meaningful quantities: 
- the divergence between the approximate posterior and the true posterior, $KL\left(q_\phi(z|x)||p_\theta(z|x)\right)$;
- the divergence between the approximate posterior and the prior, $KL\left(q_\phi(z|x)||p(z)\right)$;
- and the expected log-likelihood of the data given the latents, $\mathbb{E}_{q_\phi}[\log p_\theta(x|z)]$, which measures how well samples of $z$ explain the observed $x$. 

Take special note of the term $KL\left(q_\phi(z|x)||p_\theta(z|x)\right)$: though it's intractable, we do know that this quantity is **always $\geq 0$** - it's zero if $q_\phi(z|x)=p_\theta(z|x)$, and greater otherwise. 

This means that the other two terms **must always be less than or equal to $\log p_\theta(x)$**:
- If $q_\phi(z|x)=p_\theta(z|x)$, then $$\log p_\theta(x)=\mathbb{E}_{q_\phi}[\log p_\theta(x|z)] - KL\left(q_\phi(z|x)||p(z)\right)$$ 
- If $q_\phi(z|x)\neq p_\theta(z|x)$, then $$\log p_\theta(x)>\mathbb{E}_{q_\phi}[\log p_\theta(x|z)] - KL\left(q_\phi(z|x)||p(z)\right)$$

These two terms are referred to as the **Evidence Lower Bound (ELBO)**, simply because they put a lower bound on the log-likelihood. This means that even though we **can't evaluate** the posterior KL, we **can evaluate** these two terms and use them as a proxy objective: by maximising the ELBO, we also push up the log-likelihood. 
$$\text{ELBO}\doteq\underbrace{\mathbb{E}_{z\sim q_\phi(z|x)}[\log p_\theta(x|z)]}_{\text{reconstruction term}}- \underbrace{KL\left(q_\phi(z|x)||p(z)\right)}_{\text{regulariser}}$$

$$\text{ELBO}\doteq\underbrace$$

The two terms perform two distinct roles:
- **Reconstruction term**: encourages the model to reconstruct the input well by maximising the log-likelihood of the data given the latent.
- **Regulariser**: forces the model to keep its latents close to the prior distribution - this means that we can sample from the prior $p(z)$ to generate new images.
### Learning

This leaves us three terms to evaluate:
- **$q_\phi(z|x)$**: The approximate posterior. We select a tractable distribution then fit the parameters $\phi$. 
- **$p(z)$**: The prior over latents. We pick a simple distribution like $p(z)=\mathcal{N}(0,I)$.
- **$\mathbb{E}_{z\sim q_\phi(z|x)}[\log p_\theta(x|z)]$**: The expected log-likelihood of reconstructing $x$ given latents drawn from $q_\phi(z|x)$. We approximate this expectation via the Monte Carlo approach outlined earlier - sample $z\sim q_\phi(z|x)$, compute $\log p_\theta(x|z)$, then average over samples.

This leaves us a very tractable function that we can optimise. We want to maximise the objective function $\mathcal{L}$ (equivalently minimise $-\mathcal{L}$):
$$
\mathcal{L} = \text{ELBO} = \mathbb{E}_{z\sim q_\phi(z|x)}[\log p_\theta(x|z)]-KL\left(q_\phi(z|x)||p(z)\right)
$$
We just need to fit the parameters, $\theta$ and $\phi$, using gradient descent. Let's consider how we would do that.

#### $\theta$
$\theta$ appears only in $\log p_\theta(x|z)$. For a sample $z\sim q_\phi(z|x)$, the gradient of $\mathcal{L}$ is given by:
$$\nabla_\theta \mathcal{L}=\mathbb{E}_{z\sim q_\phi(z|x)}[\log p_\theta(x|z)]$$

For a Gaussian distribution, $\log p_\theta(x|z)=\mathcal{N}(x;\mu_\theta(z), \text{diag}(\sigma^2_\theta(z)))$, which, equivalently, is:
$$\log p_\theta(x|z)=-\frac{1}{2}\sum_i\left[\log(2\pi\sigma^2_i)+\frac{(x_i-\mu_i)^2}{\sigma_i^2}\right]$$
through which we can trivially backpropagate through $\mu_\theta(z)$ and $\sigma_\theta(z)$ to calculate their gradients w.r.t $\mathcal{L}$.

#### $\phi$
$\phi$ appears twice:
1. Inside the KL to the prior, $KL\left(q_\phi(z|x)||p(z)\right)$. For a Gaussian $q_\phi=\mathcal{N}(\mu_\phi, \text{diag}(\sigma_\phi^2))$ and $p(z)=\mathcal{N}(0,I)$, this KL yields the closed form: $$KL=\frac{1}{2}\sum_{j=1}^d (\mu_j^2 + \sigma_j^2 - \log \sigma_j^2 - 1)$$
And so we can obtain the gradient of $\phi$ w.r.t. $KL$, $\nabla_\phi KL$, by backpropagating through $\mu_\phi(x)$ and $\sigma_\phi(x)$.
  
2. Inside the expectation, $\mathbb{E}_{z\sim q_\phi(z|x)}[\log p_\theta(x|z)]$, because $z\sim q_\phi(z|x)$, so we want to find the gradient of $\phi$ w.r.t. the expectation: $$\nabla_\phi\int q_\phi(z|x)\cdot \log p_\theta(x|z)dz$$
However - observe that $z$ is sampled from $q_\phi(z|x)$ which itself depends on $\phi$. Gradients cannot flow through this step, because the act of sampling breaks the computational graph. This means that while we *can* calculate the gradients of operations downstream of (after) a sampling process (as we did with $\theta$), we *cannot* calculate the gradient of anything upstream (before), with respect to the parameters of the distribution that generated the sample. We'll need to consider an alternative approach.
### Reparameterisation Trick
We have observed how the reconstruction term's gradient w.r.t. $\phi$ is blocked by the sampling step. The problem arises from:
$$z\sim q_\phi(z|x)$$
What if we could instead separate the randomness from the parameters $\phi$? Instead of sampling $z$ directly from $q_\phi(z|x)$, where $q$ is a chosen family of tractable distributions, we can instead use the learned parameters to describe a deterministic, linear transformation of a noise drawn from a separate distribution of the same family.

For example, if we assume that the posterior is Gaussian:
$$q_\phi(z|x)=\mathcal{N}(z;\mu_\phi(x), \text{diag }\sigma_\phi^2(x))$$
Then we know that $z$ will be normally distributed, centered at $\mu_\phi(x)$, and have variance $\sigma_\phi^2(x)$. 

Instead of sampling from that Gaussian, which blocks the flow of gradients w.r.t. $\phi$, we can describe an equivalent distribution via a linear transformation of noise drawn from a unit Gaussian:
$$z=\mu_\phi(x)+\sigma_\phi(x)\odot \epsilon, \quad \epsilon\sim\mathcal{N}(0,I)$$

Now the randomness is isolated in $\epsilon$, and the mapping from $\phi$ to $z$ is fully differentiable. 

We can rewrite the expectation $\mathbb{E}_{z\sim q_\phi(z|x)}[\log p_\theta(x|z)]$ as $$\mathbb{E}_{\epsilon\sim\mathcal{N}(0,I)}\left[\log p_\theta(x|\mu_\phi(x)+\sigma_\phi(x)\odot\epsilon)\right]$$

and take the gradient of the expectation w.r.t. $\phi$ as 
$$
\nabla_\phi \mathbb{E}_{\epsilon\sim\mathcal{N}(0,I)}\left[\log p_\theta(x|z)\right]
= \mathbb{E}_\epsilon\left[\nabla_z \log p_\theta(x|z) \cdot \nabla_\phi z \right]
$$
where $\nabla_\phi z = \nabla_\phi \mu_\phi(x) + (\epsilon \odot \nabla_\phi \sigma_\phi(x))$.

We can approximate this expectation with Monte Carlo estimation using samples of $\epsilon$. For $K$ samples, this is given by:
$$\frac{1}{K}\sum_{k=1}^K\log p_\theta(x|\mu_\phi(x)+\sigma_\phi(x)\odot\epsilon^{(k)}), \quad \epsilon^{(k)}\sim\mathcal{N}(0,I)$$
As $K\rightarrow\infty$, this converges to the true expectation. However, using $K=1$ (i.e., one sample per datapoint) is usually sufficient. The gradient obtained from the reparameterisation trick has low variance, and so a single draw of $\epsilon$ is typically a good estimate of the true gradient. Furthermore, when we use minibatches during training, we're averaging over many datapoints anyway, and so across the batch, noisy estimates tend to average out. 


### Objective Function

Thus, our final per-datapoint objective function is given as $$\mathcal{L}(\theta, \phi; x) = \mathbb{E}_{\epsilon\sim\mathcal{N}(0,I)}\left[\log p_\theta(x|\mu_\phi(x)+\sigma_\phi(x)\odot\epsilon)\right]-KL\left(q_\phi(z|x)||p(z)\right)$$

Over a batch of datapoints, we either sum or average $\mathcal{L}$, and then find the parameters $\theta,\phi$ that maximise it: $$\max_{\theta, \phi}\frac{1}{B}\sum_{i=1}^B\mathcal{L}(\theta, \phi; x_i)$$ 
