# Score-Based Generative Modeling through Stochastic Differential Equations

This project is a part of the final year Deep Learning and Optimization at ENSAE. It is based on the research of this [article](https://arxiv.org/abs/2011.13456) which is summarized in this [blog](https://yang-song.net/blog/2021/score/). We also use a part of the code of the authors presented in their [github](https://github.com/yang-song/score_sde_pytorch/blob/main/sampling.py).

### Abstract
We present a method to generate new samples by slowly injecting noise using stochastic differential equations (SDE). This method is based on a score function that unlike models which optimize likelihood allows not to have problems with integration constant. By using score-based models and reverse SDE, we can sample new data points in a very efficient way with classic samplers or some more complex ones like Predictor-Corrector or by using the *probability flow ODE*. We implement these different methods and we run several experiments on two famous vision datasets: MNIST and Fashion MNIST.

Authors: [Julien MEREAU](https://github.com/Julien2048), [Agathe MINARO](https://github.com/agatheminaro)
