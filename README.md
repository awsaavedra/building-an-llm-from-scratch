# building-an-llm-from-scratch
# build-an-llm-from-scratch

The following was made after putting together a design spec reading and watching
the following materials:

1. [Let's reproduce GPT-2 124M Parameter model](https://www.youtube.com/watch?v=l8pRSuU81PU)
2. [Attention is all you need](https://arxiv.org/abs/1706.03762)
3. Build a Large Language Model (From Scratch) by Sebastian Raschka 

# Outputs you should see

![image](/gpt-2-model-output.png)

# Books Used 
Objective: More materials for foundational theory of Machine Learning and architecture

1. [The Little Book of Deep Learning by Francois Fleuret](https://fleuret.org/public/lbdl.pdf)
2. [Deep Learning: Foundations and Concepts 2024th Edition by Christopher M. Bishop (Author), Hugh Bishop](https://www.amazon.com/Deep-Learning-Foundations-Christopher-Bishop/dp/3031454677/ref=sr_1_1?crid=USGK1878YYVE)
3. [Deep Learning by Ian Goodfellow](https://www.deeplearningbook.org/)
4. MATHEMATICS FOR MACHINE LEARNING Marc Peter Deisenroth A. Aldo Faisal Cheng Soon Ong

# Machine Learning Papers

1. [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)
2. [Language Models are Unsupervised Multitask Learners](https://web.archive.org/web/20250105120712/https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
3. [Attention is all you need](https://arxiv.org/abs/1706.03762)
4. [llms, use byte token instead of word token?](https://arxiv.org/abs/2412.09871) Mostly something for future iterations

# Notes

1. Initialize the model from scratch by ourselves

2. Then we are going to try to surpass that model that we loaded
    - we will rediscover the weights from scratch

3. Things different from transformer architecture paper (Attention is all you need) - Figure 1 in paper
    - ChatGPT2 does not have an encoder, so the entire encoder is missing.
    - The attached cross-attention pathway that was using the encoder is missing as well
    - Shuffled layer normalized
    - An additional layer normalization was added right before the final classifier

4. We are using the schema to support the huggingface site style 