# build-an-llm-from-scratch
Just a project to build various OSS model work, for now it is ChatGPT2


# Project objectives
- [x] Initialize the model from scratch by ourselves
- [ ] Then we are going to try to surpass that model that we loaded
    - [ ] we will rediscover the weights from scratch
        - [ ] We want to process these token sequences and feed them into a transformer
            - [ ] Specifically, rearrange tokens into the `idx` variable within the the model.
                - Each **batch** should be `T` tokens setup which is shape of `(B, T)`
                - Where `T` cannot be larger than the `MAX_SEQUENCE` length
                - Two tensors:
                    - offset tensor tokens + 1
                    - labels 
                - So within the model we created a dataloader object that loads these objects into the transformer
                - Then outputs the loss (function)
Things different from transformer architecture paper (Attention is all you need) - Figure 1 in paper
    - ChatGPT2 does not have an encoder, so the entire encoder is missing.
    - The attached cross-attention pathway that was using the encoder is missing as well
    - Shuffled layer normalized
    - An additional layer normalization was added right before the final classifier
We are using the schema to support the huggingface site style 
The following was made after putting together a design spec reading and watching
the following materials.

- [ ] FINAL step will be to clean up the code
    - [ ] Put everything into functions
    - [ ] Create a main file and break up all classes into their own files
    - [ ] Make a single app file that runs every file from there
# Outputs you should see

![image](/gpt-2-model-output.png)

# Books Used 
Objective: More materials for foundational theory of Machine Learning and architecture

1. [The Little Book of Deep Learning by Francois Fleuret](https://fleuret.org/public/lbdl.pdf)
2. [Deep Learning: Foundations and Concepts 2024th Edition by Christopher M. Bishop (Author), Hugh Bishop](https://www.amazon.com/Deep-Learning-Foundations-Christopher-Bishop/dp/3031454677/ref=sr_1_1?crid=USGK1878YYVE)
3. [Deep Learning by Ian Goodfellow](https://www.deeplearningbook.org/)
4. MATHEMATICS FOR MACHINE LEARNING Marc Peter Deisenroth A. Aldo Faisal Cheng Soon Ong

# Videos used
5. [Let's reproduce GPT-2 124M Parameter model](https://www.youtube.com/watch?v=l8pRSuU81PU)
6. [Attention is all you need](https://arxiv.org/abs/1706.03762)
7. Build a Large Language Model (From Scratch) by Sebastian Raschka 

# Machine Learning Papers used

1. [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)
2. [Language Models are Unsupervised Multitask Learners](https://web.archive.org/web/20250105120712/https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
3. [Attention is all you need](https://arxiv.org/abs/1706.03762)
4. [llms, use byte token instead of word token?](https://arxiv.org/abs/2412.09871) Mostly something for future iterations
