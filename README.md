# word2vec_from_scratch

This repository can show you some word2vec model and datasets implementations.
You can change parameters and train specific model for yourself.

Tasks:
1. Clone one repository that contains a huge amount of .py files (for example)
2. Extract only Name entities from these files. Name entities is the thing thas hase type[0] == Name after running the algorithm from pygments package. 
3. Develop word2vec model with gensim or other implementation.

At this time 3 different variants of that algorithm have implemented:
1. Word2vec class from gensim
2. Vanilla word2vec. Where you're taking one word and trying to predict the score for every word in a vocabulary.
3. Negative sampling word2vec. Where you're using 2 words as input (word and context) and trying to solve a binary classification problem. 
For negative examples you should use a negative sampling trick.

How can I launch it?

```
1. git clone https://github.com/ansavinkov/word2vec_from_scratch.git
2. cd word2vec_from_scratch/scripts
3. ./build_pipeline.sh
4. ./build_service.sh
5. cd ..
6. docker compose -f ./services_cluster.yaml up
7. docker exec -it word2vec_impl-pipeline-1 /bin/bash
8. ./run_pipeline.sh
9. exit
10. docker exec -it word2vec_impl-service-1 /bin/bash
11. ./run_service.sh
12. go to browser localhost:80
13. go to /localhost/prediction/<word_from_vocabulary>
14. You can see models' result
```

