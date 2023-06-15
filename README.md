### Unceratinty for transformers
Experimental repository to deal with


### Custom family feud dataset

Looks like
```json
[
    {
        "question": "Name A Good Gift For Someone Who Is Always Late.",
        "link": "/question/name-a-good-gift-for-someone-who-is-always-late",
        "answers": [
            {
                "text": "Watch",
                "points": 58
            },
            {
                "text": "Alarm Clock",
                "points": 34
            },
            {
                "text": "Calendar",
                "points": 3
            }
        ]
    }
]
```

data/question_db_2.json -- deduplicated by levenstein less then 2 on lowercase questions
data/question_db_3.json -- deduplicated by SBERT ("all-mpnet-base-v2") for cos distance less then 0.17


### Close goals 
How does accuracy affected by a diversity parameter
How does probabilities changes in beam with different diversity metrics
How does distance changes betweee sequences, e.g. average levenstein


### Theory of sequence uncertainty
generally
so, there is like
p(x|context)
where you can think as context as question to chat-gpt and x is answer
more generally it could be more restricted, but whatever
and p(x) = p(xi|context+x0..xi-1)*p(xi-1|..)*p(xi-2|...)
so it's autoregressive probability distribution
with some black box able to generate p(xi)


=============================
Diversity -- check question with multiple possible answers and see is it going to answer them
Uncertainty -- ideally it should be how often you are wrong about things
So it is benchmarked on it

So it's fine to compare different methods on misclassify detection, but only if base model the same
For llm we could even extend it to rule that core model is the same

wtf if the first question is included all the options
fine tune kill diversity (mode collapsing)(how it did in gan)

=============================

=============================

encoder only model -- allow to classify, right
why it's better then encoder-decoder
this is the same reason why e2e self driving would be better
you don't squeeze it to embeddings
it's much harder to train, but it's viable
=============================

why would we have diversity
one task one life
i'm not really convinced
step by step

the sampling is questionable in malinins paper

==============================

ok, what is the idea

there is useful of entropy for sequencece
but as we can estimate the sо

==============================

idea -- proper sampling for more diverse thing
better sampling with sane answer

today: how diversity depends on temperature

lvl 1
generate 5 samples once?




