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


### Today's challenge
lvl 4
good angle of the attack?

We need to take different diversity measures and get down to them
once again, what is diversity

