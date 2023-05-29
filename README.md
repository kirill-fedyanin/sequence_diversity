### Unceratinty for transformers?

my local experimental repository

```python
questions  = [
    {
        'question': 'something?',
        'answer': {
            'points': 20,
            'text': 'boo'
        }
    }
]
```

data/question_db_2.json -- deduplicated by levenstein less then 2 on lowercase questions


Measuring the metrics

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














### no one lives forever

daily challenges

lvl 1
Launch sbert and measure the distances between this stuffs

I like to drink
Alcohol is my religion
I like carrots
Orange vegetable is nice








