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

