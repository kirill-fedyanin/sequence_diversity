import os
import time
import requests
from bs4 import BeautifulSoup
from random import random
import json


def question_from_page(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    candidates = soup.find_all('li')
    questions = [q for q in candidates if q.find("h3")]  # left if there is a h3 tag with a question
    return questions


def parse_question(block):
    question = block.find('h3').text
    link = block.a.get('href')
    answer_blocks = block.find_all('span')
    answers = []
    for ans in answer_blocks:
        text = ans.contents[0].text.strip()
        if ans.i:
            points = int(''.join(c for c in ans.i.text.split()[0] if c.isdigit()))
        else:
            points = None

        answers.append({'text': text, 'points': points})

    return {
        'question': question,
        'link': link,
        'answers': answers
    }


def parse_page(url):
    questions = question_from_page(url)
    results = []
    for q_block in questions:
        q = parse_question(q_block)
        results.append(q)

    return results


# parse_page('https://www.familyfeudquestions.com/Index/question_vote/limit/3/p/1')

categories = {
    'https://www.familyfeudquestions.com/Index/question_vote/limit/3/p/': 13,
    'https://www.familyfeudquestions.com/Index/question_vote/limit/4/p/': 55,
    'https://www.familyfeudquestions.com/Index/question_vote/limit/5/p/': 59,
    'https://www.familyfeudquestions.com/Index/question_vote/limit/6/p/': 59,
    'https://www.familyfeudquestions.com/Index/question_vote/limit/7/p/': 42,
    'https://www.familyfeudquestions.com/Index/fastmoney/p/': 5,
    'https://www.familyfeudquestions.com/Index/nopoints/limit/3/p/': 58,
    'https://www.familyfeudquestions.com/Index/nopoints/limit/4/p/': 118,
    'https://www.familyfeudquestions.com/Index/nopoints/limit/5/p/': 177,
    'https://www.familyfeudquestions.com/Index/nopoints/limit/6/p/': 135,
    'https://www.familyfeudquestions.com/Index/nopoints/limit/7/p/': 79,
}

# categories = {
#     # 'https://www.familyfeudquestions.com/Index/question_vote/limit/3/p/': 3
#     'https://www.familyfeudquestions.com/Index/fastmoney/p/': 2,
#     'https://www.familyfeudquestions.com/Index/nopoints/limit/7/p/': 2
# }


db = []

count = 0
try:
    for base, max_page in categories.items():
        for page_num in range(1, max_page+1):
            url = base + str(page_num)
            print(url)
            cache_path = f'data/tmp/page_{count}.json'
            if os.path.exists(cache_path):
                with open(cache_path, 'r') as f:
                    questions = json.load(f)[url]
            else:
                questions = parse_page(url)
                with open(cache_path, 'w') as f:
                    json.dump({url: questions}, f, indent=4)
                time.sleep(random())
            count += 1
            db.extend(questions)
except BaseException as e:
    print(e)
    import ipdb; ipdb.set_trace()

with open('data/question_db.json', 'w') as f:
    json.dump(db, f, indent=4)
