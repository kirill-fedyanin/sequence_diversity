
import requests
from bs4 import BeautifulSoup



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
            points = int(ans.i.text.split()[0])
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
        print(q)
        results.append(q)

    print(results)


parse_page('https://www.familyfeudquestions.com/Index/question_vote/limit/3/p/1')










#
# def single_question(url):
#     response = requests.get(url)
#     soup = BeautifulSoup(response.text, 'html.parser')
#     import ipdb; ipdb.set_trace()
#
#
# single_question('https://www.familyfeudquestions.com/Index/question_vote/limit/3')
#
#
#
