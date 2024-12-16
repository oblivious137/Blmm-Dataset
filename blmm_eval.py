from concurrent.futures import ThreadPoolExecutor
import json
import random
import re, os
import time

from openai import OpenAIError
from openai import OpenAI

client = OpenAI(base_url="xxxxx", api_key='xxxxx')

def openai_chat_completion(*args, force_refresh=True, **kwargs):
    '''cached openai chat completion
    :param args: args of openai_chat
    :param force_refresh: force refresh
    :param kwargs: kwargs of openai_chat
    :return: response message, usage
    '''
    key = (args, kwargs)
    str_key = str(key)
    response = client.chat.completions.create(*args, **kwargs)
    response, usage = response.choices[0].message.content, response.usage
    return response, usage

def openai_chat(text: str, model_name: str = "gpt-3.5-turbo-0613", **kwargs):
    '''openai chat model
    :param text: input message
    :param model_name: model name (gpt-3.5-turbo-0613, gpt-3.5-turbo-1106, gpt-4-1106-preview)
    :param kwargs: other parameters. (max_tokens, force_refresh)
    :return: response message
    '''
    return openai_chat_with_context([{"role": "user", "content": text}], model_name=model_name, **kwargs)


def openai_chat_with_context(messages, model_name: str = "gpt-3.5-turbo-0613", **kwargs):
    '''openai chat model with context
    :param messages: input messages (list of dict)
    :param model_name: model name (gpt-3.5-turbo-0613, gpt-3.5-turbo-1106, gpt-4-1106-preview)
    :param kwargs: other parameters. (max_tokens)
    :return: response message
    '''
    kwargs.pop('image_detail', None)
    parsed_messages = []
    for message in messages:
        parsed_message = {
            "role": message["role"],
            "content": [
                {"type": "text", "text": message["content"]},
            ],
        }
        parsed_messages.append(parsed_message)
    kwargs.setdefault("max_tokens", 400)
    response, usage = openai_chat_completion(
        model=model_name,
        messages=parsed_messages,
        **kwargs
    )
    from blmm import logger
    logger.debug("ChatGPT: "+str(usage))
    return response

def remove_illegal_escape(text):
    """Remove illegal escape characters"""
    cur = 0
    while cur < len(text):
        pos = text.find("\\", cur)
        if pos == -1:
            break
        if pos + 1 >= len(text) or text[pos + 1] not in ["\\", '"', "t", "n", "r", "b", "f", "u"]:
            text = text[:pos] + text[pos + 1:]
        cur = pos + 2
    return text.strip()

def parse_response(response, pattern=None):
    if pattern is None:
        pattern = (r"<Solution>((\S|\s)*)</Solution>",
                   r"<Solution>((\S|\s)*)$",
                   r"^((\S|\s)*)</Solution>",
                   r"({(\S|\s)*})")
    if isinstance(pattern, str):
        pattern = (pattern,)
    for p in pattern:
        matched_group = re.search(p, response)
        parsed_response = None
        if matched_group:
            parsed_response = matched_group.group(1)
            parsed_response = remove_illegal_escape(parsed_response)
            return parsed_response
    return remove_illegal_escape(response)

def chat_loop(chat_func, *args, max_retry=5, check_func=None, **kwargs):
    """Repeat querying until receiving a valid response
    :param chat_func: chat function
    :param args: arguments for chat function
    :param max_retry: maximum number of retry
    :param check_func: check function for the response (True: valid response, False: invalid response)
    """
    while max_retry > 0:
        try:
            # api failure is not counted as a retry
            sleep_time = 1
            while True:
                try:
                    response = chat_func(*args, **kwargs)
                    break
                except OpenAIError as e:
                    time.sleep(sleep_time)
                    sleep_time *= random.uniform(1.5, 1.7)
                    sleep_time = min(sleep_time, 5+random.uniform(0, 10))
            parsed_response = parse_response(response).strip()
            output = json.loads(parsed_response)
            if check_func is not None:
                assert check_func(output)
            if output:
                return output
        except Exception as e:
            max_retry -= 1
            print("failed response:", response)
            kwargs['force_refresh'] = True
            continue
    return None

def gpt_eval_training(BlmmAnswers=[], gt_sample_dict=[], num_runs=1):
	'''using gpt to eval prediction
	
	Parameters: list of dicts
		gt_sample_dict：[{
							"image": "VizWiz_train_00000000.jpg",
							"question": "What's the name of this product?",
							"answers": [
								{
									"answer_confidence": "yes",
									"answer": "basil leaves"
								},
								{
									"answer_confidence": "yes",
									"answer": "basil leaves"
								},
								...,
								{
									"answer_confidence": "yes",
									"answer": "basil"
								}
							],
							"answer_type": "other",
							"answerable": 1
						}]
	'''
	user_prompt = """You are a blind person and you are using an intelligent assistive system that is designed to take pictures and answer your questions. Now you need to compare the manual answer with the answer of the assistive system and give the assistive system a score. The scores are 0.0 (worst), 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9 or 1.0 (best).
You need to evaluate the correctness. Correctness usually requires reference to the manual answer to see if the assistive system has answered the question correctly and if there are any omissions or errors.
I'll give you some examples of scoring and the rationale for it, so you'll need to understand the logic behind the scoring of these examples and follow it.
The human answer is a list containing one or multiple values, each value is given by a human. When an answer is given by multiple people, it means that the answer is more credible and you should pay more attention to this type of answer.

Question | Human Answer | Assisted System Answer | Scoring Rationale | Correctness Score
--- | --- | --- | --- | ---
What's the name of this product? | ["basil leaves", "basil leaves", "basil", "basil", "basil leaves", "basil leaves", "basil leaves", "basil leaves", "basil leaves", "basil"] | basil leaves | The ten human answers all point to basil leaves and basil, and these two are synonyms. The auxiliary system gives one of the answers, so it should be given 1.0 points. | 1.0
Which one of these items is the children's dictionary? Is it the one on the right, or the one on the left? | ['left', 'left', 'left', 'left', 'left', 'left', 'left', 'right', 'left', 'left'] | the one on the left | There is one obviously wrong answer among the ten human answers, which can be ignored. The system answer left is correct, but it has a lot of redundancy and is not concise enough, so it is awarded 0.8 points. | 0.8
What is it? | ["juice", "doikham passion fruit juice", "passion fruit juice", "passion fruit juice", "passionfruit juice", "passion fruit juice", "juice", "passionfruit juice", "passion fruit juice", "food", "juice"] | passion fruit juice | The ten answers have similar semantics, but "passion fruit juice" has the most, so when it matches the most answers, it should be awarded 0.8 points, and when it matches other answers, it should be awarded 0.6 points. | 0.8
what is this? | ["vending machine", "soda machine", "soda vending machine", "vending machine", "vending machine", "vending machine", "vending machine", "vending machine", "vending machine", "vending machine"] | unanswerable | If the question is not answered, zero points should be given. | 0.0

Now, You need to evaluate the correctness of the following question and answer.
Question: {question}
Human Answer: {humanAnswer}
Assisted System Answer: {assistedAnswer}

Please respond in the following JSON format:

<Solution>{{"Thought": "thought", "Score": score}}</Solution>

The "Thought" field should explain your reasoning in up to 40 words, outlining why you gave the score you did. You should follow the logic of the examples above.

The "Score" field should be a number between 0.0 and 1.0, inclusive. It should be rounded to one decimal place. If you are unsure, you should give a score of 0.5.
"""
	
	assert len(BlmmAnswers) == len(gt_sample_dict)


	scores = []
	reasons = []
	for idx, answer in enumerate(gt_sample_dict):
		gt_answer_list = []
		for ans in answer["gt" if "gt" in answer else "answers"]:
			gt_answer_list.append(ans["answer"])
		question = answer["question"] if "question" in answer else answer["query"]
		assistedAnswer = BlmmAnswers[idx]["answer"]

		prompt = user_prompt.format(question=question, humanAnswer=gt_answer_list, assistedAnswer=assistedAnswer)
		avg_score = 0
		reason = list()
		for _ in range(num_runs):
			response = chat_loop(openai_chat, prompt, force_refresh=True, check_func=lambda x: "Thought" in x and "Score" in x and 0 <= float(x['Score']) <= 1)
			score = float(response['Score'])
			avg_score += score
			reason.append(response['Thought'])
		avg_score /= num_runs
		scores.append(avg_score)
		reasons.append(reason)
	
	return scores, reasons

def eval_gpt35(info, image, question, answer, return_reason=False, **kwargs):
    '''Evaluate the answer of the question. Use the gpt-3.5.
    :param info: the info of the sample
    :param image: the image
    :param question: the question
    :param answer: the answer
    :return: the similarity score
    '''
    try:
        wrapped_answer = [{
            "answer": answer,
        }]
        wrapped_info = [info]
        scores, reasons = gpt_eval_training(wrapped_answer, wrapped_info, **kwargs)
        if return_reason:
            return scores[0], reasons[0]
        else:
            return scores[0]
    except Exception as e:
        print("Exception in eval_gpt35:", e)
        return 0


if __name__ == "__main__":
    root_dir = "dataset"
    split = "blmm_val"
    ans_file = "blmm_val_answers.json"
    with open(os.path.join(root_dir, f"{split}.json")) as f:
        infos = json.load(f)
    
    total_answers = json.load(open(ans_file, "r"))

    # 评估，总共用10个线程处理所有数据，GPT-3.5
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = list()
        for info, answer in zip(infos, total_answers):
            futures.append(executor.submit(eval_gpt35, info, None, None, answer, return_reason=True, num_runs=5))
        result = list()
        scores = list()
        for future in futures:
            r = future.result()
            score, reason = r
            result.append({"score": score, "reason": reason})
            scores.append(score)

    print("score:", sum(scores) / len(scores))