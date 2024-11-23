import re

import json
import Levenshtein
from g2p_en import G2p

g2p = G2p()
requested_task = json.load(open('data/requested_task.json'))
requested_food = json.load(open('data/requested_food.json'))
state_commend = json.load(open('data/state_commend.json'))
TaskRequestIntentSamples = [
	"can you give me a recipe for {Food}",
	"what is the recipes for {Food}",
	"get a recipe for {Food}",
	"show me the recipe for {Food}",
	"can you get the recipes for a {Food}",
	"look for the ingredients for a {Food}",
	"tell a recipe for a {Food}",
	"what is the recipe for a {Food}",
	"can you get me the recipe for {Food}",
	"look for a recipe for {Food}",
	"show the recipe for a {Food}",
	"can you give me the ingredients for {Food}",
	"get the ingredients for a {Food}",
	"what is the recipe for {Food}",
	"give me a recipe for {Food}",
	"show the recipe for {Food}",
	"look for the recipe for {Food}",
	"can you get the recipes for {Food}",
	"tell me the ingredients for {Food}",
	"find me a recipe for {Food}",
	"find a recipe for a {Food}",
	"what are the ingredients for {Food}",
	"can you find me the recipes for {Food}",
	"tell the recipe for {Food}",
	"can you get me the recipes for {Food}",
	"look for the ingredients for {Food}",
	"can you look for the recipes for {Food}",
	"could you explain how to {Task}",
	"can you help {Task}",
	"what do you know about {Task}",
	"find the recipe for a {Food}",
	"can you give me the recipe for {Food}",
	"do you help me with {Task}",
	"do you have a recipe for a {Food}",
	"can you tell us about {Task}",
	"how can you get a {Food}",
	"what's the recipe for {Food}",
	"get the recipes for {Food}",
	"can you search for recipes for {Food}",
	"can you search for the recipe for {Food}",
	"can you look up the recipe for {Food}",
	"tell the recipes for {Food}",
	"can you find the recipes for {Food}",
	"give me the recipes for {Food}",
	"find me the recipes for {Food}",
	"would you explain how to {Task}",
	"can you get me how to {Task}",
	"can you get the recipe for {Food}",
	"get the recipe for {Food}",
	"can you tell me the recipes for {Food}",
	"search for the recipe for {Food}",
	"can you give us the recipe for {Food}",
	"can you look up the recipes for {Food}",
	"can you search for the recipes for {Food}",
	"do you have a recipes {Food}",
	"can you find us the recipes for {Food}",
	"look for the recipes for {Food}",
	"find us the recipes for {Food}",
	"tell how to {Task}",
	"can you tell me how to {Task}",
	"tell me how to {Task}",
	"can you get how to {Task}",
	"can you show me how to {Task}",

	"get how to {Task}",
	"get me a {Food}",
	"i want to know how to {Task}",
	"how can you get an {Food}",
	"can you look for recipes for {Food}",
	"give me the recipe for {Food}",
	"can you find recipes for {Food}",
	"search for recipes for {Food}",
	"find the recipes for {Food}",
	"tell me about {Task}",
	"search recipe {Food}",
	"search recipes for {Food}",
	"find recipes for {Food}",
	"can you explain how to {Task}",
	"show me how to {Task}",
	"what do i need to have to make a {Food}",
	"what ingredients do i need to make a {Food}",

	"how do you make a {Food}",
	"how do you make an {Food}",
	"how do you make {Food}",
	"how do you {Task}",
	"search for {Food} recipes",
	"search for {Food} recipe",

	"could you help me with {Task}",
	"would you help {Task}",
	"assist me with {Task}",
	"assist me to {Task}",
	"how to {Task}",
	"i want to {Task}",
	"i need to {Task}",

	"recipes for a {Food}",
	"recipe for a {Food}",
	"recipes for an {Food}",
	"recipe for an {Food}",
	"recipes for the {Food}",
	"recipe for the {Food}",
	"recipes for {Food}",
	"recipe for {Food}",
	"recipe {Food}",
	"ingredients for a {Food}",
	"ingredients for {Food}",
]


def clean_hypotheses(hypotheses, for_cls=False):
	noise = ['actually', 'hmm', 'oh', 'uh', 'uhhh', 'well', "please", "alexa", "echo"]
	if for_cls:
		noise.extend(['a', 'an', 'my', 'your', 'the'])
	for entry in hypotheses:
		tokens_to_keep = []
		for token in entry['tokens']:
			if token['value'] not in noise:
				token['value'] = re.sub(r'\bd\.*\s*i\.*\s*y\.*', 'diy', token['value'])
				tokens_to_keep.append(token)
		entry['tokens'] = tokens_to_keep
	return hypotheses


required_context = ['asr', 'taco_state']


def get_required_context():
	return required_context


def extract_slots(input_text):
	slots = {}
	for template in TaskRequestIntentSamples:
		pattern = re.sub(r'{(\w+)}', r'(.+)', template)
		match = re.search(pattern, input_text)
		if match:
			slot_values = match.groups()
			slot_names = re.findall(r'{(\w+)}', template)
			pattern = input_text
			for name, value in zip(slot_names, slot_values):
				slots[name] = value.strip()
				pattern = pattern.replace(value, "{" + name + "}")
			slots["template"] = pattern
			break
	return slots


def sounds_most_like(utterance, base, accuracy):
	if utterance in base:
		result = utterance
	else:
		result = None
		phoneme = " ".join(g2p(utterance))
		max_similarity = accuracy
		for k, v in base.items():
			d = Levenshtein.distance(phoneme, v)
			similarity = 1 - d / (max(len(v), len(phoneme)))
			if similarity > max_similarity:
				max_similarity = similarity
				result = k
	return result


def proposedTask_recovery(utterance, proposed_tasks):
	task_base = {}
	food_base = {}
	for k in proposed_tasks:
		key = k["title"]
		if key.startswith("How to "):
			key = key.split("How to ", 1)[-1].lower()
			if key in requested_task:
				task_base[key] = requested_task[key]
			else:
				task_base[key] = " ".join(g2p(key))
		else:
			if key in requested_food:
				food_base[key] = requested_food[key]
			else:
				food_base[key] = " ".join(g2p(key))
	return capture_recovery(utterance, {"commend_base": {}, "task_base": task_base, "food_base": food_base}, 0.75)


def state_base(state):
	commend_base = {}
	task_base = {}
	food_base = {}
	for k, v in state_commend.items():
		if k in state:
			commend_base.update(v)
	if "Welcome" in state or "TaskChoice" in state or "TaskPreparation" in state:
		task_base = requested_task
		food_base = requested_food
	return {"commend_base": commend_base, "task_base": task_base, "food_base": food_base}


def welcome_recovery(utterance, welcome_task, welcome_recipe):
	task_base = {}
	food_base = {}
	if welcome_task in requested_task:
		task_base[welcome_task] = requested_task[welcome_task]
	else:
		task_base[welcome_task] = " ".join(g2p(welcome_task))
	if welcome_recipe in requested_food:
		food_base[welcome_recipe] = requested_food[welcome_recipe]
	else:
		food_base[welcome_recipe] = " ".join(g2p(welcome_recipe))
	return capture_recovery(utterance, {"commend_base": {}, "task_base": task_base, "food_base": food_base}, 0.75)


def capture_recovery(utterance, base, accuracy=0.8):
	extracted_slots = extract_slots(utterance)
	recovery = None
	if 'Task' in extracted_slots:
		extracted_slots['Task'] = sounds_most_like(extracted_slots['Task'], base["task_base"], accuracy)
		if extracted_slots['Task']:
			recovery = extracted_slots['template'].format(**extracted_slots)
	elif 'Food' in extracted_slots:
		extracted_slots['Food'] = sounds_most_like(extracted_slots['Food'], base["food_base"], accuracy)
		if extracted_slots['Food']:
			recovery = extracted_slots['template'].format(**extracted_slots)
	else:
		recovery = sounds_most_like(utterance, {**base["commend_base"], **base["task_base"], **base["food_base"]}, accuracy)
	return recovery


def recovery_process(asr, base):
	for i in asr:
		if 'confidence' in i:
			recovery = None
			utterance = ' '.join(word['value'] for word in i['tokens'])
			if i["confidence"] > 0.9:
				recovery = utterance
			elif i["confidence"] > 0.4:
				recovery = capture_recovery(utterance, base)
			i["recovery"] = recovery
	return getText(asr)


def changed_confidence(recovery, tokens, debug=False):
	h = [word['value'] for word in tokens]
	r = recovery.split()

	costs = [[0 for inner in range(len(h) + 1)] for outer in range(len(r) + 1)]
	backtrace = [[0 for inner in range(len(h) + 1)] for outer in range(len(r) + 1)]

	OP_OK = 0
	OP_SUB = 1
	OP_INS = 2
	OP_DEL = 3

	DEL_PENALTY = 1
	INS_PENALTY = 1
	SUB_PENALTY = 1

	for i in range(1, len(r) + 1):
		costs[i][0] = DEL_PENALTY * i
		backtrace[i][0] = OP_DEL

	for j in range(1, len(h) + 1):
		costs[0][j] = INS_PENALTY * j
		backtrace[0][j] = OP_INS

	for i in range(1, len(r) + 1):
		for j in range(1, len(h) + 1):
			if r[i - 1] == h[j - 1]:
				costs[i][j] = costs[i - 1][j - 1]
				backtrace[i][j] = OP_OK
			else:
				substitutionCost = costs[i - 1][j - 1] + SUB_PENALTY
				insertionCost = costs[i][j - 1] + INS_PENALTY
				deletionCost = costs[i - 1][j] + DEL_PENALTY

				costs[i][j] = min(substitutionCost, insertionCost, deletionCost)
				if costs[i][j] == substitutionCost:
					backtrace[i][j] = OP_SUB
				elif costs[i][j] == insertionCost:
					backtrace[i][j] = OP_INS
				else:
					backtrace[i][j] = OP_DEL

	i = len(r)
	j = len(h)
	numSub = 0
	numDel = 0
	numIns = 0
	numCor = 0
	word_changes = []
	avg_confidence = sum(word['confidence'] for word in tokens) / len(tokens)
	change_confidence = 0
	if debug:
		print(avg_confidence)
		print("OP\tREF\tHYP")

	while i > 0 or j > 0:
		if backtrace[i][j] == OP_OK:
			numCor += 1
			i -= 1
			j -= 1
			if debug:
				print("OK\t" + r[i] + "\t" + h[j])
				print(h[j], tokens[j]["confidence"])
			change_confidence += tokens[j]["confidence"]
		elif backtrace[i][j] == OP_SUB:
			numSub += 1
			i -= 1
			j -= 1
			if debug:
				print("SUB\t" + r[i] + "\t" + h[j])
			change_confidence += avg_confidence
		elif backtrace[i][j] == OP_INS:
			numIns += 1
			j -= 1
			if debug:
				print("INS\t" + "****" + "\t" + h[j])
				print(h[j], tokens[j]["confidence"])
		# change_confidence += avg_confidence

		elif backtrace[i][j] == OP_DEL:
			numDel += 1
			i -= 1
			if debug:
				print("DEL\t" + r[i] + "\t" + "****")
				print("****", tokens[j]["confidence"])
			# s = 0
			# if j > 0:
			# 	s += tokens[j-1]["confidence"]/2
			#
			# 	print(tokens[j-1]["value"])
			# if j < len(tokens):
			# 	s += tokens[j]["confidence"]/2
			# 	print(tokens[j]["value"])
			change_confidence += avg_confidence

	if debug:
		print("Ncor " + str(numCor))
		print("Nsub " + str(numSub))
		print("Ndel " + str(numDel))
		print("Nins " + str(numIns))

	return change_confidence / len(r) - avg_confidence


def getText(hypotheses):
	max_confidence = hypotheses[0]["confidence"]
	result = ' '.join(word['value'] for word in hypotheses[0]['tokens'])
	for i in hypotheses:
		utterance = ' '.join(word['value'] for word in i['tokens'])
		if 'recovery' in i and i["recovery"] and i["recovery"] != utterance:
			changed = changed_confidence(i["recovery"], i['tokens'])
			confidence = i["confidence"] + changed
			# print(i["recovery"], i["confidence"],changed)
			if confidence > max_confidence:
				max_confidence = confidence
				result = i["recovery"]
	return result


def handle_message(msg):
	asr = clean_hypotheses(msg['asr'])
	taco_state = msg['taco_state']
	utterance = ' '.join(word['value'] for word in asr[0]['tokens'])
	new_utt = None
	if 'Welcome' in taco_state:
		welcome_task = msg.get('welcome_task', None)
		welcome_recipe = msg.get('welcome_recipe', None)
		if welcome_task and welcome_recipe:
			new_utt = welcome_recovery(utterance, welcome_task, welcome_recipe)
	elif 'TaskCatalog' in taco_state:
		proposed_tasks = msg.get('proposed_tasks', None)
		if proposed_tasks:
			new_utt = proposedTask_recovery(utterance, proposed_tasks)
	if not new_utt:
		base = state_base(taco_state)
		new_utt = recovery_process(asr, base)
	return {'correction': new_utt}


if __name__ == '__main__':
	# speechRecognition = {
	# 	"matcha ice cream":[
	# 			{
	# 				"tokens": [
	# 					{
	# 						"value": "recipes",
	# 						"confidence": 0.542,
	# 						"_confidenceV2": {
	# 							"score": 0.542
	# 						},
	# 						"startOffsetInMilliseconds": 480,
	# 						"endOffsetInMilliseconds": 1030
	# 					},
	# 					{
	# 						"value": "for",
	# 						"confidence": 0.65,
	# 						"_confidenceV2": {
	# 							"score": 0.65
	# 						},
	# 						"startOffsetInMilliseconds": 1170,
	# 						"endOffsetInMilliseconds": 1450
	# 					},
	# 					{
	# 						"value": "marshall",
	# 						"confidence": 0.224,
	# 						"_confidenceV2": {
	# 							"score": 0.224
	# 						},
	# 						"startOffsetInMilliseconds": 1470,
	# 						"endOffsetInMilliseconds": 2050
	# 					},
	# 					{
	# 						"value": "ice",
	# 						"confidence": 0.465,
	# 						"_confidenceV2": {
	# 							"score": 0.465
	# 						},
	# 						"startOffsetInMilliseconds": 2070,
	# 						"endOffsetInMilliseconds": 2290
	# 					},
	# 					{
	# 						"value": "queen",
	# 						"confidence": 0.438,
	# 						"_confidenceV2": {
	# 							"score": 0.438
	# 						},
	# 						"startOffsetInMilliseconds": 2310,
	# 						"endOffsetInMilliseconds": 2800
	# 					}
	# 				],
	# 				"confidence": 0.605,
	# 				"_confidenceV2": {
	# 					"score": 0.605
	# 				},
	# 				"directedness": {
	# 					"score": 0
	# 				}
	# 			},
	# 			{
	# 				"tokens": [
	# 					{
	# 						"value": "recipes",
	# 						"confidence": 0.542,
	# 						"_confidenceV2": {
	# 							"score": 0.542
	# 						},
	# 						"startOffsetInMilliseconds": 480,
	# 						"endOffsetInMilliseconds": 1030
	# 					},
	# 					{
	# 						"value": "for",
	# 						"confidence": 0.65,
	# 						"_confidenceV2": {
	# 							"score": 0.65
	# 						},
	# 						"startOffsetInMilliseconds": 1170,
	# 						"endOffsetInMilliseconds": 1450
	# 					},
	# 					{
	# 						"value": "martial",
	# 						"confidence": 0.128,
	# 						"_confidenceV2": {
	# 							"score": 0.128
	# 						},
	# 						"startOffsetInMilliseconds": 1470,
	# 						"endOffsetInMilliseconds": 2050
	# 					},
	# 					{
	# 						"value": "ice",
	# 						"confidence": 0.465,
	# 						"_confidenceV2": {
	# 							"score": 0.465
	# 						},
	# 						"startOffsetInMilliseconds": 2070,
	# 						"endOffsetInMilliseconds": 2290
	# 					},
	# 					{
	# 						"value": "queen",
	# 						"confidence": 0.438,
	# 						"_confidenceV2": {
	# 							"score": 0.438
	# 						},
	# 						"startOffsetInMilliseconds": 2310,
	# 						"endOffsetInMilliseconds": 2800
	# 					}
	# 				],
	# 				"confidence": 0.605,
	# 				"_confidenceV2": {
	# 					"score": 0.605
	# 				},
	# 				"directedness": {
	# 					"score": 0
	# 				}
	# 			},
	# 			{
	# 				"tokens": [
	# 					{
	# 						"value": "recipes",
	# 						"confidence": 0.542,
	# 						"_confidenceV2": {
	# 							"score": 0.542
	# 						},
	# 						"startOffsetInMilliseconds": 480,
	# 						"endOffsetInMilliseconds": 1030
	# 					},
	# 					{
	# 						"value": "for",
	# 						"confidence": 0.65,
	# 						"_confidenceV2": {
	# 							"score": 0.65
	# 						},
	# 						"startOffsetInMilliseconds": 1170,
	# 						"endOffsetInMilliseconds": 1450
	# 					},
	# 					{
	# 						"value": "marshal",
	# 						"confidence": 0.24,
	# 						"_confidenceV2": {
	# 							"score": 0.24
	# 						},
	# 						"startOffsetInMilliseconds": 1470,
	# 						"endOffsetInMilliseconds": 2050
	# 					},
	# 					{
	# 						"value": "ice",
	# 						"confidence": 0.465,
	# 						"_confidenceV2": {
	# 							"score": 0.465
	# 						},
	# 						"startOffsetInMilliseconds": 2070,
	# 						"endOffsetInMilliseconds": 2290
	# 					},
	# 					{
	# 						"value": "queen",
	# 						"confidence": 0.438,
	# 						"_confidenceV2": {
	# 							"score": 0.438
	# 						},
	# 						"startOffsetInMilliseconds": 2310,
	# 						"endOffsetInMilliseconds": 2800
	# 					}
	# 				],
	# 				"confidence": 0.605,
	# 				"_confidenceV2": {
	# 					"score": 0.605
	# 				},
	# 				"directedness": {
	# 					"score": 0
	# 				}
	# 			},
	# 			{
	# 				"tokens": [
	# 					{
	# 						"value": "recipes",
	# 						"confidence": 0.542,
	# 						"_confidenceV2": {
	# 							"score": 0.542
	# 						},
	# 						"startOffsetInMilliseconds": 480,
	# 						"endOffsetInMilliseconds": 1030
	# 					},
	# 					{
	# 						"value": "for",
	# 						"confidence": 0.65,
	# 						"_confidenceV2": {
	# 							"score": 0.65
	# 						},
	# 						"startOffsetInMilliseconds": 1170,
	# 						"endOffsetInMilliseconds": 1450
	# 					},
	# 					{
	# 						"value": "marshall",
	# 						"confidence": 0.224,
	# 						"_confidenceV2": {
	# 							"score": 0.224
	# 						},
	# 						"startOffsetInMilliseconds": 1470,
	# 						"endOffsetInMilliseconds": 2050
	# 					},
	# 					{
	# 						"value": "ice",
	# 						"confidence": 0.465,
	# 						"_confidenceV2": {
	# 							"score": 0.465
	# 						},
	# 						"startOffsetInMilliseconds": 2070,
	# 						"endOffsetInMilliseconds": 2290
	# 					},
	# 					{
	# 						"value": "cream",
	# 						"confidence": 0.482,
	# 						"_confidenceV2": {
	# 							"score": 0.482
	# 						},
	# 						"startOffsetInMilliseconds": 2310,
	# 						"endOffsetInMilliseconds": 2800
	# 					}
	# 				],
	# 				"confidence": 0.606,
	# 				"_confidenceV2": {
	# 					"score": 0.606
	# 				},
	# 				"directedness": {
	# 					"score": 0
	# 				}
	# 			},
	# 			{
	# 				"tokens": [
	# 					{
	# 						"value": "recipe",
	# 						"confidence": 0.537,
	# 						"_confidenceV2": {
	# 							"score": 0.537
	# 						},
	# 						"startOffsetInMilliseconds": 480,
	# 						"endOffsetInMilliseconds": 1120
	# 					},
	# 					{
	# 						"value": "for",
	# 						"confidence": 0.65,
	# 						"_confidenceV2": {
	# 							"score": 0.65
	# 						},
	# 						"startOffsetInMilliseconds": 1170,
	# 						"endOffsetInMilliseconds": 1450
	# 					},
	# 					{
	# 						"value": "marshall",
	# 						"confidence": 0.224,
	# 						"_confidenceV2": {
	# 							"score": 0.224
	# 						},
	# 						"startOffsetInMilliseconds": 1470,
	# 						"endOffsetInMilliseconds": 2050
	# 					},
	# 					{
	# 						"value": "ice",
	# 						"confidence": 0.465,
	# 						"_confidenceV2": {
	# 							"score": 0.465
	# 						},
	# 						"startOffsetInMilliseconds": 2070,
	# 						"endOffsetInMilliseconds": 2290
	# 					},
	# 					{
	# 						"value": "queen",
	# 						"confidence": 0.438,
	# 						"_confidenceV2": {
	# 							"score": 0.438
	# 						},
	# 						"startOffsetInMilliseconds": 2310,
	# 						"endOffsetInMilliseconds": 2800
	# 					}
	# 				],
	# 				"confidence": 0.605,
	# 				"_confidenceV2": {
	# 					"score": 0.605
	# 				},
	# 				"directedness": {
	# 					"score": 0
	# 				}
	# 			}
	# 		],
	# 	"bbq chicken wings":[
	# 			{
	# 				"tokens": [
	# 					{
	# 						"value": "barbecue",
	# 						"confidence": 0.608,
	# 						"_confidenceV2": {
	# 							"score": 0.608
	# 						},
	# 						"startOffsetInMilliseconds": 420,
	# 						"endOffsetInMilliseconds": 1030
	# 					},
	# 					{
	# 						"value": "chicken",
	# 						"confidence": 0.76,
	# 						"_confidenceV2": {
	# 							"score": 0.76
	# 						},
	# 						"startOffsetInMilliseconds": 1050,
	# 						"endOffsetInMilliseconds": 1450
	# 					},
	# 					{
	# 						"value": "wings",
	# 						"confidence": 0.804,
	# 						"_confidenceV2": {
	# 							"score": 0.804
	# 						},
	# 						"startOffsetInMilliseconds": 1470,
	# 						"endOffsetInMilliseconds": 2020
	# 					}
	# 				],
	# 				"confidence": 0.899,
	# 				"_confidenceV2": {
	# 					"score": 0.899
	# 				},
	# 				"directedness": {
	# 					"score": 0
	# 				}
	# 			},
	# 			{
	# 				"tokens": [
	# 					{
	# 						"value": "bbq",
	# 						"confidence": 0.12,
	# 						"_confidenceV2": {
	# 							"score": 0.12
	# 						},
	# 						"startOffsetInMilliseconds": 420,
	# 						"endOffsetInMilliseconds": 1030
	# 					},
	# 					{
	# 						"value": "chicken",
	# 						"confidence": 0.76,
	# 						"_confidenceV2": {
	# 							"score": 0.76
	# 						},
	# 						"startOffsetInMilliseconds": 1050,
	# 						"endOffsetInMilliseconds": 1450
	# 					},
	# 					{
	# 						"value": "wings",
	# 						"confidence": 0.804,
	# 						"_confidenceV2": {
	# 							"score": 0.804
	# 						},
	# 						"startOffsetInMilliseconds": 1470,
	# 						"endOffsetInMilliseconds": 2020
	# 					}
	# 				],
	# 				"confidence": 0.86,
	# 				"_confidenceV2": {
	# 					"score": 0.86
	# 				},
	# 				"directedness": {
	# 					"score": 0
	# 				}
	# 			},
	# 			{
	# 				"tokens": [
	# 					{
	# 						"value": "barbecued",
	# 						"confidence": 0.357,
	# 						"_confidenceV2": {
	# 							"score": 0.357
	# 						},
	# 						"startOffsetInMilliseconds": 420,
	# 						"endOffsetInMilliseconds": 1030
	# 					},
	# 					{
	# 						"value": "chicken",
	# 						"confidence": 0.76,
	# 						"_confidenceV2": {
	# 							"score": 0.76
	# 						},
	# 						"startOffsetInMilliseconds": 1050,
	# 						"endOffsetInMilliseconds": 1450
	# 					},
	# 					{
	# 						"value": "wings",
	# 						"confidence": 0.804,
	# 						"_confidenceV2": {
	# 							"score": 0.804
	# 						},
	# 						"startOffsetInMilliseconds": 1470,
	# 						"endOffsetInMilliseconds": 2020
	# 					}
	# 				],
	# 				"confidence": 0.883,
	# 				"_confidenceV2": {
	# 					"score": 0.883
	# 				},
	# 				"directedness": {
	# 					"score": 0
	# 				}
	# 			},
	# 			{
	# 				"tokens": [
	# 					{
	# 						"value": "barbeque",
	# 						"confidence": 0.247,
	# 						"_confidenceV2": {
	# 							"score": 0.247
	# 						},
	# 						"startOffsetInMilliseconds": 420,
	# 						"endOffsetInMilliseconds": 1030
	# 					},
	# 					{
	# 						"value": "chicken",
	# 						"confidence": 0.76,
	# 						"_confidenceV2": {
	# 							"score": 0.76
	# 						},
	# 						"startOffsetInMilliseconds": 1050,
	# 						"endOffsetInMilliseconds": 1450
	# 					},
	# 					{
	# 						"value": "wings",
	# 						"confidence": 0.804,
	# 						"_confidenceV2": {
	# 							"score": 0.804
	# 						},
	# 						"startOffsetInMilliseconds": 1470,
	# 						"endOffsetInMilliseconds": 2020
	# 					}
	# 				],
	# 				"confidence": 0.873,
	# 				"_confidenceV2": {
	# 					"score": 0.873
	# 				},
	# 				"directedness": {
	# 					"score": 0
	# 				}
	# 			}
	# 		],
	# 	"make a healthy snack for teens":[
	# 			{
	# 				"tokens": [
	# 					{
	# 						"value": "how",
	# 						"confidence": 0.943,
	# 						"_confidenceV2": {
	# 							"score": 0.943
	# 						},
	# 						"startOffsetInMilliseconds": 90,
	# 						"endOffsetInMilliseconds": 340
	# 					},
	# 					{
	# 						"value": "to",
	# 						"confidence": 0.911,
	# 						"_confidenceV2": {
	# 							"score": 0.911
	# 						},
	# 						"startOffsetInMilliseconds": 360,
	# 						"endOffsetInMilliseconds": 490
	# 					},
	# 					{
	# 						"value": "make",
	# 						"confidence": 0.91,
	# 						"_confidenceV2": {
	# 							"score": 0.91
	# 						},
	# 						"startOffsetInMilliseconds": 510,
	# 						"endOffsetInMilliseconds": 760
	# 					},
	# 					{
	# 						"value": "a",
	# 						"confidence": 0.756,
	# 						"_confidenceV2": {
	# 							"score": 0.756
	# 						},
	# 						"startOffsetInMilliseconds": 780,
	# 						"endOffsetInMilliseconds": 850
	# 					},
	# 					{
	# 						"value": "healthy",
	# 						"confidence": 0.45,
	# 						"_confidenceV2": {
	# 							"score": 0.45
	# 						},
	# 						"startOffsetInMilliseconds": 870,
	# 						"endOffsetInMilliseconds": 1360
	# 					},
	# 					{
	# 						"value": "snack",
	# 						"confidence": 0.288,
	# 						"_confidenceV2": {
	# 							"score": 0.288
	# 						},
	# 						"startOffsetInMilliseconds": 1380,
	# 						"endOffsetInMilliseconds": 1780
	# 					},
	# 					{
	# 						"value": "or",
	# 						"confidence": 0.119,
	# 						"_confidenceV2": {
	# 							"score": 0.119
	# 						},
	# 						"startOffsetInMilliseconds": 1800,
	# 						"endOffsetInMilliseconds": 1870
	# 					},
	# 					{
	# 						"value": "fourteens",
	# 						"confidence": 0.34,
	# 						"_confidenceV2": {
	# 							"score": 0.34
	# 						},
	# 						"startOffsetInMilliseconds": 1890,
	# 						"endOffsetInMilliseconds": 2710
	# 					}
	# 				],
	# 				"confidence": 0.439,
	# 				"_confidenceV2": {
	# 					"score": 0.439
	# 				},
	# 				"directedness": {
	# 					"score": 0
	# 				}
	# 			},
	# 			{
	# 				"tokens": [
	# 					{
	# 						"value": "how",
	# 						"confidence": 0.943,
	# 						"_confidenceV2": {
	# 							"score": 0.943
	# 						},
	# 						"startOffsetInMilliseconds": 90,
	# 						"endOffsetInMilliseconds": 340
	# 					},
	# 					{
	# 						"value": "to",
	# 						"confidence": 0.911,
	# 						"_confidenceV2": {
	# 							"score": 0.911
	# 						},
	# 						"startOffsetInMilliseconds": 360,
	# 						"endOffsetInMilliseconds": 490
	# 					},
	# 					{
	# 						"value": "make",
	# 						"confidence": 0.91,
	# 						"_confidenceV2": {
	# 							"score": 0.91
	# 						},
	# 						"startOffsetInMilliseconds": 510,
	# 						"endOffsetInMilliseconds": 760
	# 					},
	# 					{
	# 						"value": "a",
	# 						"confidence": 0.756,
	# 						"_confidenceV2": {
	# 							"score": 0.756
	# 						},
	# 						"startOffsetInMilliseconds": 780,
	# 						"endOffsetInMilliseconds": 850
	# 					},
	# 					{
	# 						"value": "healthy",
	# 						"confidence": 0.45,
	# 						"_confidenceV2": {
	# 							"score": 0.45
	# 						},
	# 						"startOffsetInMilliseconds": 870,
	# 						"endOffsetInMilliseconds": 1360
	# 					},
	# 					{
	# 						"value": "snack",
	# 						"confidence": 0.288,
	# 						"_confidenceV2": {
	# 							"score": 0.288
	# 						},
	# 						"startOffsetInMilliseconds": 1380,
	# 						"endOffsetInMilliseconds": 1780
	# 					},
	# 					{
	# 						"value": "are",
	# 						"confidence": 0.23,
	# 						"_confidenceV2": {
	# 							"score": 0.23
	# 						},
	# 						"startOffsetInMilliseconds": 1800,
	# 						"endOffsetInMilliseconds": 1870
	# 					},
	# 					{
	# 						"value": "fourteens",
	# 						"confidence": 0.34,
	# 						"_confidenceV2": {
	# 							"score": 0.34
	# 						},
	# 						"startOffsetInMilliseconds": 1890,
	# 						"endOffsetInMilliseconds": 2710
	# 					}
	# 				],
	# 				"confidence": 0.439,
	# 				"_confidenceV2": {
	# 					"score": 0.439
	# 				},
	# 				"directedness": {
	# 					"score": 0
	# 				}
	# 			},
	# 			{
	# 				"tokens": [
	# 					{
	# 						"value": "how",
	# 						"confidence": 0.943,
	# 						"_confidenceV2": {
	# 							"score": 0.943
	# 						},
	# 						"startOffsetInMilliseconds": 90,
	# 						"endOffsetInMilliseconds": 340
	# 					},
	# 					{
	# 						"value": "to",
	# 						"confidence": 0.911,
	# 						"_confidenceV2": {
	# 							"score": 0.911
	# 						},
	# 						"startOffsetInMilliseconds": 360,
	# 						"endOffsetInMilliseconds": 490
	# 					},
	# 					{
	# 						"value": "make",
	# 						"confidence": 0.91,
	# 						"_confidenceV2": {
	# 							"score": 0.91
	# 						},
	# 						"startOffsetInMilliseconds": 510,
	# 						"endOffsetInMilliseconds": 760
	# 					},
	# 					{
	# 						"value": "a",
	# 						"confidence": 0.756,
	# 						"_confidenceV2": {
	# 							"score": 0.756
	# 						},
	# 						"startOffsetInMilliseconds": 780,
	# 						"endOffsetInMilliseconds": 850
	# 					},
	# 					{
	# 						"value": "healthy",
	# 						"confidence": 0.45,
	# 						"_confidenceV2": {
	# 							"score": 0.45
	# 						},
	# 						"startOffsetInMilliseconds": 870,
	# 						"endOffsetInMilliseconds": 1360
	# 					},
	# 					{
	# 						"value": "snack",
	# 						"confidence": 0.288,
	# 						"_confidenceV2": {
	# 							"score": 0.288
	# 						},
	# 						"startOffsetInMilliseconds": 1380,
	# 						"endOffsetInMilliseconds": 1780
	# 					},
	# 					{
	# 						"value": "a",
	# 						"confidence": 0.18,
	# 						"_confidenceV2": {
	# 							"score": 0.18
	# 						},
	# 						"startOffsetInMilliseconds": 1800,
	# 						"endOffsetInMilliseconds": 1870
	# 					},
	# 					{
	# 						"value": "fourteens",
	# 						"confidence": 0.34,
	# 						"_confidenceV2": {
	# 							"score": 0.34
	# 						},
	# 						"startOffsetInMilliseconds": 1890,
	# 						"endOffsetInMilliseconds": 2710
	# 					}
	# 				],
	# 				"confidence": 0.439,
	# 				"_confidenceV2": {
	# 					"score": 0.439
	# 				},
	# 				"directedness": {
	# 					"score": 0
	# 				}
	# 			},
	# 			{
	# 				"tokens": [
	# 					{
	# 						"value": "how",
	# 						"confidence": 0.943,
	# 						"_confidenceV2": {
	# 							"score": 0.943
	# 						},
	# 						"startOffsetInMilliseconds": 90,
	# 						"endOffsetInMilliseconds": 340
	# 					},
	# 					{
	# 						"value": "to",
	# 						"confidence": 0.911,
	# 						"_confidenceV2": {
	# 							"score": 0.911
	# 						},
	# 						"startOffsetInMilliseconds": 360,
	# 						"endOffsetInMilliseconds": 490
	# 					},
	# 					{
	# 						"value": "make",
	# 						"confidence": 0.91,
	# 						"_confidenceV2": {
	# 							"score": 0.91
	# 						},
	# 						"startOffsetInMilliseconds": 510,
	# 						"endOffsetInMilliseconds": 760
	# 					},
	# 					{
	# 						"value": "a",
	# 						"confidence": 0.756,
	# 						"_confidenceV2": {
	# 							"score": 0.756
	# 						},
	# 						"startOffsetInMilliseconds": 780,
	# 						"endOffsetInMilliseconds": 850
	# 					},
	# 					{
	# 						"value": "healthy",
	# 						"confidence": 0.45,
	# 						"_confidenceV2": {
	# 							"score": 0.45
	# 						},
	# 						"startOffsetInMilliseconds": 870,
	# 						"endOffsetInMilliseconds": 1360
	# 					},
	# 					{
	# 						"value": "snack",
	# 						"confidence": 0.288,
	# 						"_confidenceV2": {
	# 							"score": 0.288
	# 						},
	# 						"startOffsetInMilliseconds": 1380,
	# 						"endOffsetInMilliseconds": 1750
	# 					},
	# 					{
	# 						"value": "fourteens",
	# 						"confidence": 0.34,
	# 						"_confidenceV2": {
	# 							"score": 0.34
	# 						},
	# 						"startOffsetInMilliseconds": 1770,
	# 						"endOffsetInMilliseconds": 2710
	# 					}
	# 				],
	# 				"confidence": 0.44,
	# 				"_confidenceV2": {
	# 					"score": 0.44
	# 				},
	# 				"directedness": {
	# 					"score": 0
	# 				}
	# 			},
	# 			{
	# 				"tokens": [
	# 					{
	# 						"value": "how",
	# 						"confidence": 0.943,
	# 						"_confidenceV2": {
	# 							"score": 0.943
	# 						},
	# 						"startOffsetInMilliseconds": 90,
	# 						"endOffsetInMilliseconds": 340
	# 					},
	# 					{
	# 						"value": "to",
	# 						"confidence": 0.911,
	# 						"_confidenceV2": {
	# 							"score": 0.911
	# 						},
	# 						"startOffsetInMilliseconds": 360,
	# 						"endOffsetInMilliseconds": 490
	# 					},
	# 					{
	# 						"value": "make",
	# 						"confidence": 0.91,
	# 						"_confidenceV2": {
	# 							"score": 0.91
	# 						},
	# 						"startOffsetInMilliseconds": 510,
	# 						"endOffsetInMilliseconds": 760
	# 					},
	# 					{
	# 						"value": "a",
	# 						"confidence": 0.756,
	# 						"_confidenceV2": {
	# 							"score": 0.756
	# 						},
	# 						"startOffsetInMilliseconds": 780,
	# 						"endOffsetInMilliseconds": 850
	# 					},
	# 					{
	# 						"value": "healthy",
	# 						"confidence": 0.45,
	# 						"_confidenceV2": {
	# 							"score": 0.45
	# 						},
	# 						"startOffsetInMilliseconds": 870,
	# 						"endOffsetInMilliseconds": 1360
	# 					},
	# 					{
	# 						"value": "snake",
	# 						"confidence": 0.161,
	# 						"_confidenceV2": {
	# 							"score": 0.161
	# 						},
	# 						"startOffsetInMilliseconds": 1380,
	# 						"endOffsetInMilliseconds": 1780
	# 					},
	# 					{
	# 						"value": "or",
	# 						"confidence": 0.119,
	# 						"_confidenceV2": {
	# 							"score": 0.119
	# 						},
	# 						"startOffsetInMilliseconds": 1800,
	# 						"endOffsetInMilliseconds": 1870
	# 					},
	# 					{
	# 						"value": "fourteens",
	# 						"confidence": 0.34,
	# 						"_confidenceV2": {
	# 							"score": 0.34
	# 						},
	# 						"startOffsetInMilliseconds": 1890,
	# 						"endOffsetInMilliseconds": 2710
	# 					}
	# 				],
	# 				"confidence": 0.439,
	# 				"_confidenceV2": {
	# 					"score": 0.439
	# 				},
	# 				"directedness": {
	# 					"score": 0
	# 				}
	# 			}
	# 		],
	# }
	# for k, v in speechRecognition.items():
	# 	print(k,":",recovery_process(v))

	msg = {
		"taco_state": "TaskChoice",
		"asr": [
			   {
				   "tokens": [
					   {
						   "value": "recipes",
						   "confidence": 0.63,
						   "_confidenceV2": {
							   "score": 0.63
						   },
						   "startOffsetInMilliseconds": 210,
						   "endOffsetInMilliseconds": 880
					   },
					   {
						   "value": "for",
						   "confidence": 0.773,
						   "_confidenceV2": {
							   "score": 0.773
						   },
						   "startOffsetInMilliseconds": 900,
						   "endOffsetInMilliseconds": 1090
					   },
					   {
						   "value": "pumping",
						   "confidence": 0.296,
						   "_confidenceV2": {
							   "score": 0.296
						   },
						   "startOffsetInMilliseconds": 1110,
						   "endOffsetInMilliseconds": 1630
					   },
					   {
						   "value": "soup",
						   "confidence": 0.443,
						   "_confidenceV2": {
							   "score": 0.443
						   },
						   "startOffsetInMilliseconds": 1650,
						   "endOffsetInMilliseconds": 1990
					   }
				   ],
				   "confidence": 0.83,
				   "_confidenceV2": {
					   "score": 0.83
				   },
				   "directedness": {
					   "score": 0
				   }
			   },
			   {
				   "tokens": [
					   {
						   "value": "recipes",
						   "confidence": 0.63,
						   "_confidenceV2": {
							   "score": 0.63
						   },
						   "startOffsetInMilliseconds": 210,
						   "endOffsetInMilliseconds": 880
					   },
					   {
						   "value": "for",
						   "confidence": 0.773,
						   "_confidenceV2": {
							   "score": 0.773
						   },
						   "startOffsetInMilliseconds": 900,
						   "endOffsetInMilliseconds": 1090
					   },
					   {
						   "value": "pumpkin",
						   "confidence": 0.357,
						   "_confidenceV2": {
							   "score": 0.357
						   },
						   "startOffsetInMilliseconds": 1110,
						   "endOffsetInMilliseconds": 1630
					   },
					   {
						   "value": "soup",
						   "confidence": 0.443,
						   "_confidenceV2": {
							   "score": 0.443
						   },
						   "startOffsetInMilliseconds": 1650,
						   "endOffsetInMilliseconds": 1990
					   }
				   ],
				   "confidence": 0.832,
				   "_confidenceV2": {
					   "score": 0.832
				   },
				   "directedness": {
					   "score": 0
				   }
			   },
			   {
				   "tokens": [
					   {
						   "value": "recipes",
						   "confidence": 0.63,
						   "_confidenceV2": {
							   "score": 0.63
						   },
						   "startOffsetInMilliseconds": 210,
						   "endOffsetInMilliseconds": 880
					   },
					   {
						   "value": "for",
						   "confidence": 0.773,
						   "_confidenceV2": {
							   "score": 0.773
						   },
						   "startOffsetInMilliseconds": 900,
						   "endOffsetInMilliseconds": 1090
					   },
					   {
						   "value": "pumping",
						   "confidence": 0.296,
						   "_confidenceV2": {
							   "score": 0.296
						   },
						   "startOffsetInMilliseconds": 1110,
						   "endOffsetInMilliseconds": 1630
					   },
					   {
						   "value": "suit",
						   "confidence": 0.174,
						   "_confidenceV2": {
							   "score": 0.174
						   },
						   "startOffsetInMilliseconds": 1650,
						   "endOffsetInMilliseconds": 1990
					   }
				   ],
				   "confidence": 0.823,
				   "_confidenceV2": {
					   "score": 0.823
				   },
				   "directedness": {
					   "score": 0
				   }
			   },
			   {
				   "tokens": [
					   {
						   "value": "recipes",
						   "confidence": 0.63,
						   "_confidenceV2": {
							   "score": 0.63
						   },
						   "startOffsetInMilliseconds": 210,
						   "endOffsetInMilliseconds": 880
					   },
					   {
						   "value": "for",
						   "confidence": 0.773,
						   "_confidenceV2": {
							   "score": 0.773
						   },
						   "startOffsetInMilliseconds": 900,
						   "endOffsetInMilliseconds": 1090
					   },
					   {
						   "value": "pumpkin",
						   "confidence": 0.357,
						   "_confidenceV2": {
							   "score": 0.357
						   },
						   "startOffsetInMilliseconds": 1110,
						   "endOffsetInMilliseconds": 1630
					   },
					   {
						   "value": "suit",
						   "confidence": 0.174,
						   "_confidenceV2": {
							   "score": 0.174
						   },
						   "startOffsetInMilliseconds": 1650,
						   "endOffsetInMilliseconds": 1990
					   }
				   ],
				   "confidence": 0.824,
				   "_confidenceV2": {
					   "score": 0.824
				   },
				   "directedness": {
					   "score": 0
				   }
			   },
			   {
				   "tokens": [
					   {
						   "value": "recipes",
						   "confidence": 0.63,
						   "_confidenceV2": {
							   "score": 0.63
						   },
						   "startOffsetInMilliseconds": 210,
						   "endOffsetInMilliseconds": 880
					   },
					   {
						   "value": "for",
						   "confidence": 0.773,
						   "_confidenceV2": {
							   "score": 0.773
						   },
						   "startOffsetInMilliseconds": 900,
						   "endOffsetInMilliseconds": 1090
					   },
					   {
						   "value": "pomping",
						   "confidence": 0.116,
						   "_confidenceV2": {
							   "score": 0.116
						   },
						   "startOffsetInMilliseconds": 1110,
						   "endOffsetInMilliseconds": 1630
					   },
					   {
						   "value": "soup",
						   "confidence": 0.443,
						   "_confidenceV2": {
							   "score": 0.443
						   },
						   "startOffsetInMilliseconds": 1650,
						   "endOffsetInMilliseconds": 1990
					   }
				   ],
				   "confidence": 0.823,
				   "_confidenceV2": {
					   "score": 0.823
				   },
				   "directedness": {
					   "score": 0
				   }
			   }
		   ],
		"proposed_tasks": [],
		'welcome_task': "build a picnic table",
		"welcome_recipe": "pumpkin soup"
	}

	print(handle_message(msg))
