import json
from cogagent.utils.policyBLEU_util import BLEUScorer
from cogagent.utils.policy_util import normalize
from cogagent.core.metric.base_metric import BaseMetric

import random
import sys
sys.path.append('..')
# from convlab2.policy.mdrg.multiwoz.utils.dbPointer import queryResultVenues
# from convlab2.policy.mdrg.multiwoz.utils.delexicalize import *
# from convlab2.policy.mdrg.multiwoz.utils.nlp import *
domains = ['restaurant', 'hotel', 'attraction', 'train', 'taxi', 'hospital', 'police']
requestables = ['phone', 'address', 'postcode', 'reference', 'id']

class BasePoliyMetric(BaseMetric):
    def __init__(self, dbs,delex_dialogues, default_metric_name=None):
        super().__init__()
        # self.label_list = list()
        # self.pre_list = list()
        self.label_list = {}
        self.pre_list = {}
        self.dbs = dbs
        self.delex_dialogues = delex_dialogues
        self.default_metric_name = default_metric_name
        if default_metric_name is None:
            self.default_metric_name = "total" 

        else:
            self.default_metric_name = default_metric_name
        

    def evaluate(self, pred, label): 
        self.pre_list = pred
        self.label_list = label
    
       
    def get_metric(self, reset=True):
        """Gathers statistics for the whole sets."""
        successes, matches = 0, 0
        total = 0
        gen_stats = {'restaurant': [0, 0, 0], 'hotel': [0, 0, 0], 'attraction': [0, 0, 0], 'train': [0, 0,0], 'taxi': [0, 0, 0],
                'hospital': [0, 0, 0], 'police': [0, 0, 0]}
        sng_gen_stats = {'restaurant': [0, 0, 0], 'hotel': [0, 0, 0], 'attraction': [0, 0, 0], 'train': [0, 0, 0],'taxi': [0, 0, 0],
                        'hospital': [0, 0, 0], 'police': [0, 0, 0]}
        
        val_dials_gen = self.pre_list
        delex_dialogues = self.delex_dialogues

        for filename, dialogue in val_dials_gen.items():    
            delex_data = delex_dialogues[filename]
        
            goal, _, _, requestables, _ = evaluateRealDialogue(self.dbs, delex_data)
            success, match, stats = evaluateGeneratedDialogue(self.dbs, dialogue, goal, delex_data, requestables)

            successes += success
            matches += match
            total += 1

            for domain in gen_stats.keys():
                gen_stats[domain][0] += stats[domain][0]
                gen_stats[domain][1] += stats[domain][1]
                gen_stats[domain][2] += stats[domain][2]

            if 'SNG' in filename:
                for domain in gen_stats.keys():
                    sng_gen_stats[domain][0] += stats[domain][0]
                    sng_gen_stats[domain][1] += stats[domain][1]
                    sng_gen_stats[domain][2] += stats[domain][2]

        # BLUE SCORE
        corpus = []
        model_corpus = []
        bscorer = BLEUScorer()
        val_dials = self.label_list

        for dialogue in val_dials_gen:
            data = val_dials[dialogue]
            model_turns, corpus_turns = [], []
            for idx, turn in enumerate(data['sys']):
                corpus_turns.append([turn])
            for turn in val_dials_gen[dialogue]:
                model_turns.append([turn])

            # if len(model_turns) == len(corpus_turns):
            corpus.extend(corpus_turns)
            model_corpus.extend(model_turns)
            # else:
            #     print('wrong length!!!')
            #     print(model_turns)

        # Print results
        # if mode == 'valid':
        try: print("Valid BLUES SCORE %.10f" % bscorer.score(model_corpus, corpus))
        except: print('BLUE SCORE ERROR')
        print('Valid Corpus Matches : %2.2f%%' % (matches / float(total) * 100))
        print('Valid Corpus Success : %2.2f%%' %  (successes / float(total) * 100))
        print('Valid Total number of dialogues: %s ' % total)
       
        match = matches / float(total) 
        success = successes / float(total) 
        bleuscore = bscorer.score(model_corpus, corpus)
        evaluate_result = {"match": match,
                            "success": success,
                            "total": total,
                            "BLEU": bleuscore
                            }
        if reset:
            self.label_list = {}
            self.pre_list = {}
        return evaluate_result


def parseGoal(goal, d, domain):
    """Parses user goal into dictionary format."""
    goal[domain] = {}
    goal[domain] = {'informable': [], 'requestable': [], 'booking': []}
    if 'info' in d['goal'][domain]:
        if domain == 'train':
            # we consider dialogues only where train had to be booked!
            if 'book' in d['goal'][domain]:
                goal[domain]['requestable'].append('reference')
            if 'reqt' in d['goal'][domain]:
                if 'trainID' in d['goal'][domain]['reqt']:
                    goal[domain]['requestable'].append('id')
        else:
            if 'reqt' in d['goal'][domain]:
                for s in d['goal'][domain]['reqt']:  # addtional requests:
                    if s in ['phone', 'address', 'postcode', 'reference', 'id']:
                        # ones that can be easily delexicalized
                        goal[domain]['requestable'].append(s)
            if 'book' in d['goal'][domain]:
                goal[domain]['requestable'].append("reference")

        goal[domain]["informable"] = d['goal'][domain]['info']
        if 'book' in d['goal'][domain]:
            goal[domain]["booking"] = d['goal'][domain]['book']

    return goal

def evaluateGeneratedDialogue(dbs, dialog, goal, realDialogue, real_requestables):
    """Evaluates the dialogue created by the model.
    First we load the user goal of the dialogue, then for each turn
    generated by the system we look for key-words.
    For the Inform rate we look whether the entity was proposed.
    For the Success rate we look for requestables slots"""
    # for computing corpus success
    requestables = ['phone', 'address', 'postcode', 'reference', 'id']

    # CHECK IF MATCH HAPPENED
    provided_requestables = {}
    venue_offered = {}
    domains_in_goal = []

    for domain in goal.keys():
        venue_offered[domain] = []
        provided_requestables[domain] = []
        domains_in_goal.append(domain)

    for t, sent_t in enumerate(dialog):
        for domain in goal.keys():
            # for computing success
            if '[' + domain + '_name]' in sent_t or '_id' in sent_t:
                if domain in ['restaurant', 'hotel', 'attraction', 'train']:
                    # HERE YOU CAN PUT YOUR BELIEF STATE ESTIMATION
                    if (t*2 + 1) < len(realDialogue['log']):
                        venues = queryResultVenues(dbs,domain, realDialogue['log'][t*2 + 1])

                        # if venue has changed
                        if len(venue_offered[domain]) == 0 and venues:
                            venue_offered[domain] = random.sample(venues, 1)
                        else:
                            flag = False
                            for ven in venues:
                                if venue_offered[domain][0] == ven:
                                    flag = True
                                    break
                            if not flag and venues:  # sometimes there are no results so sample won't work
                                # print venues
                                venue_offered[domain] = random.sample(venues, 1)
                else:  # not limited so we can provide one
                    venue_offered[domain] = '[' + domain + '_name]'

            # ATTENTION: assumption here - we didn't provide phone or address twice! etc
            for requestable in requestables:
                if requestable == 'reference':
                    if domain + '_reference' in sent_t:
                        if 'restaurant_reference' in sent_t:
                            if (t*2 < len(realDialogue['log'])):
                                if realDialogue['log'][t * 2]['db_pointer'][-5] == 1:  # if pointer was allowing for that?
                                    provided_requestables[domain].append('reference')

                        elif 'hotel_reference' in sent_t:
                            if (t*2 < len(realDialogue['log'])):
                                if realDialogue['log'][t * 2]['db_pointer'][-3] == 1:  # if pointer was allowing for that?
                                    provided_requestables[domain].append('reference')

                        elif 'train_reference' in sent_t:
                            if (t*2 < len(realDialogue['log'])):
                                if realDialogue['log'][t * 2]['db_pointer'][-1] == 1:  # if pointer was allowing for that?
                                    provided_requestables[domain].append('reference')

                        else:
                            provided_requestables[domain].append('reference')
                else:
                    if domain + '_' + requestable + ']' in sent_t:
                        provided_requestables[domain].append(requestable)

    # if name was given in the task
    for domain in goal.keys():
        # if name was provided for the user, the match is being done automatically
        if 'info' in realDialogue['goal'][domain]:
            if 'name' in realDialogue['goal'][domain]['info']:
                venue_offered[domain] = '[' + domain + '_name]'

        # special domains - entity does not need to be provided
        if domain in ['taxi', 'police', 'hospital']:
            venue_offered[domain] = '[' + domain + '_name]'


        if domain == 'train':
            if not venue_offered[domain]:
                if 'reqt' in realDialogue['goal'][domain] and 'id' not in realDialogue['goal'][domain]['reqt']:
                    venue_offered[domain] = '[' + domain + '_name]'

    """
    Given all inform and requestable slots
    we go through each domain from the user goal
    and check whether right entity was provided and
    all requestable slots were given to the user.
    The dialogue is successful if that's the case for all domains.
    """
    # HARD EVAL
    stats = {'restaurant': [0, 0, 0], 'hotel': [0, 0, 0], 'attraction': [0, 0, 0], 'train': [0, 0,0], 'taxi': [0, 0, 0],
             'hospital': [0, 0, 0], 'police': [0, 0, 0]}

    match = 0
    success = 0
    # MATCH
    for domain in goal.keys():
        match_stat = 0
        if domain in ['restaurant', 'hotel', 'attraction', 'train']:
            goal_venues = queryResultVenues(dbs, domain, goal[domain]['informable'], real_belief=True)
            if type(venue_offered[domain]) is str and '_name' in venue_offered[domain]:
                match += 1
                match_stat = 1
            elif len(venue_offered[domain]) > 0 and venue_offered[domain][0] in goal_venues:
                match += 1
                match_stat = 1
        else:
            if domain + '_name]' in venue_offered[domain]:
                match += 1
                match_stat = 1

        stats[domain][0] = match_stat
        stats[domain][2] = 1

    if match == len(goal.keys()):
        match = 1
    else:
        match = 0

    # SUCCESS
    if match:
        for domain in domains_in_goal:
            success_stat = 0
            domain_success = 0
            if len(real_requestables[domain]) == 0:
                success += 1
                success_stat = 1
                stats[domain][1] = success_stat
                continue
            # if values in sentences are super set of requestables
            for request in set(provided_requestables[domain]):
                if request in real_requestables[domain]:
                    domain_success += 1

            if domain_success >= len(real_requestables[domain]):
                success += 1
                success_stat = 1

            stats[domain][1] = success_stat

        # final eval
        if success >= len(real_requestables):
            success = 1
        else:
            success = 0

    #rint requests, 'DIFF', requests_real, 'SUCC', success
    return success, match, stats

def evaluateRealDialogue(dbs, delex_data):
    """Evaluation of the real dialogue.
    First we loads the user goal and then go through the dialogue history.
    Similar to evaluateGeneratedDialogue above."""
    domains = ['restaurant', 'hotel', 'attraction', 'train', 'taxi', 'hospital', 'police']
    requestables = ['phone', 'address', 'postcode', 'reference', 'id']
    # get the list of domains in the goal
    domains_in_goal = []
    goal = {}
    for domain in domains:
        if delex_data['goal'][domain]:
            goal = parseGoal(goal, delex_data, domain)
            domains_in_goal.append(domain)

    # compute corpus success
    real_requestables = {}
    provided_requestables = {}
    venue_offered = {}
    for domain in goal.keys():
        provided_requestables[domain] = []
        venue_offered[domain] = []
        real_requestables[domain] = goal[domain]['requestable']

    # iterate each turn
    m_targetutt = [turn['text'] for idx, turn in enumerate(delex_data['log']) if idx % 2 == 1]
    for t in range(len(m_targetutt)):
        for domain in domains_in_goal:
            sent_t = m_targetutt[t]
            # for computing match - where there are limited entities
            if domain + '_name' in sent_t or '_id' in sent_t:
                if domain in ['restaurant', 'hotel', 'attraction', 'train']:
                    # HERE YOU CAN PUT YOUR BELIEF STATE ESTIMATION
                    venues = queryResultVenues(dbs, domain, delex_data['log'][t * 2 + 1])

                    # if venue has changed
                    if len(venue_offered[domain]) == 0 and venues:
                        venue_offered[domain] = random.sample(venues, 1)
                    else:
                        flag = False
                        for ven in venues:
                            if venue_offered[domain][0] == ven:
                                flag = True
                                break
                        if not flag and venues:  # sometimes there are no results so sample won't work
                            #print venues
                            venue_offered[domain] = random.sample(venues, 1)
                else:  # not limited so we can provide one
                    venue_offered[domain] = '[' + domain + '_name]'

            for requestable in requestables:
                # check if reference could be issued
                if requestable == 'reference':
                    if domain + '_reference' in sent_t:
                        if 'restaurant_reference' in sent_t:
                            if delex_data['log'][t * 2]['db_pointer'][-5] == 1:  # if pointer was allowing for that?
                                provided_requestables[domain].append('reference')

                        elif 'hotel_reference' in sent_t:
                            if delex_data['log'][t * 2]['db_pointer'][-3] == 1:  # if pointer was allowing for that?
                                provided_requestables[domain].append('reference')

                                #return goal, 0, match, real_requestables
                        elif 'train_reference' in sent_t:
                            if delex_data['log'][t * 2]['db_pointer'][-1] == 1:  # if pointer was allowing for that?
                                provided_requestables[domain].append('reference')

                        else:
                            provided_requestables[domain].append('reference')
                else:
                    if domain + '_' + requestable in sent_t:
                        provided_requestables[domain].append(requestable)

    # offer was made?
    for domain in domains_in_goal:
        # if name was provided for the user, the match is being done automatically
        if 'info' in delex_data['goal'][domain]:
            if 'name' in delex_data['goal'][domain]['info']:
                venue_offered[domain] = '[' + domain + '_name]'

        # special domains - entity does not need to be provided
        if domain in ['taxi', 'police', 'hospital']:
            venue_offered[domain] = '[' + domain + '_name]'

        # if id was not requested but train was found we dont want to override it to check if we booked the right train
        if domain == 'train' and (not venue_offered[domain] and 'id' not in goal['train']['requestable']):
            venue_offered[domain] = '[' + domain + '_name]'

    # HARD (0-1) EVAL
    stats = {'restaurant': [0, 0, 0], 'hotel': [0, 0, 0], 'attraction': [0, 0, 0], 'train': [0, 0,0], 'taxi': [0, 0, 0],
             'hospital': [0, 0, 0], 'police': [0, 0, 0]}

    match, success = 0, 0
    # MATCH
    for domain in goal.keys():
        match_stat = 0
        if domain in ['restaurant', 'hotel', 'attraction', 'train']:
            goal_venues = queryResultVenues(dbs, domain, delex_data['goal'][domain]['info'], real_belief=True)
            #print(goal_venues)
            if type(venue_offered[domain]) is str and '_name' in venue_offered[domain]:
                match += 1
                match_stat = 1
            elif len(venue_offered[domain]) > 0 and venue_offered[domain][0] in goal_venues:
                match += 1
                match_stat = 1

        else:
            if domain + '_name' in venue_offered[domain]:
                match += 1
                match_stat = 1

        stats[domain][0] = match_stat
        stats[domain][2] = 1

    if match == len(goal.keys()):
        match = 1
    else:
        match = 0

    # SUCCESS
    if match:
        for domain in domains_in_goal:
            domain_success = 0
            success_stat = 0
            if len(real_requestables[domain]) == 0:
                # check that
                success += 1
                success_stat = 1
                stats[domain][1] = success_stat
                continue
            # if values in sentences are super set of requestables
            for request in set(provided_requestables[domain]):
                if request in real_requestables[domain]:
                    domain_success += 1

            if domain_success >= len(real_requestables[domain]):
                success +=1
                success_stat = 1

            stats[domain][1] = success_stat

        # final eval
        if success >= len(real_requestables):
            success = 1
        else:
            success = 0

    return goal, success, match, real_requestables, stats

def queryResultVenues(dbs, domain, turn, real_belief=False):
    # query the db
    sql_query = "select * from {}".format(domain)

    flag = True
    if real_belief == True:
        items = turn.items()
    elif real_belief=='tracking':
        for slot in turn[domain]:
            key = slot[0].split("-")[1]
            val = slot[0].split("-")[2]
            if key == "price range":
                key = "pricerange"
            elif key == "leave at":
                key = "leaveAt"
            elif key == "arrive by":
                key = "arriveBy"
            if val == "do n't care":
                pass
            else:
                if flag:
                    sql_query += " where "
                    val2 = val.replace("'", "''")
                    val2 = normalize(val2)
                    val2 = val2.replace("'", "''")
                    if key == 'leaveAt':
                        sql_query += key + " > " + r"'" + val2 + r"'"
                    elif key == 'arriveBy':
                        sql_query += key + " < " + r"'" + val2 + r"'"
                    else:
                        sql_query += r" " + key + "=" + r"'" + val2 + r"'"
                    flag = False
                else:
                    val2 = val.replace("'", "''")
                    val2 = normalize(val2)
                    val2 = val2.replace("'", "''")
                    if key == 'leaveAt':
                        sql_query += r" and " + key + " > " + r"'" + val2 + r"'"
                    elif key == 'arriveBy':
                        sql_query += r" and " + key + " < " + r"'" + val2 + r"'"
                    else:
                        sql_query += r" and " + key + "=" + r"'" + val2 + r"'"

            try:  # "select * from attraction  where name = 'queens college'"
                return dbs[domain].execute(sql_query).fetchall()
            except:
                return []  # TODO test it
        pass
    else:
        items = turn['metadata'][domain]['semi'].items()

    flag = True
    for key, val in items:
        if val == "" or val == "dontcare" or val == 'not mentioned' or val == "don't care" or val == "dont care" or val == "do n't care":
            pass
        else:
            if flag:
                sql_query += " where "
                val2 = val.replace("'", "''")
                val2 = normalize(val2)
                val2 = val2.replace("'", "''")
                if key == 'leaveAt':
                    sql_query += r" " + key + " > " + r"'" + val2 + r"'"
                elif key == 'arriveBy':
                    sql_query += r" " +key + " < " + r"'" + val2 + r"'"
                else:
                    sql_query += r" " + key + "=" + r"'" + val2 + r"'"
                flag = False
            else:
                val2 = val.replace("'", "''")
                val2 = normalize(val2)
                val2 = val2.replace("'", "''")
                if key == 'leaveAt':
                    sql_query += r" and " + key + " > " + r"'" + val2 + r"'"
                elif key == 'arriveBy':
                    sql_query += r" and " + key + " < " + r"'" + val2 + r"'"
                else:
                    sql_query += r" and " + key + "=" + r"'" + val2 + r"'"

    try:  # "select * from attraction  where name = 'queens college'"
        return dbs[domain].execute(sql_query).fetchall()
    except:
        raise
        return []  # TODO test it

        