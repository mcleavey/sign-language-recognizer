import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    
    
    hwords = test_set.get_all_Xlengths()
    
    # for each unknown word i 
    for i in hwords:
        
        # build a dictionary - for each trained word our model knows, give the probability the unknown word is this word
        prob_i = dict()
        
        X,lengths = hwords[i]
        
        # test each trained word
        for m in models:
            try:
                logL = models[m].score(X, lengths)
            except Exception as e:
                logL = float("-inf")
            prob_i[m] = logL
            
        probabilities.append(prob_i)
        
        # of that dictionary, output the key (word) with the highest probability, append that to the list of guesses, so
        # that guesses[i] will be the highest probability word for unknown word i
        guesses.append(max(prob_i, key=lambda key: prob_i[key]))

    return probabilities, guesses
