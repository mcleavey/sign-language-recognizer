import math
import arpa
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from word_search import SentenceProblem

class SentenceSelectorBase(object):
    '''
    base class for sentence selection (strategy design pattern)
    '''

    def __init__(self, lm_model: arpa.models.simple.ARPAModelSimple, sentence: dict, num_words_to_test=5):
        self.lm_model = lm_model
        self.sentence = sentence
        self.num_words_to_test = num_words_to_test
        
    def remove_numbers(self, choice: list):
        # remove the digit from the ends of words that have multiple signs - like "GO1"
        #    choice is a 2D list - for each word i in the test sentence, we have the j best guesses from HMM

        for i in range(len(choice)):
            for j in range(len(choice[i])):
                if '0' <= choice[i][j][-1] and choice[i][j][-1] <= '9':
                    choice[i][j] = choice[i][j][:-1]
        return choice
        
    def find_best(self, sentences: list, word_score: list):
        best_sentence = ""    
        second_best_sentence = ""
        best_score = float("-inf")
        word_score_factor = .9
        
        best_sentence = sentences[0]
        LMscore = []       # the language model score for a given sentence (cumulative likelihood that each set of 3 words  
                                                                    # would occur in that order 
        HMMscore = []      # the HMM score for the sentence (cumulative likelihood that each word in the sentence is correct
                                                                    # based on the HMM confidence it's detected that word

        # consider all the possible sentences (based on all possible combinations of the top few HMM guesses for each word
        #     determine the cumulative LMscore and HMMscore for each possible sentence)  
        for ix_s, sent in enumerate(sentences):
            words = sent.split(' ')

            #  Determine cumulative LMscore for this proposed sentence sent
            LMscore.append(0)
#            LMscore[ix_s] += self.lm_model.log_p("<s> "+words[0])    # Results were better if I don't include the probability
                                                                     # for the start of the sentence. Possibly this was just a chance
                                                                     # overfitting to the test set here, so I'm leaving this line
                                                                     # so there's the option to add it back in
            LMscore[ix_s] += self.lm_model.log_p(words[0] + " " + words[1]) 
            for ix_w in range(2, len(words)-1):
                LMscore[ix_s] += np.log10(self.lm_model.p(words[ix_w-2]+" "+words[ix_w-1]+" "+words[ix_w])) 
            LMscore[ix_s] += np.log10(self.lm_model.p(words[len(words)-2] + " " + words[len(words)-1]+" </s>"))
            LMscore[ix_s] += np.log10(self.lm_model.p(words[len(words)-1]+" </s>"))
            

            # Determine cumulative HMMscore for this proposed sentence sent
            HMMscore.append(0)
            for ix_w in range(0, len(words)-1):
                HMMscore[ix_s] += word_score[ix_s][ix_w]
            

        # Normalize LMscore and HMMscore (subtract minimum value and then divide by maximum -- plus a tiny bit in case max is zero)
        LMscore -= min(LMscore)
        HMMscore -= min(HMMscore)
        LMscore /= (max(LMscore)+.00001)
        HMMscore /= (max(HMMscore)+.00001)


        # Now look for the highest score out of all the proposed sentences.  (For model development, I also kept track of the
        #   second best score, to see if my model was just barely missing, or was completely off the mark).
        for ix_s, sent in enumerate(sentences):
            LMscore[ix_s] += word_score_factor * HMMscore[ix_s]    #word_score_factor is a tuning parameter - how much to weight
                                                                           # HMMscore vs LMscore
            if LMscore[ix_s]>best_score:
                second_best_sentence = best_sentence
                best_sentence = sent
                best_score = LMscore[ix_s]
        
        return best_sentence, second_best_sentence
        
        
        
    
 
    
class SentenceSelector(object):
    '''
    Sentence Selector - uses the top num_words_to_test guesses from the HMM for each unknown word in the sentence
    (defaults to using the top 5 guesses).  These are combined into all the possible sentences 
    (for a sentence that is 4 words long, this would be num_words_to_test ^ 4 possible sentences).
    
    SentenceSelector.score is called once for each test sentence, and returns the best guess for the sentence (based on combining
       Language Model and HMM scores), and also the second best guess for the sentence.
    
    Inputs: 
        lm_model:  the language model
        sentence: dict
            for each unknown word in the test sentence, this is the HMM score for every word the HMM has been trained on
            we then consider only the highest num_words_to_test scores for each unknown word
        num_words_to_test: int
            the number of best guesses to consider for each unknown word
             
    '''

    def __init__(self, lm_model: arpa.models.simple.ARPAModelSimple, sentence: dict, num_words_to_test = 5):
        self.lm_model = lm_model
        self.sentence = sentence                       
        self.num_words_to_test = num_words_to_test
        
    def score(self):
        num_best = self.num_words_to_test
        choice = []   # the guess words that are selected as possibilities for each unknown word
        prob = []     # the HMM probabilities of each guess word

        for i in range(num_best):
            choice.append([])
            prob.append([])
            
        for i, s in enumerate(self.sentence):    # for each unknown word in the test sentence
            for j in range(num_best):          # find the num_best best guesses (the num_best words with the highest HMM scores)
                choice[j].append(max(self.sentence[s], key=lambda key: self.sentence[s][key]))   # append the best word
                prob[j].append(self.sentence[s][choice[j][i]])                                   # append the probability for that word
                self.sentence[s].pop(choice[j][i], None)                             # remove that word from the dictionary sentence



        # From here on out, we're only considering the num_best guesses for each unknown word.
        #  We now normalize those probabilities.
        np_prob = np.array(prob)
        sk = MinMaxScaler()
        scaled_prob = sk.fit_transform(np_prob)

        
        # Combine the num_best guesses for each word into all the possible sentences.
        #  For a sentence that is 4 words long, we should have num_best^4 possible sentences.
        sentences = []
        sentence_score = []
        word_score = []
        

        # First, clean up our words by removing the digit from the ends of words like "GO1"
        #  choice is the 2D list of the num_best guess words that are possibilities for each unknown word in the test sentence
        for i in range(len(choice)):
            for j in range(len(choice[i])):
                if '0' <= choice[i][j][-1] and choice[i][j][-1] <= '9':
                    choice[i][j] = choice[i][j][:-1]
                    

        # Construct all the possible sentences
        for i in range(np.power(len(choice), len(choice[0]))):   # cycle through the (number of words in sentence)^(number guesses) 
                                                                 #   possible sentences
                j = i
                sentences.append("")
                word_score.append([])

                for k in range(0, len(choice[0])):              #  for each unknown word in the test sentence, pick a guess word
                                                                # and add it to build sentences[i].  Think of this sort of like counting 
                                                                # in base num_best - say the unknown sentence is 4 words long, you want
                                                                # to cycle through the indices 0000, 1000, 2000, 3000, 0100, 1100, 2100...
                                                                # Each one of those indices is a possible sentence: sentences[i].
                    sentences[i] += choice[int(j%num_best)][k] + " "
                    word_score[i].append(scaled_prob[int(j%num_best)][k])
                    j /= num_best
                               

            
        # Of all those sentences, now pick the highest scoring one (also keep track of the second best sentence, just out of interest)
        best_sentence = ""    
        second_best_sentence = ""
        best_score = float("-inf")
        word_score_factor = .9    # A hyperparameter to set how much to weight the HMM score as compared with the LM score.
        
        best_sentence = sentences[0]
        LMscore = []
        HMMscore = []

        # Determine the LM score and HMM score for each proposed sentence
        for ix_s, sent in enumerate(sentences):
            words = sent.split(' ')
            LMscore.append(0)
#            LMscore[ix_s] += self.lm_model.log_p("<s> "+words[0])
            LMscore[ix_s] += self.lm_model.log_p(words[0] + " " + words[1]) 
            for ix_w in range(2, len(words)-1):
                LMscore[ix_s] += np.log10(self.lm_model.p(words[ix_w-2]+" "+words[ix_w-1]+" "+words[ix_w])) 
            LMscore[ix_s] += np.log10(self.lm_model.p(words[len(words)-2] + " " + words[len(words)-1]+" </s>"))
            LMscore[ix_s] += np.log10(self.lm_model.p(words[len(words)-1]+" </s>"))
            

            HMMscore.append(0)
            for ix_w in range(0, len(words)-1):
                HMMscore[ix_s] += word_score[ix_s][ix_w]
            
    
        # Normalize the LM and HMM scores (subtract min, and then divide by maximum - plus a tiny extra in case max is 0)
        LMscore -= min(LMscore)
        HMMscore -= min(HMMscore)
        LMscore /= (max(LMscore)+.00001)
        HMMscore /= (max(HMMscore)+.00001)

        # Find the best (and second best) sentences by finding the sentence with the highest max combined LM and HMM scores
        for ix_s, sent in enumerate(sentences):
            LMscore[ix_s] += word_score_factor * HMMscore[ix_s]
            if LMscore[ix_s]>best_score:
                second_best_sentence = best_sentence
                best_sentence = sent
                best_score = LMscore[ix_s]
                
            
        return best_sentence, second_best_sentence

    
   
# Below this are versions that I tested, but which did not yield better results. I'm leaving them here only if
# I return to tinker at a later date.  In particular, I like the idea of using search to get better results,
# but I didn't have time to finish that.




class SentenceSelectorFast(SentenceSelectorBase):
    '''
    base class for sentence selection (strategy design pattern)
    '''

    def __init__(self, lm_model: arpa.models.simple.ARPAModelSimple, sentence: dict, num_words_to_test = 10):
        super(SentenceSelectorFast, self).__init__(lm_model, sentence, num_words_to_test)
        
    def score(self):
        num_words_to_test = 10
        choice = []
        prob = []
        for i in range(num_words_to_test):
            choice.append([])
            prob.append([])
            
        for i, s in enumerate(self.sentence):
            for j in range(num_words_to_test-1):
                choice[j].append(max(self.sentence[s], key=lambda key: self.sentence[s][key]))
                prob[j].append(self.sentence[s][choice[j][i]])
                self.sentence[s].pop(choice[j][i], None)

            choice[num_words_to_test-1].append(max(self.sentence[s], key=lambda key: self.sentence[s][key]))
            prob[num_words_to_test-1].append(self.sentence[s][choice[num_words_to_test-1][i]])


        np_prob = np.array(prob)
        sk = MinMaxScaler()
        scaled_prob = sk.fit_transform(np_prob)

        #make sentences
        sentences = []
        sentence_score = []
        word_score = []
        
        choice = self.remove_numbers(choice)
                    
        for i in range(len(choice)*len(choice[i])):
                sentences.append("")
                sentence_score.append(1)

                                
        for i, s in enumerate(sentences):
                j = i
                word_score.append([])
                for k in range(0, len(choice[0])):
                    sentences[i] += choice[int(j%num_words_to_test)][k] + " "
                    word_score[i].append(scaled_prob[int(j%num_words_to_test)][k])
                    j /= num_words_to_test
                               

        best_sentence, second_best_sentence = self.find_best(sentences, word_score)
                
            
        return best_sentence, second_best_sentence

    
class SentenceSelectorTwoProb(object):
    '''
    base class for sentence selection (strategy design pattern)
    '''

    def __init__(self, lm_model: arpa.models.simple.ARPAModelSimple, sentence: dict, sentenceB: dict):
        self.lm_model = lm_model
        self.sentence = sentence
        self.sentenceB = sentenceB
        
    def score(self):
        num_best = 3
        other_sentence_factor = 0
        choice = []
        prob = []
        
        for i in range(num_best*2):
            choice.append([])
            prob.append([])
            

        temp = {}
        
        for i, s in enumerate(self.sentence):            
            for j in range(num_best):
                choice[2*j].append(max(self.sentence[s], key=lambda key: self.sentence[s][key]))
                prob[2*j].append(self.sentence[s][choice[2*j][i]]
                            + other_sentence_factor * self.sentenceB[s][choice[2*j][i]])
                if s in temp:
                    temp[s][choice[2*j][i]] = self.sentence[s][choice[2*j][i]]
                else:
                    temp_item = {}
                    temp_item[choice[2*j][i]] = self.sentence[s][choice[2*j][i]]
                    temp[s] = temp_item 
                self.sentence[s].pop(choice[2*j][i], None)


        for t in temp:
            for w in temp[t]:
                self.sentence[t][w] = temp[t][w]

            

        for i, s in enumerate(self.sentenceB):
            for j in range(num_best):
                choice[2*j+1].append(max(self.sentenceB[s], key=lambda key: self.sentenceB[s][key]))
                prob[2*j+1].append(self.sentenceB[s][choice[2*j+1][i]] 
                                   + other_sentence_factor * self.sentence[s][choice[2*j+1][i]])
                self.sentenceB[s].pop(choice[2*j+1][i], None)                
                 

#        print(choice)

        np_prob = np.array(prob)
        sk = MinMaxScaler()
        scaled_prob = sk.fit_transform(np_prob)
#        print(scaled_prob) 
        
        #make sentences
        sentences = []
        sentence_score = []
        word_score = []
        
        # remove the digit from the ends of words like GO1
        for i in range(len(choice)):
            for j in range(len(choice[i])):
                if '0' <= choice[i][j][-1] and choice[i][j][-1] <= '9':
                    choice[i][j] = choice[i][j][:-1]
                    
                    
#                sentences.append("")
#                sentence_score.append(1)

                                
#        for i, s in enumerate(sentences):
        for i in range(np.power(len(choice), len(choice[0]))):
                sentences.append("")
                sentence_score.append(1)

                j = i
                word_score.append([])
                for k in range(0, len(choice[0])):
                    sentences[i] += choice[int(j%num_best)][k] + " "
                    word_score[i].append(scaled_prob[int(j%num_best)][k])
                    sentence_score[i] *= scaled_prob[int(j%num_best)][k]
                    j /= num_best
                               

            
        best_sentence = ""    
        second_best_sentence = ""
        best_score = float("-inf")
        word_score_factor = .9
        
        best_sentence = sentences[0]
        lscore = []
        lword = []
        for ix_s, sent in enumerate(sentences):
            words = sent.split(' ')
            lscore.append(0)
#            lscore[ix_s] += self.lm_model.log_p("<s> "+words[0])
            lscore[ix_s] += self.lm_model.log_p(words[0] + " " + words[1]) 
            for ix_w in range(2, len(words)-1):
                lscore[ix_s] += np.log10(self.lm_model.p(words[ix_w-2]+" "+words[ix_w-1]+" "+words[ix_w])) 
            lscore[ix_s] += np.log10(self.lm_model.p(words[len(words)-2] + " " + words[len(words)-1]+" </s>"))
            lscore[ix_s] += np.log10(self.lm_model.p(words[len(words)-1]+" </s>"))
            

            lword.append(0)
            for ix_w in range(0, len(words)-1):
                lword[ix_s] += word_score[ix_s][ix_w]
            
    
        lscore -= min(lscore)
        lword -= min(lword)
        lscore /= (max(lscore)+.00001)
        lword /= (max(lword)+.00001)

 #       print("----------------------------------")

        for ix_s, sent in enumerate(sentences):
            lscore[ix_s] += word_score_factor * lword[ix_s]
 #           print(str(sent)+ ": " +str(lscore[ix_s]))
            if lscore[ix_s]>best_score:
                second_best_sentence = best_sentence
                best_sentence = sent
                best_score = lscore[ix_s]
                
            
        return best_sentence, second_best_sentence
                              
       
        
class SentenceSelectorSearch(SentenceSelectorBase):
    '''
    base class for sentence selection (strategy design pattern)
    '''

    def __init__(self, lm_model: arpa.models.simple.ARPAModelSimple, sentence: dict, num_words_to_test = 10):
        super(SentenceSelectorFast, self).__init__(lm_model, sentence, num_words_to_test)
        
    def score(self):
        num_best = self.num_best
        choice = []
        prob = []
        
        for i in range(num_best):
            choice.append([])
            prob.append([])
            
        for i, s in enumerate(self.sentence):
            for j in range(num_best):
                choice[j].append(max(self.sentence[s], key=lambda key: self.sentence[s][key]))
                prob[j].append(self.sentence[s][choice[j][i]])
                self.sentence[s].pop(choice[j][i], None)


        np_prob = np.array(prob)
        sk = MinMaxScaler()
        scaled_prob = sk.fit_transform(np_prob)

        
        #make sentences
        sentences = []
        sentence_score = []
        word_score = []
        
        # remove the digit from the ends of words like GO1
        choice = self.remove_numbers(choice)

                                
        problem = SentenceProblem(choice, scaled_prob)
                               

            
        best_sentence = ""    
        second_best_sentence = ""

        

                
            
        return best_sentence, second_best_sentence
