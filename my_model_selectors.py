import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        pass

    
    
    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """
    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        super(SelectorBIC, self).__init__(all_word_sequences,
                                         all_word_Xlengths,
                                         this_word,
                                         n_constant=3,
                                         min_n_components=min_n_components,
                                         max_n_components=max_n_components,
                                         random_state=random_state,
                                         verbose=verbose)

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # Set best model to nonexistant
        best_mod = None
        bestBIC = float("+inf")
        alpha = 1.0   # comparable to regularization term for neural nets

        for n in range(self.min_n_components, self.max_n_components+1):
        # Find BIC score for each option of n_components
        
            try:  # in case model create or score throws exception, ignore this model and test next one
                mod_n = GaussianHMM(n_components=n, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
                logL = mod_n.score(self.X, self.lengths)
            except:
                logL = float("-inf")
                mod_n = None
                

            numFeatures = self.X.shape[1]

            # Number of parameters (Formula 1)
            p = n*n + 2*n*(numFeatures-1)      
                                                    # n*(n-1) options for transition probabilities
                                                    # n-1                 starting probabilities
                                                    # n*numFeatures       number of means
                                                    # n*numFeatures       variances (diag of covars array)

            # Formula 2 (almost exactly the same, though in some cases yields better results than formula 1)
#            p = n*(n-1) + (n-1) + 2*n*numFeatures   

            numData = len(self.X)              # Number of data points
                
            BIC = -2*logL + alpha * p*np.log(numData)  # Calculate BIC score

            # If this model is better than previous best, set this as the new best
            if BIC < bestBIC:
                bestBIC = BIC
                best_mod = mod_n
            
        return best_mod



class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))

    '''
    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        super(SelectorDIC, self).__init__(all_word_sequences,
                                         all_word_Xlengths,
                                         this_word,
                                         n_constant=3,
                                         min_n_components=min_n_components,
                                         max_n_components=max_n_components,
                                         random_state=random_state,
                                         verbose=verbose)

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # Set best model to nonexistant
        best_mod = None
        bestDIC = float("-inf")
        copy_factor = .2
        
        m = len(self.hwords)        # total number of training words
        
        for n in range(self.min_n_components, self.max_n_components):
            m_scored = m            # keep track of how many other words get scored
            m_copies = 0            # keep track if this target word has multiple versions (like GO, GO1)
            logOtherWords=0 
            logCopies = 0
            
            try:
                mod_n = GaussianHMM(n_components=n, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
                logL = mod_n.score(self.X, self.lengths)
                   
            # in case model create or score throws exception, ignore this model and test next one
            except:
                logL = float("-inf")
                mod_n = None

                
            # for all possible training words (check that your proposed model for this word doesn't match well with other words)
            for w in self.words:
                if w==self.this_word:    # skip if it's the word for the current model
                    continue
                
                is_copy = False
                
                if "0"<=w[-1] and "9">=w[-1]:
                    if self.this_word == w[:-1]:
                        is_copy = True
                    if self.this_word[:-1] == w[:-1]:
                        is_copy = True
                        
                if "0"<=self.this_word[-1] and "9">=self.this_word[-1]:
                    if w==self.this_word[:-1]:
                        is_copy = True
                    if w[:-1] == self.this_word[:-1]:
                        is_copy = True
                        
                x, l = self.hwords[w] 
                
                # See how well a different word matches our current word model (skip this word if it fails to return a score)
                try:   
                    if is_copy:
                        logCopies += mod_n.score(x, l)
                        m_scored -= 1
                        m_copies += 1
                    else:
                        logOtherWords += mod_n.score(x, l)
                except:
                    m_scored -= 1
                    
                 
            DIC = logL - logOtherWords/max(1, (m_scored-1))
            DIC += copy_factor * logCopies / max(2, m_copies)

            # If this is our new best model, replace the old best model
            if DIC>bestDIC:
                bestDIC=DIC
                best_mod = mod_n
        return best_mod
        

        

class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''
    
    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        super(SelectorCV, self).__init__(all_word_sequences,
                                         all_word_Xlengths,
                                         this_word,
                                         n_constant=3,
                                         min_n_components=min_n_components,
                                         max_n_components=max_n_components,
                                         random_state=random_state,
                                         verbose=verbose)
        
        
    def averageLL(self, n):
        # Set best model to nonexistant
        best_mod = None
        bestLogL = float("-inf")

        # Split sequences (default to 3 unless self.sequences length is too short) 

        if len(self.sequences)<2:
            try:
                mod_n = GaussianHMM(n_components=n, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
                logL = mod_n.score(self.X, self.lengths)
            except:
                logL = float("-inf")
                mod_n = None
            return logL, mod_n
        
        n_splits = min(3, len(self.sequences))
        split_method = KFold(n_splits = n_splits)

        for cv_train_idx, cv_test_idx in split_method.split(self.sequences):

            # for a given split of test/train data, train model and test log loss
            Xtrain, lengths_train = combine_sequences(cv_train_idx, self.sequences)
            Xtest, lengths_test = combine_sequences(cv_test_idx, self.sequences)
            
            try:
                mod_n = GaussianHMM(n_components=n, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(Xtrain, lengths_train)
                logL = mod_n.score(Xtest, lengths_test)

                # If this model is better than previous best, set this as the new best
                if logL>bestLogL:
                    bestLogL = logL
                    best_mod = mod_n
            
            # in case model create or score throws exception, ignore this model and test next one
            except:
                logL = float("-inf")
                mod_n = None

        return bestLogL, best_mod
        
        
    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        best_logL = float("-inf")
        best_model = None

        # Implement model selection using CV
        for n in range(self.min_n_components, self.max_n_components+1):
            # get logL, mod_n for n hidden components - averageLL finds average log loss with 
            # different validation splits
            logL, mod_n = self.averageLL(n)

            if logL>best_logL:
                best_logL = logL
                best_model = mod_n
        
        return best_model
