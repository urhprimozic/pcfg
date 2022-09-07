from runpy import run_module


class CFG:
    '''
    Context-free grammar. 
    
    - self.N - a set of nonterminals
    - self.T - a set of terminals
    - self.S - start symbols
    - self.R - a set of rewrite rules ( tuples (A, alpha)), where A is in N and rewrites to alpha
    - self.rule_dict - a dictionary with keys from N and values alpha such that (A, alpha) is in self.R
    '''
    def from_tuple(self, N, T, S, R):
        '''
        Creates context-free grammar with nonterminal sybols from N, terminal from T, start symbol S and rules R

        Parameters
        ------------
         - N - a set (or iterable) of nonterminals
         - T - a set (or iterable) of terminals
         - S - a symbol from N
         - R - a set (or iterable) of rewrite rules (tuples (A, alpha))
        '''
        self.N = set(N)
        self.T = set(T)
        self.R = set(R)
        self.S = set(S)
        # dictionary nonterminal : rules
        self.rule_dict = {A : () for A in N}
        for A, alpha in R:
            self.rule_dict[A] = self.rule_dict[A] + alpha

    def from_rule_dict(self, rule_dict, S, T):
        '''
        Creates cfg from a dictionary with keys from N and values alpha such that (A, alpha) is in rules.

        Parameters
        -----------
         - rule_dict - a dictionary with keys from N and values alpha such that A rewrites into alpha
         - T - a set of terminal symbols
         - S - start symbol
        '''
        self.rule_dict=rule_dict
        self.N = set()
        self.T = T
        self.S = S
        for A in rule_dict.keys():
            self.N.add(A)
        self.R = set()
        for A, alpha in rule_dict.items():
            self.R.add((A, alpha))


    def __init__(self, *args) -> None:
        if len(args=0)
