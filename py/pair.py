class ChildParentPair(object):

    def __init__(self, child, parent, type_):
        self.child = child
        self.parent = parent
        self.type_ = type_
        if type_ == 'PREFIX':
            self.type_coarse = 'pre'
        elif type_ == 'SUFFIX' or type_ == 'DELETE' or type_ == 'REPEAT' or type_ == 'MODIFY':
            self.type_coarse = 'suf'
        else:
            self.type_coarse = None

    def get_affix(self):
        if hasattr(self, 'affix'):
            return self.affix
        else:
            return self.get_affix_and_transformation()[0]

    def get_affix_and_transformation(self):
        affix, trans = None, None
        if hasattr(self, 'affix'):
            affix = self.affix
            trans = self.trans
            return affix, trans
        else:
            if self.type_ in ['COM_LEFT', 'COM_RIGHT', 'HYPHEN', 'STOP']:
                pass
            # global affix_cache
            # key = (self.child, candidate)
            # if key in affix_cache: return affix_cache[key]
            # self.parent, type_ = candidate
            elif self.type_ == 'PREFIX':
                affix = self.child[:len(self.child) - len(self.parent)]
            elif self.type_ in ['SUFFIX', 'APOSTR']:
                affix = self.child[len(self.parent):]
            elif self.type_ == 'MODIFY':
                affix = self.child[len(self.parent):]
                trans = 'MOD_' + self.parent[-1] + '_' +  self.child[len(self.parent) - 1]
            elif self.type_ == 'DELETE':
                affix = self.child[len(self.parent) - 1:]
                trans = 'DEL_' + self.parent[-1]
            elif self.type_ == 'REPEAT':
                assert self.child[len(self.parent)] == self.child[len(self.parent) - 1], self.child + '\t' + self.parent
                affix = self.child[len(self.parent) + 1:]
                trans = 'REP_' + self.parent[-1]
            else:
                raise NotImplementedError, 'no such type %s' %(self.type_)
            self.affix = affix
            self.trans = trans
        return affix, trans

            # elif self.type_ == 'COMPOUND':
            #     return (None, 'COMPOUND')
            # elif type(self.type_) == tuple:
            #     assert self.type_[1] == 'COMPOUND'
            #     return (None, 'COMPOUND')
            # elif self.type_ == 'COM_LEFT' or self.type_ == 'COM_RIGHT':
            #         return (None, self.type_)
            #     else:
            #         assert False, self.child + '\t' + str(candidate)
                #
                # if self.type_ == 'PREFIX':
                #     affix_cache[key] = (affix, 'p')
                #     return (affix, 'p')
                # else:
                #     affix_cache[key] = (affix, 's')
                #     return (affix, 's')
        #
        # self.parent, type_ = candidate
        # if type_ == 'SUFFIX' or type_ == 'PREFIX': return None
        # if type_ == 'MODIFY':
        #     return (self.parent[-1] + '->' + self.child[len(self.parent) - 1], type_)
        # elif type_ == 'DELETE':
        #     return (self.parent[-1], type_)
        # elif type_ == 'REPEAT':
        #     return (self.parent[-1], type_)
        # elif type_ == 'COM_LEFT' or type_ == 'COM_RIGHT':
        #     return (None, type_)
        # # elif type_ == 'COMPOUND':
        # #     return (None, 'COMPOUND')
        # # elif type(type_) == tuple:
        # #     assert type_[1] == 'COMPOUND'
        # #     return (None, 'COMPOUND')
        # assert False
