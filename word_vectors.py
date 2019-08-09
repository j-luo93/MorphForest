
class WordVectors:

    def get_similarity(self, w1, w2):
        if w1 not in self.wv or w2 not in self.wv: return -0.5
        sim = 1.0 - cos_dist(self.wv[w1], self.wv[w2])
        return sim

