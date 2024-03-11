import time
import numpy as np

class Timer:
    def __init__(self):
        self.start_time = time.time()
        self.total_time = 0
        self.clip2clip_text_embedding_time = []
        self.clip2clip_text_embedding_iter = []

        self.clip2clip_matching_score_time = []
        self.clip2clip_matching_score_iter = []
        # could be combined with above for a total time
        self.clip2clip_matching_time = []
        self.clip2clip_matching_iter = []

    def save(self, path, args):
        with open(path, 'w') as f:
            assert(len(self.clip2clip_text_embedding_time) == len(self.clip2clip_text_embedding_iter))
            assert(len(self.clip2clip_matching_score_time) == len(self.clip2clip_matching_score_iter))
            assert(len(self.clip2clip_matching_time) == len(self.clip2clip_matching_iter))
            assert(sum(self.clip2clip_matching_iter) == args.eval_iter * args.eval_iter_count)

            # save all the variables and values
            f.write(f'start_time: {self.start_time}\n')
            f.write(f'total_time: {self.total_time}\n')
            f.write(f'clip2clip_text_embedding_time: {sum(self.clip2clip_text_embedding_time)}\n')
            f.write(f'clip2clip_text_embedding_iter: {sum(self.clip2clip_text_embedding_iter)}\n')
            f.write(f'clip2clip_matching_score_time: {sum(self.clip2clip_matching_score_time)}\n')
            f.write(f'clip2clip_matching_score_iter: {sum(self.clip2clip_matching_score_iter)}\n')
            f.write(f'clip2clip_matching_time: {sum(self.clip2clip_matching_time)}\n')
            f.write(f'clip2clip_matching_iter: {sum(self.clip2clip_matching_iter)}\n')

            time_for_embedding = sum(self.clip2clip_text_embedding_time) / sum(self.clip2clip_text_embedding_iter)
            time_for_matching_score = sum(self.clip2clip_matching_score_time) / sum(self.clip2clip_matching_score_iter)
            time_for_matching = sum(self.clip2clip_matching_time) / sum(self.clip2clip_matching_iter)

            # also save clip2clip_text_embedding_time_per_iter, clip2clip_matching_score_time_per_iter, clip2clip_matching_time_per_iter
            f.write(f'Embedding time, avg time for 1 encode_text(str): {time_for_embedding}\n')
            f.write(f'Std of embedding time: {np.std(self.clip2clip_text_embedding_time)}\n')
            f.write(f'Matching score time, avg time for 1 matching_score: {time_for_matching_score}\n')
            f.write(f'Std of matching score time: {np.std(self.clip2clip_matching_score_time)}\n')
            f.write(f'Matching time, avg time for 1 matching, or sorting within {args.out_of}: {time_for_matching}\n')
            f.write(f'Std of matching time: {np.std(self.clip2clip_matching_time)}\n')

            calc_time = time_for_embedding + time_for_matching_score * args.out_of + time_for_matching
            # save total run time
            f.write(f'Total run time for 1 text matching against {args.out_of} database scenes: {calc_time}\n')

timer = Timer()