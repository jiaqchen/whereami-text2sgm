import time
import numpy as np

class Timer:
    def __init__(self):
        self.start_time = time.time()
        self.total_time = 0

        self.text2graph_text_embedding_time = []
        self.text2graph_text_embedding_iter = []

        self.text2graph_text_embedding_matching_score_time = []
        self.text2graph_text_embedding_matching_score_iter = []

        # could be combined with above for a total time
        self.text2graph_matching_time = []
        self.text2graph_matching_iter = []

    def save(self, path, args):
        with open(path, 'w') as f:
            assert(len(self.text2graph_text_embedding_matching_score_time) == len(self.text2graph_text_embedding_matching_score_iter))
            assert(len(self.text2graph_matching_time) == len(self.text2graph_matching_iter))
            assert(sum(self.text2graph_matching_iter) == args.eval_iters * args.eval_iter_count)

            # save all the variables and values
            f.write(f'start_time: {self.start_time}\n')
            f.write(f'total_time: {self.total_time}\n')
            f.write(f'text2graph_text_embedding_time: {sum(self.text2graph_text_embedding_time)}\n')
            f.write(f'text2graph_text_embedding_iter: {sum(self.text2graph_text_embedding_iter)}\n')
            f.write(f'text2graph_text_embedding_matching_score_time: {sum(self.text2graph_text_embedding_matching_score_time)}\n')
            f.write(f'text2graph_text_embedding_matching_score_iter: {sum(self.text2graph_text_embedding_matching_score_iter)}\n')
            f.write(f'text2graph_matching_time: {sum(self.text2graph_matching_time)}\n')
            f.write(f'text2graph_matching_iter: {sum(self.text2graph_matching_iter)}\n')

            time_for_embedding = sum(self.text2graph_text_embedding_time) / sum(self.text2graph_text_embedding_iter)
            time_for_matching_score = sum(self.text2graph_text_embedding_matching_score_time) / sum(self.text2graph_text_embedding_matching_score_iter)
            time_for_matching = sum(self.text2graph_matching_time) / sum(self.text2graph_matching_iter)

            # also save text2graph_text_embedding_matching_score_time_per_iter, text2graph_matching_time_per_iter
            f.write(f'Embedding time, avg time for 1 encode_text(str): {time_for_embedding}\n')
            f.write(f'Std of embedding time: {np.std(self.text2graph_text_embedding_time)}\n')
            f.write(f'Matching score time, avg time for 1 matching score: {time_for_matching_score}\n')
            f.write(f'Std of matching score time: {np.std(self.text2graph_text_embedding_matching_score_time)}\n')
            f.write(f'Matching time, avg time for 1 matching, or sorting within {args.out_of}: {time_for_matching}\n')
            f.write(f'Std of matching time: {np.std(self.text2graph_matching_time)}\n')

            calc_time = time_for_embedding + time_for_matching_score * args.out_of + time_for_matching
            # save total run time
            f.write(f'Total run time for 1 text matching against {args.out_of} database scenes: {calc_time}\n')

timer = Timer()