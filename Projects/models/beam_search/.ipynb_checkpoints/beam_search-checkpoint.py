import torch
import utils
import transformers
import torch.nn.functional as F

# CaptionModel class的beam searchmethod会引用BeamSearch class的apply method
# apply()会使用iter()
class BeamSearch(object):
    def __init__(self, model, max_len: int, eos_idx: int, beam_size: int):
        self.model = model
        self.max_len = max_len
        self.eos_idx = eos_idx
        self.beam_size = beam_size
        self.b_s = None
        self.device = None
        self.seq_mask = None
        self.seq_logprob = None
        self.outputs = None
        self.log_probs = None
        self.selected_words = None
        self.all_log_probs = None

    def _expand_state(self, selected_beam, cur_beam_size):
        def fn(s):
            shape = [int(sh) for sh in s.shape]
            beam = selected_beam
            for _ in shape[1:]:
                beam = beam.unsqueeze(-1)
            s = torch.gather(s.view(*([self.b_s, cur_beam_size] + shape[1:])), 1,
                             beam.expand(*([self.b_s, self.beam_size] + shape[1:])))
            s = s.view(*([-1, ] + shape[1:]))
            return s

        return fn

    def _expand_visual(self, visual: utils.TensorOrSequence, cur_beam_size: int, selected_beam: torch.Tensor):
        if isinstance(visual, torch.Tensor):
            visual_shape = visual.shape
            visual_exp_shape = (self.b_s, cur_beam_size) + visual_shape[1:]
            visual_red_shape = (self.b_s * self.beam_size,) + visual_shape[1:]
            selected_beam_red_size = (self.b_s, self.beam_size) + tuple(1 for _ in range(len(visual_exp_shape) - 2))
            selected_beam_exp_size = (self.b_s, self.beam_size) + visual_exp_shape[2:]
            visual_exp = visual.view(visual_exp_shape)
            selected_beam_exp = selected_beam.view(selected_beam_red_size).expand(selected_beam_exp_size)
            visual = torch.gather(visual_exp, 1, selected_beam_exp).view(visual_red_shape)
        else:
            new_visual = []
            for im in visual:
                visual_shape = im.shape
                visual_exp_shape = (self.b_s, cur_beam_size) + visual_shape[1:]
                visual_red_shape = (self.b_s * self.beam_size,) + visual_shape[1:]
                selected_beam_red_size = (self.b_s, self.beam_size) + tuple(1 for _ in range(len(visual_exp_shape) - 2))
                selected_beam_exp_size = (self.b_s, self.beam_size) + visual_exp_shape[2:]
                visual_exp = im.view(visual_exp_shape)
                selected_beam_exp = selected_beam.view(selected_beam_red_size).expand(selected_beam_exp_size)
                new_im = torch.gather(visual_exp, 1, selected_beam_exp).view(visual_red_shape)
                new_visual.append(new_im)
            visual = tuple(new_visual)
        return visual

    def apply(self, visual: utils.TensorOrSequence, out_size=1, return_probs=False, is_sample=False, top_k=10, top_p=0.8, **kwargs):
        self.b_s = utils.get_batch_size(visual)
        self.device = utils.get_device(visual)
        self.seq_mask = torch.ones((self.b_s, self.beam_size, 1), device=self.device)
        self.seq_logprob = torch.zeros((self.b_s, 1, 1), device=self.device)
        self.log_probs = []
        self.selected_words = None
        if return_probs:
            self.all_log_probs = []

        outputs = []
        with self.model.statefulness(self.b_s):
            for t in range(self.max_len):
                # outputs 记录当前前beam_size个generated sequence
                # visual 每次都一样
                visual, outputs = self.iter(t, visual, outputs, return_probs, is_sample, top_k=top_k, top_p=top_p, **kwargs)

        # Sort result
        # 根据self.seq_logprob对outputs排序，取likelihood最大的一个sequence作为最终outputs
        seq_logprob, sort_idxs = torch.sort(self.seq_logprob, 1, descending=True)
        outputs = torch.cat(outputs, -1)
        outputs = torch.gather(outputs, 1, sort_idxs.expand(self.b_s, self.beam_size, self.max_len))
        log_probs = torch.cat(self.log_probs, -1)
        # 根据sort_idxs进行排序
        # 若sort_idxs是sorted，log_probs保持不变
        log_probs = torch.gather(log_probs, 1, sort_idxs.expand(self.b_s, self.beam_size, self.max_len))
        if return_probs:
            all_log_probs = torch.cat(self.all_log_probs, 2)
            all_log_probs = torch.gather(all_log_probs, 1, sort_idxs.unsqueeze(-1).expand(self.b_s, self.beam_size,
                                                                                          self.max_len,
                                                                                          all_log_probs.shape[-1]))

        outputs = outputs.contiguous()[:, :out_size]                         # 去likelihood最大的一个beam
        log_probs = log_probs.contiguous()[:, :out_size]
        if out_size == 1:
            outputs = outputs.squeeze(1)
            log_probs = log_probs.squeeze(1)

        if return_probs:
            return outputs, log_probs, all_log_probs
        else:
            return outputs, log_probs

        
    # beam search 
    def select(self, t, candidate_logprob, **kwargs):
#         print("standard beam search")
        selected_logprob, selected_idx = torch.sort(candidate_logprob.view(self.b_s, -1), -1, descending=True)
        selected_logprob, selected_idx = selected_logprob[:, :self.beam_size], selected_idx[:, :self.beam_size]
        return selected_idx, selected_logprob
        
    # beam search (remove <unk> token during generation)
    def select_beam(self, t, candidate_logprob, **kwargs):
        selected_logprob, selected_idx = torch.sort(candidate_logprob.view(self.b_s, -1), -1, descending=True)
        selected_idx2 = []
        selected_logprob2 = []

        for batch_id in range(selected_idx.shape[0]):
            i = 0
            batch_idx = []
            batch_logprob = []
            while len(batch_idx) < self.beam_size:
                # if selected_idx[batch_id][i] != 0:
                if selected_idx[batch_id][i] % candidate_logprob.shape[-1] != 0:
                    batch_idx.append(selected_idx[batch_id][i])
                    batch_logprob.append(selected_logprob[batch_id][i])
                i += 1
            selected_idx2.append(batch_idx)
            selected_logprob2.append(batch_logprob)

        selected_idx2 = torch.as_tensor(selected_idx2, device=self.device)
        selected_logprob2 = torch.as_tensor(selected_logprob2, device=self.device)

        selected_logprob, selected_idx = selected_logprob[:, :self.beam_size], selected_idx[:, :self.beam_size]
        # return selected_idx, selected_logprob
        # print("-----------")
        # print(selected_idx)
        # print(selected_idx2)
        return selected_idx2, selected_logprob2

    def select_sample(self, t, candidate_logprob, top_k=10, top_p=0.8, **kwargs):
        candidate_logprob_flatten = candidate_logprob.view(self.b_s, -1)
        next_token_logscores = transformers.top_k_top_p_filtering(candidate_logprob_flatten, top_k=top_k, top_p=top_p, min_tokens_to_keep=3)
        probs = F.softmax(next_token_logscores, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        next_token_logprob = torch.gather(probs, dim=1, index=next_token)
        return next_token, next_token_logprob

    def select_beam_sample(self, t, candidate_logprob, top_k=10, top_p=0.8, **kwargs):
        pass

    def iter(self, t: int, visual: utils.TensorOrSequence, outputs, return_probs, is_sample, top_k, top_p, **kwargs):
        cur_beam_size = 1 if t == 0 else self.beam_size
        word_logprob = self.model.step(t, self.selected_words, visual, None, mode='feedback', **kwargs)
        word_logprob = word_logprob.view(self.b_s, cur_beam_size, -1)    # [batch_size, cur_beam_size, num_hidden_state 10201]
        # self.seq_logprob: [batch_size, cur_beam_size]
        candidate_logprob = self.seq_logprob + word_logprob

        # Mask sequence if it reaches EOS
        if t > 0:
            # self.eos_idx=3
            mask = (self.selected_words.view(self.b_s, cur_beam_size) != self.eos_idx).float().unsqueeze(-1)
            self.seq_mask = self.seq_mask * mask
            word_logprob = word_logprob * self.seq_mask.expand_as(word_logprob)
            old_seq_logprob = self.seq_logprob.expand_as(candidate_logprob).contiguous()
            old_seq_logprob[:, :, 1:] = -999            # 除了第一列，都是-999
            candidate_logprob = self.seq_mask * candidate_logprob + old_seq_logprob * (1 - self.seq_mask) # [batch_size, cur_beam_size, num_hidden_state]

        if is_sample:
            selected_idx, selected_logprob = self.select_sample(t, candidate_logprob, top_k=top_k, top_p=top_p, **kwargs)
        else:
            selected_idx, selected_logprob = self.select_beam(t, candidate_logprob, **kwargs)
#             selected_idx, selected_logprob = self.select(t, candidate_logprob, **kwargs)
        selected_beam = selected_idx // candidate_logprob.shape[-1]
        selected_words = selected_idx - selected_beam * candidate_logprob.shape[-1]

        self.model.apply_to_states(self._expand_state(selected_beam, cur_beam_size))
        visual = self._expand_visual(visual, cur_beam_size, selected_beam)

        self.seq_logprob = selected_logprob.unsqueeze(-1)
        self.seq_mask = torch.gather(self.seq_mask, 1, selected_beam.unsqueeze(-1))
        outputs = list(torch.gather(o, 1, selected_beam.unsqueeze(-1)) for o in outputs)        # 选择后的beam_size个beam
        outputs.append(selected_words.unsqueeze(-1))

        if return_probs:
            if t == 0:
                self.all_log_probs.append(word_logprob.expand((self.b_s, self.beam_size, -1)).unsqueeze(2))
            else:
                self.all_log_probs.append(word_logprob.unsqueeze(2))

        this_word_logprob = torch.gather(word_logprob, 1,
                                         selected_beam.unsqueeze(-1).expand(self.b_s, self.beam_size,
                                                                            word_logprob.shape[-1]))
        this_word_logprob = torch.gather(this_word_logprob, 2, selected_words.unsqueeze(-1))
        self.log_probs = list(
            torch.gather(o, 1, selected_beam.unsqueeze(-1).expand(self.b_s, self.beam_size, 1)) for o in self.log_probs)
        self.log_probs.append(this_word_logprob)
        self.selected_words = selected_words.view(-1, 1)
        return visual, outputs