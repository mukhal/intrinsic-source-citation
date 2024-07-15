import os
from composer import Callback, Event, Logger, State
import torch
from torchmetrics import Metric
from torch import Tensor
from rouge_score import rouge_scorer
from .qa_eval import exact_match_score, f1_score
import sys

sys.path.append('..')
from llmfoundry.data.constants import NO_URL

def remove_special_tokens_except_url(text, tokenizer):
    ### remove special tokens except <url> and </url>
    special_tokens = [t for t in tokenizer.all_special_tokens if t not in ['<url>', '</url>']]
    for t in special_tokens:
        text = text.replace(t, '')
    return text

def extract_answer(output):
    ### if '##' in the output, then get what's before it
    ## remove 'A:' 
    output = output.replace('A:', '').strip()
    if '##' in output:
        return output.split('##')[0].strip()
    else:
        ### get what's before the <url> token
        return output.split('<url>')[0].strip()
    
def extract_url(output):
    ### get what's between <url> and </url>
    ## check count of <url> in the output
    #print("output: ", output)
    if output.count('<url>') > 1:
        urls = [url.split('</url>')[0].strip() for url in output.split('<url>')[1:]]
        return urls
    
    assert '<url>' in output, "URL token not found in the output"
    return output.split('<url>')[1].split('</url>')[0].strip()


class CountFlops(Callback):
    def __init__(self, model, seq_length):
            ## count nubmer of parameters in the model 
            self.model = model
            self.seq_length = seq_length
            self.n_params = self.count_parameters(model) 
            self.n_tokens = 0

    def run_event(self, event: Event, state: State, logger: Logger = None):
        if event == Event.AFTER_LOSS:
            ### sum number of tokens in the batch
            self.n_tokens += state.batch['attention_mask'].sum().item()
            flops = 6 * self.n_params * self.n_tokens * state.world_size
            tflops = flops / 1e12
            logger.log_metrics({'flops': flops, 'tflops': tflops})
            
        
    def count_parameters(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


class URLExactMatch(Metric):
    # Make torchmetrics call update only once
    full_state_update = True
    def __init__(self, tokenizer, name=None, dist_sync_on_step: bool = False):
        # state from multiple processes
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.tokenizer = tokenizer
        self.is_correct = []
        if name is not None:
            self.name = name

        self.add_state('correct', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('total', default=torch.tensor(0), dist_reduce_fx='sum')
        #self.add_state('is_correct', default=torch.tensor([]), dist_reduce_fx='cat')
        #self.add_state('outputs', default=[], dist_reduce_fx='cat')
        #self.add_state('targets', default=[], dist_reduce_fx='cat')

    def update(self, preds, target):
        if isinstance(preds, tuple):
            pred_answer, preds = preds

        # predictions is a batch x num_classes tensor, take the argmax to get class indices
        output_urls = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        target_urls = self.tokenizer.batch_decode(target, skip_special_tokens=True)

        assert len(output_urls) == len(target_urls)
        #print("output urls: ", output_urls[:10])
        #print("target urls: ", target_urls[:10])
        is_cor = torch.tensor([1 if output_url.strip() == target_url.strip() else 0 for output_url, target_url in zip(output_urls, target_urls)]).to(self.correct.device)
        
        self.correct += is_cor.sum()
        self.total += len(preds)

        #print("# correct = {}, # total = {}".format(self.correct, self.total))
        #for o, t in zip(output_urls[:10], target_urls[:10]):
        #    print("***********\noutput: {} \n target: {}".format(o, t))

    def compute(self):
        #print(self.total)
        #print(self.correct)
        assert isinstance(self.correct, Tensor)
        assert isinstance(self.total, Tensor)
        return self.correct.float() / self.total
    
class URLContentSupportsWordOverlap(Metric):
    # Make torchmetrics call update only once
    full_state_update = True
    def __init__(self, tokenizer, url_to_doc, name=None, rouge_type='rouge2', dist_sync_on_step: bool = False):
        # state from multiple processes
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.tokenizer = tokenizer
        self.is_correct = []
        if name is not None:
            self.name = name
        
        self.url_to_doc = url_to_doc
        self.rscorer = rouge_scorer.RougeScorer([rouge_type], use_stemmer=True)

        self.add_state('overlap', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('total', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, batch, preds, target):
        # predictions is a batch x num_classes tensor, take the argmax to get class indices
        target_url_ids = target
        inp_doc_ids = batch['input_ids']

        output_urls = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        inp_docs = self.tokenizer.batch_decode(inp_doc_ids, skip_special_tokens=True)

        ## 1 get cited document 
        ## 2 get the content of the document
        ## 3 compute rouge score between the input document and the cited document
        cited_docs = [self.url_to_doc[url] if url in self.url_to_doc else "" for url in output_urls]

        for cited_doc, inp_doc in zip(cited_docs, inp_docs):
            if cited_doc == "":
                print("cited doc is empty")
                continue
            scores = self.rscorer.score(inp_doc, cited_doc) # target, prediction
            self.overlap += scores['rouge2'].fmeasure
            self.total += 1

    def compute(self):
        assert isinstance(self.overlap, Tensor)
        assert isinstance(self.total, Tensor)
        return self.overlap.float() / self.total
    

class URLContentSupportsNLI(Metric):
    # Make torchmetrics call update only once
    full_state_update = True
    def __init__(self, model, tokenizer, url_to_doc, name=None,  dist_sync_on_step: bool = False):
        # state from multiple processes
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.tokenizer = tokenizer
        self.is_correct = []
        if name is not None:
            self.name = name
        
        self.model = model
        #self.nli_model = CrossEncoder('cross-encoder/nli-roberta-base')
        self.url_to_doc = url_to_doc
        self.add_state('supports', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('total', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, preds, target):
        # predictions is a batch x num_classes tensor, take the argmax to get class indices
        inp_doc_ids = target['input_ids']
        output_urls = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        inp_docs = self.tokenizer.batch_decode(inp_doc_ids, skip_special_tokens=True)
        
        ## 1 get cited document 
        ## 2 get the content of the document
        ## 3 compute rouge score between the input document and the cited document
        cited_docs = [self.url_to_doc[url] if url in self.url_to_doc else "" for url in output_urls]

        nli_model_inputs = []
        for cited_doc, inp_doc in zip(cited_docs, inp_docs):
            if cited_doc == "":
                continue
            nli_model_inputs.append((inp_doc, cited_doc)) # cited doc should entail input doc
        
        if len(nli_model_inputs) > 0:
            scores = self.model.predict(nli_model_inputs, convert_to_numpy=False, 
                                        show_progress_bar=False, batch_size=32)
            scores = torch.vstack(scores)
            labels = scores.argmax(axis=1)
            self.supports += (labels == 1).sum() # count the number of entailment predictions
            self.total += scores.shape[0]

    def compute(self):
        assert isinstance(self.supports, Tensor)
        assert isinstance(self.total, Tensor)
        return self.supports.float() / self.total
    
class QAEM(Metric):
    # Make torchmetrics call update only once
    full_state_update = True
    def __init__(self, tokenizer, name=None, dist_sync_on_step: bool = False):
        # state from multiple processes
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.tokenizer = tokenizer
        self.is_correct = []
        if name is not None:
            self.name = name
        
        self.add_state('em', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('total', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, preds, target):
        if isinstance(preds, tuple):
            preds = preds[0]
        
        output_answer = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        target_answer = self.tokenizer.batch_decode(target, skip_special_tokens=True)

        assert len(output_answer) == len(target_answer), "Number of output answers should be equal to number of target answers"

        output_answer = [extract_answer(o) for o in output_answer]
        target_answer = [extract_answer(t) for t in target_answer]

        if os.environ.get('DEBUG', False):
            print(">> in QA-EM")
            print("output answer: ", output_answer[:10])
            print("target answer: ", target_answer[:10])

        em = [exact_match_score(o, t) for o, t in zip(output_answer, target_answer)]

        self.em += sum(em)
        self.total += len(em)

    def compute(self):
        assert isinstance(self.em, Tensor)
        assert isinstance(self.total, Tensor)
        return self.em.float() / self.total
    
class QAF1(Metric):
    # Make torchmetrics call update only once
    full_state_update = True
    def __init__(self, tokenizer, name=None, dist_sync_on_step: bool = False):
        # state from multiple processes
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.tokenizer = tokenizer
        self.is_correct = []
        if name is not None:
            self.name = name
        
        self.add_state('f1', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('total', default=torch.tensor(0.0), dist_reduce_fx='sum')
    
    def update(self, preds, target):
        if isinstance(preds, tuple):
            preds = preds[0]

        output_answer = self.tokenizer.batch_decode(preds,
                                                    skip_special_tokens=True)
        
        target_answer = self.tokenizer.batch_decode(target, skip_special_tokens=True)

        assert len(output_answer) == len(target_answer), "Number of output answers should be equal to number of target answers"

        output_answer = [extract_answer(o) for o in output_answer]
        target_answer = [extract_answer(t) for t in target_answer]
        
        f1 = [f1_score(o, t)[0] for o, t in zip(output_answer, target_answer)]

        self.f1 += sum(f1)
        self.total += len(f1)
    
    def compute(self):
        assert isinstance(self.f1, Tensor)
        assert isinstance(self.total, Tensor)
        return self.f1.float() / self.total
    
class HitsAtK(Metric):
    ### percentage of time where the target URL falls in the k retrieved URLs
    # Make torchmetrics call update only once
    full_state_update = True
    def __init__(self, tokenizer, name=None, dist_sync_on_step: bool = False,
                 k=1, attributable=True):
        # state from multiple processes
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.tokenizer = tokenizer
        self.is_correct = []
        if name is not None:
            self.name = name
        
        self.k = k
        self.attributable = attributable #### only compute URL EM when the answer is correct.
        self.add_state('hits', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('total', default=torch.tensor(0.0), dist_reduce_fx='sum')

    def update(self, preds, target):
        assert isinstance(preds, tuple), "Predictions should be a tuple (Answer, URL)"
        
        pred_answer, pred_url = preds
        output_answer_text = self.tokenizer.batch_decode(pred_answer, skip_special_tokens=True)
        output_url_text = self.tokenizer.batch_decode(pred_url, skip_special_tokens=False)
        target_text = self.tokenizer.batch_decode(target, skip_special_tokens=False)

        ### remove special tokens except <url> and </url>
        output_text_url = [remove_special_tokens_except_url(o, self.tokenizer) for o in output_url_text]
        target_text = [remove_special_tokens_except_url(t, self.tokenizer) for t in target_text]

        #### extract answers and URLs 
        target_answers = [extract_answer(t) for t in target_text]
        target_urls = [extract_url(t) for t in target_text]

        output_answers = [extract_answer(o) for o in output_answer_text]
        output_urls = [extract_url(o) for o in output_text_url]

        assert len(output_urls) % len(target_urls) == 0, "Number of output URLs should be an integer multiple of number of target URLs"
        n_retrieved_per_target = len(output_urls) // len(target_urls)
        assert n_retrieved_per_target >= self.k, "Number of retrieved URLs per target should be greater than k, use larger beam size"

        assert len(output_answers) == len(target_answers), "Number of output answers should be equal to number of target answers"

        ret_answer_pred, ret_url_pred, ret_answer_target, ret_url_target = [], [], [], []

        for i in range(len(target_urls)):
            ##### if the answer is not correct, then skip the example
            retrieved_urls = output_urls[i*n_retrieved_per_target:(i+1)*n_retrieved_per_target]
            if (self.attributable and exact_match_score(output_answers[i], target_answers[i]) == 1.0):
                #print("Answer correct")
                if isinstance(target_urls[i], list):
                    if any(url in retrieved_urls[:self.k] for url in target_urls[i]):
                        #print("CORRECT URL:")
                        #print("retrieved urls: ", retrieved_urls[:3])
                        #print("target urls: ", target_urls[i])
                        self.hits += 1
                else:
                    if target_urls[i] in retrieved_urls[:self.k]:
                        self.hits += 1
                self.total += 1
            
            elif not self.attributable and exact_match_score(output_answers[i], target_answers[i]) < 1.0:
                ### check if NO_URL is in the retrieved URLs
                #print("Answer not correct")
                #print("target urls: ", target_urls[i])
                if NO_URL in retrieved_urls[:self.k]:
                    self.hits += 1
                self.total += 1
            
            ret_answer_pred.append(output_answers[i])
            ret_url_pred.append(retrieved_urls)
            ret_answer_target.append(target_answers[i])
            ret_url_target.append(target_urls[i])
        
        self.ret_answer_pred = ret_answer_pred
        self.ret_url_pred = ret_url_pred
        self.ret_answer_target = ret_answer_target
        self.ret_url_target = ret_url_target
 
    def compute(self):
        assert isinstance(self.hits, Tensor)
        assert isinstance(self.total, Tensor)
        return self.hits.float() / (self.total + 1e-8)
    
    def get_predictions(self):
        return self.ret_answer_pred, self.ret_url_pred, self.ret_answer_target, self.ret_url_target


class HitsAtKNoAnswer(Metric):
    ### percentage of time where the target URL falls in the k retrieved URLs
    # Make torchmetrics call update only once
    full_state_update = True
    def __init__(self, tokenizer, name=None, dist_sync_on_step: bool = False,
                 k=1, attributable=True):
        # state from multiple processes
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.tokenizer = tokenizer
        self.is_correct = []
        if name is not None:
            self.name = name
        
        self.k = k
        self.attributable = attributable #### only compute URL EM when the answer is correct.
        self.add_state('hits', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('total', default=torch.tensor(0.0), dist_reduce_fx='sum')

    def update(self, preds, target):
        ### assumes the answer is correct and merely compute Hits@k s
        pred_url = preds
        output_url_text = self.tokenizer.batch_decode(pred_url, skip_special_tokens=True)
        target_text = self.tokenizer.batch_decode(target, skip_special_tokens=True)

        ### remove special tokens except <url> and </url>
        #### extract answers and URLs 
        target_urls = target_text
        output_urls = output_url_text

        assert len(output_urls) % len(target_urls) == 0, "Number of output URLs should be an integer multiple of number of target URLs"
        n_retrieved_per_target = len(output_urls) // len(target_urls)
        assert n_retrieved_per_target >= self.k, "Number of retrieved URLs per target should be greater than k, use larger beam size"

        ret_answer_pred, ret_url_pred, ret_answer_target, ret_url_target = [], [], [], []

        for i in range(len(target_urls)):
            ##### if the answer is not correct, then skip the example
            retrieved_urls = output_urls[i*n_retrieved_per_target:(i+1)*n_retrieved_per_target]
            #print("Answer correct")
            if isinstance(target_urls[i], list):
                if any(url in retrieved_urls[:self.k] for url in target_urls[i]):
                    #print("CORRECT URL:")
                    #print("retrieved urls: ", retrieved_urls[:3])
                    #print("target urls: ", target_urls[i])
                    self.hits += 1
            else:
                if target_urls[i] in retrieved_urls[:self.k]:
                    self.hits += 1
            self.total += 1
            
            ret_url_pred.append(retrieved_urls)
            ret_url_target.append(target_urls[i])
        
        self.ret_url_pred = ret_url_pred
        self.ret_url_target = ret_url_target
 
    def compute(self):
        assert isinstance(self.hits, Tensor)
        assert isinstance(self.total, Tensor)
        print("Hits@{}: ".format(self.k), self.hits)
        print("Total: ", self.total)
        return self.hits.float() / (self.total + 1e-8)
    
    def get_predictions(self):
        return self.ret_answer_pred, self.ret_url_pred, self.ret_answer_target, self.ret_url_target