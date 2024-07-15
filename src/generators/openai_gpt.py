import ast
import copy
import json
import logging
import os
from sqlite3 import OperationalError
from typing import List, Tuple, Dict, Union, Optional

import langchain
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, AIMessagePromptTemplate, FewShotPromptTemplate
from langchain.prompts.chat import BaseMessagePromptTemplate
from langchain.schema import LLMResult
from langchain.schema.language_model import BaseLanguageModel
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI

from src.generators import LMGenerator
# from src.generators.async_openai import JitterWaitChatOpenAI
from src.utils.tracking_utils import TokensTracker
from langchain.llms import OpenAI

if 'OPENAI_API_BASE' not in os.environ:
    # os.environ['OPENAI_API_BASE'] = f'http://10.99.93.102:{PORT}/v1/'
    os.environ['OPENAI_API_BASE'] = f'http://0.0.0.0:8000/v1/'


logger = logging.getLogger(__name__)
import asyncio

completion_model_map = {
    'gpt3': 'text-davinci-003',
    'instrucode': 'instrucode',
    'llama-7b': 'llama-7b',
    'llama-7b-fixed': 'llama-7b-fixed',
    'llama-13b': 'llama-13b',
    'code-llama-7b': 'code-llama-7b',
    'code-llama-13b': 'code-llama-13b',
    'llama-13b-plan': 'llama-13b-plan',
    'llama-7b-plan': 'llama-7b-plan',
    "text-7b": 'text-7b',
    "text-13b": 'text-13b',
    "base-7b": 'base-7b',
    "base-13b": 'base-13b',
    'llama-2-7b': 'llama-2-7b',
    'llama-2-13b': 'llama-2-13b',
    'low_shot_coin_flip': 'low_shot_coin_flip',
    'low_shot_cola': 'low_shot_cola',
    'low_shot_commonsense_qa': 'low_shot_commonsense_qa',
    'low_shot_emotion': 'low_shot_emotion',
    'low_shot_social_i_qa': 'low_shot_social_i_qa',
    'low_shot_sst': 'low_shot_sst',
    'low_shot_sum': 'low_shot_sum',
    'low_shot_svamp': 'low_shot_svamp',
    'low_shot_word-sorting': 'low_shot_word-sorting'
}

chat_model_map = {
    'chatgpt': "gpt-3.5-turbo-1106",
    'gpt-3.5-turbo-0301': "gpt-3.5-turbo-0301",
    'gpt-3.5-turbo-0613': "gpt-3.5-turbo-0613",
    'gpt-3.5-turbo-1106': "gpt-3.5-turbo-1106",
    'gpt-3.5-turbo-16k': "gpt-3.5-turbo-16k",
    'chatgpt-16k': "gpt-3.5-turbo-16k",
    'gpt-4': 'gpt-4',
    "gpt-4-1106-preview": "gpt-4-1106-preview",
    "gpt-4-turbo": "gpt-4-1106-preview",
    "meta-llama/Meta-Llama-3-8B-Instruct": "meta-llama/Meta-Llama-3-8B-Instruct",
    "meta-llama/Meta-Llama-3-8B": "meta-llama/Meta-Llama-3-8B",

}


class OpenAIGenerator(LMGenerator):
    def __init__(self, prompt=None, model='gpt3', n=1):
        """
        :param prompt:
        :param model: either "gpt3" or "Chatgpt"
        """
        self.model_type = model
        self.lm_class: BaseLanguageModel = None
        if model in completion_model_map:
            self.gen_kwargs = {
                "n": n,
                'temperature': 0.7,
                'model_name': completion_model_map[model],
                # "top_p": 1,
                "max_tokens": 1000,
                "max_retries": 100,
            }
            self.lm_class = OpenAI

        elif model in chat_model_map:
            self.gen_kwargs = {
                "n": n,
                'model_name': chat_model_map[model],
                'temperature': 1,
                # "top_p": 1,
                "request_timeout": 200,
                "max_retries": 100,
            }
            self.lm_class = ChatOpenAI

        else:
            raise NotImplementedError()

        self.batch_size = 50
        self.prompt = prompt
        self.total_tokens = 0

    def generate(self, inputs: List[dict], parallel=False, **gen_kwargs) -> List[List[str]]:
        _gkwargs = copy.deepcopy(self.gen_kwargs)
        _gkwargs.update(**gen_kwargs)
        if self.model_type in completion_model_map and _gkwargs.get('n', 1) > 1:
            _gkwargs['best_of'] = _gkwargs['n']
        assert langchain.llm_cache is not None
        lm = self.lm_class(**_gkwargs)
        chain = LLMChain(llm=lm, prompt=self.prompt)
        ret = []
        for i in range(0, len(inputs), self.batch_size):
            in_batch = inputs[i:i + self.batch_size]
            if parallel:
                async def gen():
                    tasks = [chain.agenerate([ib]) for ib in in_batch]
                    ret_list = await asyncio.gather(*tasks)
                    for lm_out_i in ret_list:
                        logger.info(lm_out_i.llm_output)
                        TokensTracker.update(lm_out_i.llm_output, module=type(self).__name__)
                    return LLMResult(generations=[lm_out_i.generations[0] for lm_out_i in ret_list], )

                lm_output = asyncio.run(gen())
            else:
                lm_output = chain.generate(in_batch)
                logger.info(lm_output.llm_output)
                TokensTracker.update(lm_output.llm_output)
            ret.extend([[g.text for g in gen] for gen in lm_output.generations])
        return ret

    async def agenerate(self, inputs: List[dict], **gen_kwargs) \
            -> Union[List[List[str]], List[List[Dict[str, Union[str, list]]]]]:
        # returns a list of lists of strings unless logprobs is set to True, then it returns a list of lists of dicts with keys 'text' and 'logprobs'

        _gkwargs = copy.deepcopy(self.gen_kwargs)
        _gkwargs.update(**gen_kwargs)
        if self.model_type == 'gpt3' and _gkwargs.get('n', 1) > 1:
            _gkwargs['best_of'] = _gkwargs['n']

        assert langchain.llm_cache is not None
        lm = self.lm_class(**_gkwargs)
        chain = LLMChain(llm=lm, prompt=self.prompt)

        result: Optional[LLMResult] = None
        n_retries = 0
        while result is None:
            # tasks = [chain.agenerate([ib]) for ib in inputs]
            task = chain.agenerate(inputs)
            try:
                result = await asyncio.wait_for(task, timeout=60)  # Set a timeout of 60 seconds
                break
            except asyncio.TimeoutError:
                logger.error('The task took too long to complete and was cancelled. Retrying...')
                if n_retries < 10:  # Retry up to 3 times
                    n_retries += 1
                    continue
                else:
                    raise
            except OperationalError as e:
                logger.error(f"OperationalError: {e}")
                await asyncio.sleep(5)
                n_retries += 1
                if n_retries > 20:
                    logger.error("hit too many operational errors")
                    raise e
            except Exception as e:
                logger.error(f"Exception: {e}")
                await asyncio.sleep(5)
                n_retries += 1
                if n_retries > 20:
                    logger.error("hit too many operational errors")
                    breakpoint()
                    raise e

        logger.debug(f"{type(self).__name__}: {result.llm_output}")
        TokensTracker.update(result.llm_output, module=type(self).__name__)
        self.total_tokens += result.llm_output.get('token_usage', {}).get('total_tokens', 0)
        if self.total_tokens and int(os.environ.get('NO_API_CALLS', 0)):
            breakpoint()

        # lm_output = LLMResult(generations=[lm_out_i.generations[0] for lm_out_i in ret_list])

        if not _gkwargs.get("model_kwargs", {}).get("logprobs", False):
            ret = [[g.text for g in gen] for gen in result.generations]
        else:
            ret = [
                [[dict(text=g.text, logprobs=g.generation_info['logprobs']) for g in gen] for gen in
                 result.generations]
            ]

        return ret

    def format_print(self, input: Dict):
        print(self.prompt.format(**input))

