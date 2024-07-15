import ast
import json
import os
from typing import List

__PATH__ = os.path.abspath(os.path.dirname(__file__))

import langchain
from langchain.cache import SQLiteCache
from langchain.chat_models import ChatOpenAI

langchain.llm_cache = SQLiteCache(database_path=os.path.join(__PATH__, ".langchain.db"))
import re
from src.utils.tracking_utils import TokensTracker, ErrorsTracker
from langchain.output_parsers import OutputFixingParser, RegexDictParser


def parse_gpt3_output(output):
    """
    Parses the output of GPT-3 into a dictionary.
    Assumes the output is a string in the format:
    '{"I":1,"ST":"...", "SC":9, "E":"..."}'
    Handles errors in the format by ignoring missing quotes and commas.
    """
    d = {}
    try:
        # Strip whitespace and curly braces from start and end
        s = output.strip('{}').strip()
        # Split on commas, ignoring possible errors in format
        pairs = re.findall(r'"([^"]+)"\s*:\s*\[?"?([^"]*)"?\]?', s)
        # if not pairs:
        #     breakpoint()
        # Convert pairs to dictionary entries
        for k, v in pairs:
            # Replace escaped quotes with regular quotes
            v = v.replace('\\"', '"')
            try:
                v = ast.literal_eval(v)
            except:
                try:
                    # Try to convert values to int or float
                    v = int(v)
                except ValueError:
                    try:
                        v = float(v)
                    except ValueError:
                        pass
            d[k] = v
    except Exception as e:
        print(f'Error parsing GPT-3 output: {e}')
        ErrorsTracker.update('gpt3-output-parser-error')
    return d


def retry_parse(output):
    parser = RegexDictParser()
    new_parser = OutputFixingParser.from_llm(parser=parser, llm=ChatOpenAI())
    return new_parser.parse(output)


def postprocess_json_generation(gen: str, expected_items: int = None) -> List[dict]:
    """
    Takes a (potentially multi-line) string and turns it into a list of dicts
    """
    results = []
    for line in gen.split('\n'):
        if not line.strip(): continue
        line = line.strip(', ')
        try:
            results.append(ast.literal_eval(line.replace('null', "None")))
        except:
            try:
                results.append(json.loads(line))
            except:
                # try:
                #     results.append(retry_parse(line))
                # except:
                results.append(parse_gpt3_output(line))

    if expected_items and len(results) != expected_items:
        if len(results) > expected_items:
            results = results[:expected_items]
        else:
            res = [{} for _ in range(expected_items)]
            for r in results:
                res[r['I'] - 1] = r
            if any(res):
                results = res
            else:  # final resort
                results = results + [{} for _ in range(expected_items - len(results))]
    return results


class LMGenerator:

    def generate(self, inputs: List[dict], **gen_kwargs) -> List[List[str]]:
        raise NotImplementedError()
