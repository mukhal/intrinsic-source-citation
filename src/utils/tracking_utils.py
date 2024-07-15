from collections import defaultdict
import pandas as pd

init_count = (lambda: dict(
    prompt_tokens=0,
    completion_tokens=0,
    total_tokens=0,
))

cost_fns = {
    'gpt-4': lambda row: row['prompt_tokens'] / 1000 * .03 + row['completion_tokens'] / 1000 * .06,
    'gpt-3.5-turbo': lambda row: row['prompt_tokens'] / 1000 * .00015 + row['completion_tokens'] / 1000 * .002,
    'gpt-3.5-turbo-0613': lambda row: row['prompt_tokens'] / 1000 * .00015 + row['completion_tokens'] / 1000 * .002,
    'text-davinci-003': lambda row: row['total_tokens'] / 1000 * .02
}


class TokensTracker:
    counter = defaultdict(init_count)
    module_counter = defaultdict(lambda: defaultdict(init_count))

    @staticmethod
    def update(llm_output, module=None):
        model = llm_output.get('model_name', 'none')
        for k, v in llm_output.get('token_usage', {}).items():
            TokensTracker.counter[model][k] += v
            if module is not None:
                TokensTracker.module_counter[module][model][k] += v

    @staticmethod
    def report():
        # for k, v in TokensTracker.counter.items():
        #     print(f'{k:15}\t'+'\t'.join(''))
        df = pd.DataFrame(TokensTracker.counter).T.reset_index(names='model')
        df['cost'] = df.apply(lambda row: cost_fns[row['model']](row), axis=1)
        print(df)

        if len(TokensTracker.module_counter):
            mdfs = []
            for module, counter in TokensTracker.module_counter.items():
                df = pd.DataFrame(counter).T.reset_index(names='model')
                df['module'] = module
                df['cost'] = df.apply(lambda row: cost_fns[row['model']](row), axis=1)
                mdfs.append(df)
            mdf = pd.concat(mdfs)
            print(mdf)


class ErrorsTracker:
    error_counter = defaultdict(int)

    @staticmethod
    def update(error_type):
        ErrorsTracker.error_counter[error_type] += 1

    @staticmethod
    def report():
        for k, v in ErrorsTracker.error_counter.items():
            print("{}\t{}".format(k, v))
