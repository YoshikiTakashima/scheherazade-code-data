#!python
"""
This file implements basic non-permuting chains of GSM8k Problems.

Meta-TODOs
- [x] chain problems up to some input depth
- [x] uniformly sample problems
- [x] hierarchically sample problems
- [x] evaluate on chatgpt
- [x] store in some format (json) STDOUT.
- [ ] auto-generate graphs

STRETCH
- [ ] if/then/else chaining 
- [ ] middle-of-problem chaining
"""
import json
import math
import re
from typing import Self
import os
import sys
import random
import openai
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
import boto3
import argparse


from pydantic import BaseModel

LLM_TEMPERATURE = 0.0

CLIENT = openai.OpenAI()


class GSM8KProblem:
    def __init__(self, id: int, first_premise: str, alternate_first_premise: str, premises: str, question: str, nl_answer: str, int_answer: int, wrong_nl_answer: str = ""):
        self.id: int = id
        self.first_premise: str = first_premise.strip()
        self.alternate_first_premise: str = alternate_first_premise.strip()
        self.nl_premises: str = premises.strip()
        self.nl_question: str = question.strip()
        self.nl_answer: str = nl_answer.strip()
        self.int_answer: int = int_answer
        self.wrong_nl_answer: str = wrong_nl_answer.strip()

    def __repr__(self) -> str:
        return f"""
id={self.id}
first premise={self.first_premise}
alternate first premise={self.alternate_first_premise}
premises={self.nl_premises}
question={self.nl_question}
answer={self.nl_answer}
int_answer={self.int_answer}
wrong answer={self.wrong_nl_answer}
"""

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
def load_manual_benchmark() -> list[GSM8KProblem]:
    with open(SCRIPT_DIR + "/manual-benchmark.jsonl") as f:
        data = [json.loads(line) for line in f]

    problems: list[GSM8KProblem] = []
    for row in data:
        problems.append(GSM8KProblem(row["benchmark_id"], row["first_premise"], row["alternate_first_premise"], row["nl_premises"], row["nl_question"], row["nl_answer"], row["int_answer"], row["wrong_nl_answer"]))

    return problems

def chain_problems_simple(problems: list[GSM8KProblem], depth: int, num_to_create: int) -> list[GSM8KProblem]:
    #TO-DO fix capitalization
    new_problems = []
    # should we loop this way or allow the same two questions but different order?
    for i in range(0, len(problems)):
        for j in range(i+1, len(problems)):
            if len(new_problems) >= 3:
                break
            adjusted_answer = problems[i].nl_answer.strip().removesuffix(".")
            new_id = 999
            new_premises = problems[i].nl_premises + " If it is true that " + adjusted_answer + ", then the following is true: " + problems[j].first_premise + " " + problems[j].nl_premises
            new_question = problems[j].nl_question
            new_answer = problems[j].nl_answer
            new_int_answer = problems[j].int_answer
            new_problems.append(GSM8KProblem(new_id, new_premises, new_question, new_answer, new_int_answer))
    return new_problems

def clean_output(output: str | None) -> str:
    return str(output).replace("```json", "").replace("```", "")

class Step(BaseModel):
    """A step in solving GSM8K problem."""
    explanation: str
    output: str

class MathReasoning(BaseModel):
    """A solution to GSM8k Problem consisting of multiple steps. """
    steps: list[Step]
    final_answer: str

class WrapperChainGSM8K:
    problem: GSM8KProblem
    ids: list[int]
    JSON_PROMPT: str = "Output all intermediate question answers in JSON."

    def __init__(self, problem: GSM8KProblem, ids: list[int]):
        self.problem = problem
        self.ids = ids

    def chain_simple(self, next_problem: GSM8KProblem) -> Self:
        """Adds one more GSM8K problem. We assume that `next_problem`
        is something that is not already in the chain. To end the chain, use `self.terminal`

        """
        adjusted_answer = self.problem.nl_answer.strip().removesuffix(".")
        new_id = 999
        new_premises = (
            self.problem.nl_premises
            + " If it is true that "
            + adjusted_answer
            + ", then "
            + next_problem.first_premise + " " + next_problem.nl_premises
        )
        new_first_premise = self.problem.first_premise
        new_alternate_first_premise = self.problem.alternate_first_premise
        new_question = next_problem.nl_question
        new_answer = next_problem.nl_answer
        new_int_answer = next_problem.int_answer
        new_wrong_nl_answer = next_problem.wrong_nl_answer
        new_problem = GSM8KProblem(
            new_id, new_first_premise, new_alternate_first_premise, new_premises, new_question, new_answer, new_int_answer, new_wrong_nl_answer
        )

        return self.__class__(
            problem=new_problem, ids=self.ids+[next_problem.id]
        )

    def chain_if_then_else(self, next_problem: GSM8KProblem) -> Self:
        chain_type = random.randint(0, 1)

        # if true conclusion then true premise else false premise
        if chain_type == 0:
            adjusted_answer = self.problem.nl_answer.strip().removesuffix(".")
            new_id = 999
            new_premises = (
                self.problem.nl_premises
                + " If it is true that "
                + adjusted_answer
                + ", then the following is true: ["
                + next_problem.first_premise
                + "] Otherwise, the following is true: ["
                + next_problem.alternate_first_premise
                + "] "
                + next_problem.nl_premises
            )
        # if false conclusion then false premise else true premise
        elif chain_type == 1:
            adjusted_answer = self.problem.wrong_nl_answer.strip().removesuffix(".")
            new_id = 999
            new_premises = (
                self.problem.nl_premises
                + " If it is true that "
                + adjusted_answer
                + ", then the following is true: [ "
                + next_problem.alternate_first_premise
                + "] Otherwise, the following is true: ["
                + next_problem.first_premise
                + "] "
                + next_problem.nl_premises
            )

        new_first_premise = self.problem.first_premise
        new_alternate_first_premise = self.problem.alternate_first_premise
        new_question = next_problem.nl_question
        new_answer = next_problem.nl_answer
        new_wrong_answer = next_problem.wrong_nl_answer
        new_int_answer = next_problem.int_answer
        new_problem = GSM8KProblem(
            new_id, new_first_premise, new_alternate_first_premise, new_premises, new_question, new_answer, new_int_answer, new_wrong_answer
        )

        return self.__class__(
            problem=new_problem, ids=self.ids+[next_problem.id]
        )

    def chain_if_then_else_backwards(self, next_problem: GSM8KProblem) -> Self:
        chain_type = random.randint(0, 1)

        # if true conclusion then true premise else false premise
        if chain_type == 0:
            adjusted_answer = self.problem.nl_answer.strip().removesuffix(".")
            new_id = 999
            new_premises = (
                " If it is true that "
                + adjusted_answer
                + ", then the following is true: ["
                + next_problem.first_premise
                + "] Otherwise, the following is true: ["
                + next_problem.alternate_first_premise
                + "] "
                + next_problem.nl_premises
                + " "
                + self.problem.nl_premises
            )
        elif chain_type == 1:
            adjusted_answer = self.problem.wrong_nl_answer.strip().removesuffix(".")
            new_id = 999
            new_premises = (
                " If it is true that "
                + adjusted_answer
                + ", then the following is true: ["
                + next_problem.alternate_first_premise
                + "] Otherwise, the following is true: ["
                + next_problem.first_premise
                + "] "
                + next_problem.nl_premises
                + " "
                + self.problem.nl_premises
            )

        new_problem = GSM8KProblem(new_id, next_problem.first_premise, next_problem.alternate_first_premise, new_premises, next_problem.nl_question, next_problem.nl_answer, next_problem.int_answer, next_problem.wrong_nl_answer)
        return self.__class__(
            problem=new_problem, ids=self.ids+[next_problem.id]
        )

    def chain_if_then_else_backwards_first_link(self, next_problem: GSM8KProblem) -> Self:
        chain_type = random.randint(0, 1)

        # if true conclusion then true premise else false premise
        if chain_type == 0:
            adjusted_answer = self.problem.nl_answer.strip().removesuffix(".")
            new_id = 999
            new_premises = (
                " If it is true that "
                + adjusted_answer
                + ", then the following is true: ["
                + next_problem.first_premise
                + "] Otherwise, the following is true: ["
                + next_problem.alternate_first_premise
                + "] "
                + next_problem.nl_premises
                + " "
                + self.problem.first_premise
                + " "
                + self.problem.nl_premises
            )
        elif chain_type == 1:
            adjusted_answer = self.problem.wrong_nl_answer.strip().removesuffix(".")
            new_id = 999
            new_premises = (
                " If it is true that "
                + adjusted_answer
                + ", then the following is true: ["
                + next_problem.alternate_first_premise
                + "] Otherwise, the following is true: ["
                + next_problem.first_premise
                + "] "
                + next_problem.nl_premises
                + " "
                + self.problem.first_premise
                + " "
                + self.problem.nl_premises
            )

        new_problem = GSM8KProblem(new_id, next_problem.first_premise, next_problem.alternate_first_premise, new_premises, next_problem.nl_question, next_problem.nl_answer, next_problem.int_answer, next_problem.wrong_nl_answer)
        return self.__class__(
            problem=new_problem, ids=self.ids+[next_problem.id]
        )

    def make_question_length_2_plus(self) -> str:
        return self.problem.nl_premises + " " + self.problem.nl_question

    def make_question_length_1(self) -> str:
        return self.problem.first_premise + " " + self.problem.nl_premises + " " + self.problem.nl_question

    def make_question(self) -> str:
        return (
            self.make_question_length_1()
            if len(self.ids) == 1
            else self.make_question_length_2_plus()
        )

    def __run_gpt4(
        self, model_string: str, prompt: str, json_schema: dict
    ) -> dict[str, int]:
        resp = CLIENT.chat.completions.create(
            model=model_string,
            temperature=LLM_TEMPERATURE,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        )

        resp = CLIENT.chat.completions.create(
            model=model_string,
            temperature=LLM_TEMPERATURE,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                },
                {
                    "role": "assistant",
                    "content": str(resp.choices[0].message.content),
                },
                {
                    "role": "user",
                    "content": self.JSON_PROMPT,
                },
            ],
            response_format={
                "type": "json_schema",
                "json_schema": json_schema,
            },
        )
        if resp.choices[0].message.content is not None:
            cleaned_string = clean_output(resp.choices[0].message.content)
            return dict(json.loads(cleaned_string))
        else:
            return dict()

    def __run_gemini(
        self, model_string: str, prompt: str, json_schema: dict
    ) -> dict[str, int]:
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",
            temperature=0,
            timeout=None,
            max_retries=2,
        )
        cot_answer_nl = str(llm.invoke(prompt).content)

        model = genai.GenerativeModel(
            "gemini-1.5-pro",
            generation_config={"response_mime_type": "application/json"},
        )
        schema_str = "{" + ", ".join([f'"{e}": int' for e in exprs]) + "}"

        structured_prompt = f"""
{prompt}

Solution:
{cot_answer_nl}

Using this JSON schema:
answer = {schema_str}
  """
        response = model.generate_content(structured_prompt)
        answers: dict[str, int] = json.loads(response.text)
        return answers

    def __run_bedrock(
        self,
        model_string: str,
        prompt: str,
        json_schema: dict,
        override_answer: str | None = None,
    ) -> dict[str, int]:
        session = boto3.Session()
        bedrock = session.client(service_name="bedrock-runtime")
        sanitized_schema = json.loads(json.dumps(json_schema))  # deep copy
        original_names_map = {
            str(hash(k)): k for k in json_schema["schema"]["required"]
        }
        sanitized_schema["schema"]["required"] = [
            str(hash(k)) for k in json_schema["schema"]["required"]
        ]
        sanitized_schema["schema"]["properties"] = {
            str(hash(k)): {
                "type": "integer",
                "description": k,
            }
            for k in json_schema["schema"]["required"]
        }

        TOOL_NAME = "all_steps"
        tool_list = [
            {
                "toolSpec": {
                    "name": TOOL_NAME,
                    "description": "Provide all intermediate answers derived while solving this question.",
                    "inputSchema": {"json": sanitized_schema["schema"]},
                }
            }
        ]

        response = bedrock.converse(
            modelId="anthropic.claude-3-5-sonnet-20240620-v1:0",  # Only Anthropic supports tool use
            messages=[
                {
                    "role": "user",
                    "content": [{"text": prompt}],
                },
                (
                    bedrock.converse(
                        modelId=model_string,
                        messages=[
                            {
                                "role": "user",
                                "content": [{"text": prompt}],
                            },
                        ],
                        inferenceConfig={"maxTokens": 2000, "temperature": 0},
                    )["output"]["message"]
                    if override_answer == None
                    else {
                        "role": "assistant",
                        "content": [{"text": override_answer}],
                    }
                ),
                {
                    "role": "user",
                    "content": [{"text": self.JSON_PROMPT}],
                },
            ],
            inferenceConfig={"maxTokens": 2000, "temperature": 0},
            toolConfig={
                "tools": tool_list,
                "toolChoice": {"tool": {"name": TOOL_NAME}},
            },
        )
        response_message = response["output"]["message"]
        response_content_blocks = response_message["content"]
        content_block = next(
            (block for block in response_content_blocks if "toolUse" in block), None
        )
        keys_hashed_map = (
            content_block["toolUse"]["input"] if content_block is not None else dict()
        )
        return {
            original_names_map[k]: v
            for k, v in keys_hashed_map.items()
            if k in original_names_map.keys()
        }

    def __run_passthrough_print_tsv(
        self, prompt: str,
    ) -> dict[str, int]:
        """Fake evaluation function used to print CSV.

        Header:
        length, chained ids, question, o1_answer (empty), is_correct (empty)
        """
        print(f"{len(self.ids)}\t{self.ids}\t{prompt}\t\t")
        return dict()

    def run_model(self, model_string: str, override_answer: str| None = None) -> dict[str, int]:
        exprs: list[str] = [
            p.nl_question for p in load_manual_benchmark() if p.id in self.ids
        ]
        json_schema = {
            "name": "math_answer_mapping",
            "schema": {
                "type": "object",
                "properties": {e: {"type": "integer"} for e in exprs},
                "required": list(exprs),
                "additionalProperties": False,
            },
            "strict": True,
        }

        prompt = self.make_question()
        print(prompt, file=sys.stderr)

        if model_string.startswith("gpt-"):
            return self.__run_gpt4(model_string, prompt, json_schema)
        elif model_string == "anthropic.claude-3-5-sonnet-20240620-v1:0":
            return self.__run_bedrock(model_string, prompt, json_schema)
        elif model_string == "meta.llama3-70b-instruct-v1:0":
            return self.__run_bedrock(model_string, prompt, json_schema)
        elif model_string.startswith("gemini"):
            return self.__run_gemini(model_string, prompt, json_schema)
        elif override_answer != None:
            return self.__run_bedrock("o1", prompt, json_schema, override_answer=override_answer)
        else:
            return self.__run_passthrough_print_tsv(prompt)

    def validate_answer(
        self, output: dict[str, int], original_problems: list[GSM8KProblem]
    ) -> bool:

        problems_by_question: dict[str, GSM8KProblem] = {
            p.nl_question: p for p in original_problems
        }

        return len(self.ids) == len(output) and all(
            problems_by_question[q].int_answer == answer for q, answer in output.items()
        )


    def evaluate_on_model(
            self, model_string: str, original_problems: list[GSM8KProblem], override_answer: str | None = None
    ):
        """Run the model and evaluate result. Print to STDOUT"""
        try:
            output = self.run_model(model_string, override_answer=override_answer)
        except:
            output = dict()

        is_correct = self.validate_answer(output, original_problems)
        print(
            json.dumps(
                {
                    "id_chain": self.ids,
                    "is_correct": is_correct,
                    "model_string": model_string,
                    "output": output,
                    "problem": self.make_question(),
                    "solutions": [p.int_answer for p in load_manual_benchmark() if p.id in self.ids]
                }
            )
        )
        return

    def evaluate_o1_override(
        self, ids: list[int], question: str, o1_answer: str,
    ):
        self.ids = ids
        original_problems: list[GSM8KProblem] = load_manual_benchmark()
        model_string = "o1"
        try:
            output = self.run_model(model_string, override_answer=o1_answer)
        except:
            output = dict()

        is_correct = self.validate_answer(output, original_problems)
        print(
             json.dumps(
                 {
                     "id_chain": self.ids,
                     "is_correct": is_correct,
                     "model_string": model_string,
                     "output": output,
                     "problem": question,
                     "solutions": [p.int_answer for p in load_manual_benchmark() if p.id in self.ids]
                 }
             )
        )
        return

def hierarchy_aware_enumeration(original_problems: list[GSM8KProblem], depth: int):
    """Enumerate all passing chains of length K."""

    chains_of_length_k: list[WrapperChainGSM8K] = [
        WrapperChainGSM8K(p, [p.id]) for p in original_problems
    ]

    chains_of_k_plus_one: list[WrapperChainGSM8K] = []
    for _ in range(1, depth):
        for c in chains_of_length_k:
            all_extensions = [
                c.chain(p) for p in original_problems if p.id not in c.ids
            ]
            chains_of_k_plus_one += [
                p
                for p in all_extensions
                if p.evaluate_on_model("gpt-4o-2024-08-06",
                                       original_problems)
            ]
        chains_of_length_k = chains_of_k_plus_one
        chains_of_k_plus_one = []

    return [c.problem for c in chains_of_length_k]

def substitute_pronouns_with_proper(text: str, model_string) -> str:
    """Uses OpenAI to substitute all pronouns in the text with proper
    nouns they refer to. This allows the sentences to be mixed more
    freely."""

    PROMPT=f"""Please substitute all pronouns in TEXT with proper nouns they refer to. Don't replace pronouns within the same sentence. Put the answers inside <PROPER-NOUN-ONLY>.
{text}
</TEXT>
<PROPER-NOUN-ONLY>
"""
    resp = CLIENT.chat.completions.create(
            model=model_string,
            temperature=LLM_TEMPERATURE,
            messages=[
                {"role": "user", "content": PROMPT},
            ],
        )
    proper_nouns_only = re.sub("<.*?>", "", str(resp.choices[0].message.content)).strip()

    initial_word_soup = set[str](re.split(r"\s+", text.replace(".", "")))
    proper_nouns_only_word_soup = set[str](
        re.split(r"\s+", proper_nouns_only.replace(".", ""))
    )
    print(f"deleted pronouns (?): {initial_word_soup - proper_nouns_only_word_soup}")
    return proper_nouns_only


def chain_n_problems(problems: list[GSM8KProblem], depth: int) -> list[GSM8KProblem]:
    """Enumerates all chains of up to length k."""
    chains_of_length_k: list[WrapperChainGSM8K] = [
        WrapperChainGSM8K(p, [p.id]) for p in problems
    ]

    chains_of_k_plus_one: list[WrapperChainGSM8K] = []
    for _ in range(1, depth):
        for c in chains_of_length_k:
            chains_of_k_plus_one += [c.chain(p) for p in problems if p.id not in c.ids]
        chains_of_length_k = chains_of_k_plus_one
        chains_of_k_plus_one = []

    return [c.problem for c in chains_of_length_k]

def sample_random_problems_as_chains(problems: list[GSM8KProblem], num_to_sample: int) -> list[WrapperChainGSM8K]:
    random_problems = random.sample(problems, num_to_sample)
    random_chains = [WrapperChainGSM8K(p, [p.id]) for p in random_problems]
    return random_chains

def uniformly_sample_problems(problems: list[GSM8KProblem], chain_length: int, num_to_sample: int) -> list[WrapperChainGSM8K]:
    if len(problems) < chain_length:
        ValueError("Not enough problems to sample from")

    if chain_length == 1:
        random_chains = sample_random_problems_as_chains(problems, num_to_sample)
        return random_chains

    sampled_chains = []
    random_permutations = []
    for _ in range(0, num_to_sample):
        random_permutation = get_random_number_with_distinct_digits(len(problems), chain_length)
        while random_permutation in random_permutations:
            random_permutation = get_random_number_with_distinct_digits(len(problems), chain_length)
        random_permutations.append(random_permutation)

        random_problems = [problems[i] for i in random_permutation]

        chain_wrapper = WrapperChainGSM8K(random_problems[0], [random_problems[0].id])
        for c in range(0, len(random_problems)-1):
            chain_wrapper = chain_wrapper.chain_simple(random_problems[c+1])
        sampled_chains.append(chain_wrapper)

    return sampled_chains

def uniformly_sample_problems_if_then_else(problems: list[GSM8KProblem], chain_length: int, num_to_sample: int) -> list[WrapperChainGSM8K]:
    if len(problems) < chain_length:
        ValueError("Not enough problems to sample from")

    if chain_length == 1:
        random_chains = sample_random_problems_as_chains(problems, num_to_sample)
        return random_chains

    sampled_chains = []
    random_permutations = []
    for _ in range(0, num_to_sample):
        random_permutation = get_random_number_with_distinct_digits(len(problems), chain_length)
        while random_permutation in random_permutations:
            random_permutation = get_random_number_with_distinct_digits(len(problems), chain_length)
        random_permutations.append(random_permutation)

        random_problems = [problems[i] for i in random_permutation]

        chain_wrapper = WrapperChainGSM8K(random_problems[0], [random_problems[0].id])
        for c in range(0, len(random_problems)-1):
            chain_wrapper = chain_wrapper.chain_if_then_else(random_problems[c+1])
        sampled_chains.append(chain_wrapper)

    return sampled_chains

def uniformly_sample_problems_if_then_else_backwards(problems: list[GSM8KProblem], chain_length: int, num_to_sample: int) -> list[WrapperChainGSM8K]:
    if len(problems) < chain_length:
        ValueError("Not enough problems to sample from")

    if chain_length == 1:
        random_chains = sample_random_problems_as_chains(problems, num_to_sample)
        return random_chains

    sampled_chains = []
    random_permutations = []
    for _ in range(0, num_to_sample):
        random_permutation = get_random_number_with_distinct_digits(len(problems), chain_length)
        while random_permutation in random_permutations:
            random_permutation = get_random_number_with_distinct_digits(len(problems), chain_length)
        random_permutations.append(random_permutation)

        random_problems = [problems[i] for i in random_permutation]

        chain_wrapper = WrapperChainGSM8K(random_problems[0], [random_problems[0].id])
        chain_wrapper = chain_wrapper.chain_if_then_else_backwards_first_link(random_problems[1])
        for c in range(1, len(random_problems)-1):
            chain_wrapper = chain_wrapper.chain_if_then_else_backwards(random_problems[c+1])
        sampled_chains.append(chain_wrapper)

    return sampled_chains

def get_random_number_with_distinct_digits(number_base, num_digits):
    random_digits = []
    while True:
        random_index = random.randrange(0, number_base)
        if random_index not in random_digits:
            random_digits.append(random_index)
        if len(random_digits) == num_digits:
            break
    return random_digits

def nPr(n: int, r: int) -> int:
    return math.factorial(n) // math.factorial(n-r)

def main():
    parser = argparse.ArgumentParser(description='Process flags with multiple integer arguments.')

    parser.add_argument('-f', nargs=2, type=int, help='arguments: length, sample size')
    parser.add_argument('-b', nargs=2, type=int, help='arguments: length, sample size')
    parser.add_argument('-m', nargs=1, type=str,
                        help='model name:= "gpt-4o-2024-08-06" | "anthropic.claude-3-5-sonnet-20240620-v1:0" | "meta.llama3-70b-instruct-v1:0"',
                        required=True)
    parser.add_argument('-r', nargs="?", help='Replay inputs from stdin')

    args = parser.parse_args()

    problems = load_manual_benchmark()
    if args.f:
        print(f'Flag -f received arguments: {args.f[0]} and {args.f[1]}', file=sys.stderr)
        chain_length = int(args.f[0])
        sample_size = int(args.f[1])
        model_name = str(args.m[0])
        sampled_chains = uniformly_sample_problems_if_then_else(problems, chain_length, sample_size)
        for chain in sampled_chains:
            chain.evaluate_on_model(model_name, problems)
    elif args.b:
        print(f'Flag -b received arguments: {args.b[0]} and {args.b[1]}', file=sys.stderr)
        chain_length = int(args.b[0])
        sample_size = int(args.b[1])
        model_name = str(args.m[0])
        sampled_chains = uniformly_sample_problems_if_then_else_backwards(problems, chain_length, sample_size)
        for chain in sampled_chains:
            chain.evaluate_on_model(model_name, problems)
    else:
        print("replay. Ignore the question about Janet.", file=sys.stderr)
        tmp_replay_chain = WrapperChainGSM8K(load_manual_benchmark()[0], [])
        for line in [line for index, line in enumerate(sys.stdin) if index > 0]:
            columns = line.split ("\t")
            id_chain = json.loads(columns[1])
            question = columns[3 -1 ]
            o1_answer = columns[4 -1]
            if o1_answer.strip() != "" and len(id_chain) == 1:
                tmp_replay_chain.evaluate_o1_override(id_chain, question, o1_answer)


if __name__ == "__main__":
    main()
