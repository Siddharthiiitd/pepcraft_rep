import argparse
import time
import os
import shutil
import re
import json
import importlib
from pathlib import Path
from typing import Literal

import pandas as pd
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableConfig

from states import PlanState


def _load_dotenv(env_path=".env"):
    """Minimal .env loader -- no python-dotenv dependency required.
    Reads KEY=VALUE lines from env_path (relative to cwd) and sets them
    in os.environ, without overwriting anything already set for real."""
    if not os.path.exists(env_path):
        return
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


_load_dotenv()

# --- API key: read from the environment (or the .env file above), never
# hardcode it here. Put this in a .env file next to this script:
#   GOOGLE_API_KEY=your-key-here
# and make sure .env is in your .gitignore.
if "GOOGLE_API_KEY" not in os.environ:
    raise RuntimeError(
        "GOOGLE_API_KEY is not set. Add it to a .env file next to this script "
        "(GOOGLE_API_KEY=your-key-here) or `export GOOGLE_API_KEY=your-key-here` "
        "before launching."
    )

MAX_RESPONSE_RETRIES = 5   # cap on malformed-JSON retries for a single LLM call
MAX_EXECUTOR_RETRIES = 5   # cap on tool-execution-error retries per plan step


def call_tool(tool: str, agent: str, input_data: dict):
    module = importlib.import_module(f"tools.{agent}.{tool}")
    func = getattr(module, tool)
    return func(input_data)


# Only the 3.x Gemini line supports the `thinking_level` param as of this
# writing -- 2.5 and earlier models reject it with a 400 INVALID_ARGUMENT.
# Extend this set as you verify other models support it; don't assume
# every "gemini*" model does.
THINKING_LEVEL_SUPPORTED_PREFIXES = ("gemini-3",)


def _model_kwargs(model_id: str, thinking_level: str = "medium") -> dict:
    kwargs = dict(
        model=model_id,
        include_thoughts=True,
        temperature=1.0,
        timeout=60.0,
    )
    if model_id.lower().startswith(THINKING_LEVEL_SUPPORTED_PREFIXES):
        kwargs["thinking_level"] = thinking_level
    return kwargs


class AMP_Agents:
    def __init__(self, user_prompt: str, run_id: str, output_base: str,
                 planner_model_id="gemini-3.1-pro-preview",
                 executor_model_id="gemini-3.1-flash-lite-preview",
                 num_gen=20):
        self.num_gen = num_gen
        self.run_id = run_id

        # Portable output dir -- no hardcoded /home/raymondlab path.
        self.output_dir = Path(output_base) / f"output_{self.run_id}_pro_{self.num_gen}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        seq_csv = self.output_dir / "generated_sequences.csv"
        if seq_csv.exists():
            seq_csv.unlink()

        self.user_prompt = user_prompt
        self.record_time_df = []

        self.Planner = ChatGoogleGenerativeAI(**_model_kwargs(planner_model_id))
        self.Executor = ChatGoogleGenerativeAI(**_model_kwargs(executor_model_id))

        self.builder = StateGraph(PlanState)
        self.builder.add_node("Planning", self.call_planner)

        self.base_dir = Path(__file__).resolve().parent
        self.agent_specs = []
        self.agents = ["Planning"]
        agents_dir = self.base_dir / "agents"
        for agent_path in agents_dir.glob("*.json"):
            with agent_path.open("r", encoding="utf-8") as f:
                self.agent_specs.append(json.load(f))
            self.agents.append(self.agent_specs[-1]['name'])
            self.builder.add_node(self.agent_specs[-1]['name'], self.call_executor)
            self.builder.add_edge(self.agent_specs[-1]['name'], "Planning")
        self.builder.add_node("END", self.call_end)

        self.builder.add_edge(START, "Planning")
        self.builder.add_conditional_edges("Planning", self.plan)
        self.builder.add_edge("END", END)

        self.graph = self.builder.compile()
        print(self.graph.get_graph().draw_ascii())

        self.prompt_builder()
        self.reset_log()

    def get_prompts(self, file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    def prompt_builder(self):
        self.planner_prompt = self.get_prompts(self.base_dir / "prompts" / "Planner" / "init.txt")
        self.planner_next_prompt = self.get_prompts(self.base_dir / "prompts" / "Planner" / "next.txt")
        self.executor_prompt = self.get_prompts(self.base_dir / "prompts" / "Executor" / "init.txt")
        self.executor_report_prompt = self.get_prompts(self.base_dir / "prompts" / "Executor" / "report.txt")

        agent_descriptions = ""
        agent_contexts = ""

        for spec in self.agent_specs:
            agent_descriptions += f"**{spec['name']}**\nDescription: {spec['description']}\n\nSkills:\n"
            for i, skill in enumerate(spec['tools']):
                agent_descriptions += f"{i+1}. {skill['tool']}: {skill['description']}\n  - input: {skill['input']}\n"
            agent_descriptions += "\n\n"
            agent_contexts += f"{spec['name']} \n===================\n" + "\n".join(spec['context']) + "\n\n"

        self.planner_prompt = self.planner_prompt.replace("{Agent Description}", agent_descriptions)
        self.planner_prompt = self.planner_prompt.replace("{Agent Context}", agent_contexts)
        self.planner_prompt += f"\n\nUser Instruction: {self.user_prompt}"

    def reset_log(self):
        self.log_dir = self.base_dir / f"logs_{self.run_id}_pro_{self.num_gen}"
        if self.log_dir.is_symlink() or self.log_dir.is_file():
            self.log_dir.unlink(missing_ok=True)
        else:
            shutil.rmtree(self.log_dir, ignore_errors=True)

        for agent in self.agents:
            for sub in ("prompt", "thinking", "response"):
                (self.log_dir / agent / sub).mkdir(parents=True, exist_ok=True)

    def generate_response(self, agent, prompt, json_structure=False):
        """Calls the LLM and (optionally) parses JSON out of the reply.
        Bounded retries -- a persistently malformed response now raises
        instead of hanging the process forever."""
        think_str = ""
        text_str = ""
        root = None
        last_error = None

        for attempt in range(1, MAX_RESPONSE_RETRIES + 1):
            try:
                response = agent.invoke(prompt)

                for block in response.content:
                    if isinstance(block, dict) and block.get("type") == "thinking":
                        think_str = block.get("thinking", "No thought text found.")
                    elif isinstance(block, dict) and block.get("type") == "text":
                        text_str = block.get("text", "")

                if json_structure:
                    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text_str, re.DOTALL)
                    json_str = match.group(1) if match else text_str
                    root = json.loads(json_str)

                return text_str, think_str, (root if json_structure else None)

            except Exception as e:
                last_error = e
                print(f"[generate_response] attempt {attempt}/{MAX_RESPONSE_RETRIES} failed: {e}")
                time.sleep(min(2 ** attempt, 30))

        raise RuntimeError(
            f"generate_response failed after {MAX_RESPONSE_RETRIES} attempts. Last error: {last_error}"
        )

    def log_response(self, agent_name, prompt, think_str, response_str):
        plan_num = os.listdir(self.log_dir / agent_name / "prompt")
        with open(self.log_dir / agent_name / "prompt" / f"prompt_{len(plan_num)+1}.txt", "w", encoding="utf-8") as f:
            f.write(prompt)
        with open(self.log_dir / agent_name / "thinking" / f"thinking_{len(plan_num)+1}.txt", "w", encoding="utf-8") as f:
            f.write(think_str)
        with open(self.log_dir / agent_name / "response" / f"response_{len(plan_num)+1}.txt", "w", encoding="utf-8") as f:
            f.write(response_str)

    def plan(self, state: PlanState, config: RunnableConfig) -> Literal["Generating", "Filtering", "Verifying", "END"]:
        if state['stage'] == "END":
            with open(self.base_dir / "prompts" / "Planner" / "reporting.txt", "r", encoding="utf-8") as f:
                reporting_prompt = f.read()
            reporting_prompt = reporting_prompt.replace("{user_prompt}", self.user_prompt)
            reporting_prompt += f"\n\nUser Instruction: {self.user_prompt}\n\n"

            df = pd.read_csv(self.output_dir / "generated_sequences.csv")
            report_columns = [col for col in df.columns if "report" in col]
            df = df.dropna(subset=report_columns)

            for i, row in df.iterrows():
                reporting_prompt += f"\n\nFor the generated sequence {row['sequence']}:\n\n"
                for col in report_columns:
                    parts = col.split("_")
                    reporting_prompt += f"Here is the {parts[0]} {parts[-1]} of the generated sequences:\n\n{row[col]}\n"

            plan_str, think_str, _ = self.generate_response(self.Planner, reporting_prompt, json_structure=False)
            logging_path = self.log_dir / "Final_Report"
            (logging_path / "prompt").mkdir(parents=True, exist_ok=True)
            (logging_path / "thinking").mkdir(parents=True, exist_ok=True)
            (logging_path / "response").mkdir(parents=True, exist_ok=True)
            self.log_response("Final_Report", reporting_prompt, think_str, plan_str)

            final_report_path = self.output_dir / f"final_report_{self.run_id}_pro_{self.num_gen}.txt"
            with open(final_report_path, "w", encoding="utf-8") as f:
                f.write(plan_str)

            print(f"Final report has been generated at {final_report_path}")

        return state['stage']

    def call_planner(self, state: PlanState, config: RunnableConfig) -> PlanState:
        start = time.time()

        if state['from_exec']:
            prompt = self.planner_prompt + "\n\n" + self.planner_next_prompt
            prompt = prompt.replace("{Instruction}", state['messages'][-1])
            prompt = prompt.replace("{Agent}", state['stage'])
            prompt = prompt.replace("{Reports}", state['executor'][-1]['report'] if state['executor'] else "")
            state['stage'] = "Planning"
        else:
            prompt = self.planner_prompt

        plan_str, think_str, root = self.generate_response(self.Planner, prompt, json_structure=True)
        self.log_response(state['stage'], prompt, think_str, plan_str)

        agent = root["Planning"]["Agent"]
        state['stage'] = agent
        state['messages'].append(plan_str)

        end = time.time()
        self.record_time_df.append({"agent": "Planning", "time": end - start})
        return state

    def call_executor(self, state: PlanState, config: RunnableConfig) -> PlanState:
        start = time.time()
        agent_description = ""

        for spec in self.agent_specs:
            if spec['name'] == state['stage']:
                agent_description = f"Description: {spec['description']}\n\nSkills:\n"
                for i, skill in enumerate(spec['tools']):
                    agent_description += f"{i+1}. {skill['tool']}: {skill['description']}\n  - input: {skill['input']}\n"
                break

        prompt = self.executor_prompt.replace("{Agent}", state['stage'])
        prompt = prompt.replace("{Agent Description}", agent_description)
        prompt = prompt.replace("{Instruction}", state['messages'][-1])
        tool_prompt = prompt

        plan_str, think_str, root, breif_log = None, None, None, ""
        for attempt in range(1, MAX_EXECUTOR_RETRIES + 1):
            plan_str, think_str, root = self.generate_response(self.Executor, tool_prompt, json_structure=True)
            self.log_response(state['stage'], tool_prompt, think_str, plan_str)

            steps = root["Steps"]
            breif_log = ""
            try:
                for step in steps:
                    step_id = step["id"]
                    tool = step["Tool"]
                    input_data = step["Input"]
                    output = call_tool(tool, state['stage'], input_data)
                    breif_log += f"Step {step_id}: {output}"
            except Exception as e:
                print(f"[call_executor] tool execution error (attempt {attempt}/{MAX_EXECUTOR_RETRIES}): {e}")
                tool_prompt = prompt + "\n\n" + (
                    f"The execution of the plan encountered an error: {str(e)}. "
                    f"Please revise the plan and provide a new execution plan."
                )
                if attempt == MAX_EXECUTOR_RETRIES:
                    raise RuntimeError(
                        f"call_executor failed after {MAX_EXECUTOR_RETRIES} attempts for stage "
                        f"'{state['stage']}'. Last error: {e}"
                    )
            else:
                break

        report_prompt = self.executor_report_prompt.replace("{Agent}", state['stage'])
        report_prompt = report_prompt.replace("{Instruction}", plan_str)
        report_prompt = report_prompt.replace("{Summary}", breif_log)

        text_str, think_str, _ = self.generate_response(self.Executor, report_prompt, json_structure=False)
        self.log_response(state['stage'], report_prompt, think_str, text_str)

        entry = {"agent": state['stage'], "execution_plan": plan_str, "report": text_str, "step_reports": breif_log}
        if state['executor'] is None:
            state['executor'] = [entry]
        else:
            state['executor'].append(entry)
        state['from_exec'] = True

        end = time.time()
        self.record_time_df.append({"agent": state['stage'], "time": end - start})
        return state

    def call_end(self, state: PlanState, config: RunnableConfig) -> PlanState:
        pd.DataFrame(self.record_time_df).to_csv(self.output_dir / "execution_time.csv", index=False)
        return state

    def run(self):
        self.graph.invoke({
            "stage": "Planning",
            "executor": None,
            "messages": [],
            "from_exec": False,
        })


def parse_args():
    parser = argparse.ArgumentParser(description="Run the PepCraft AMP agent pipeline.")
    parser.add_argument("--run_id", required=True, help="Label for this run (replaces old sys.argv[1]).")
    parser.add_argument("--output_base", default="./outputs", help="Base directory for run outputs.")
    parser.add_argument("--counts", nargs="+", type=int, default=[5, 10, 20],
                         help="List of target sequence counts to run, one full pipeline pass each.")
    parser.add_argument("--species", default="ecoli",
                         choices=["ecoli", "paeruginosa", "kpneumoniae", "saureus", "bsubtilis", "sepidermidis"])
    parser.add_argument("--planner_model", default="gemini-3.1-pro-preview")
    parser.add_argument("--executor_model", default="gemini-3.1-flash-lite-preview")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    for n in args.counts:
        user_prompt = (
            f"Design exactly {n} AMP sequences with D-amino acids targeting {args.species}. "
            f"The target length is 10 - 20. Apply physicochemical filters for cationicity "
            f"(range: 2 to 8) and hydrophobicity (range: -0.5 to 0.5). "
            f"Use AMPGAN-v3 to generate. Please cross-reference with the protein "
            f"database and explain the candidate."
        )
        agent = AMP_Agents(
            user_prompt,
            run_id=args.run_id,
            output_base=args.output_base,
            planner_model_id=args.planner_model,
            executor_model_id=args.executor_model,
            num_gen=n,
        )
        agent.run()