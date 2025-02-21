import os
import re
from dataclasses import dataclass

import yaml
from graph import Prompt


think_regex = re.compile(r"<think>.*?</think>", flags=re.DOTALL | re.MULTILINE)


@dataclass
class Patient:
    patient_id: str
    biomarker_report: str
    imaging_report: str
    pathology_report: str


@dataclass
class CrewPrompts:
    biomarker: Prompt
    imaging: Prompt
    pathology: Prompt
    oncologist: Prompt


def get_think_tag(response: str) -> str:
    think_tags = think_regex.findall(response)
    return "".join(think_tags)


def remove_think_tag(response: str) -> str:
    return think_regex.sub("", response).strip()


def get_patient_data(patient_id: str, cursor) -> Patient:
    cursor.execute(
        "SELECT biomarker, imaging, pathology FROM patients WHERE id = ?",
        (patient_id,),
    )
    biomarker_report, imaging_report, pathology_report = cursor.fetchone()
    return Patient(patient_id, biomarker_report, imaging_report, pathology_report)


def load_patient_data_from_sqlite_file(path):
    import sqlite3

    conn = sqlite3.connect(path)
    c = conn.cursor()
    c.execute("SELECT * FROM patients")
    patient_ids = [row[0] for row in c.fetchall()]
    patient_id_to_patient = {pid: get_patient_data(pid, c) for pid in patient_ids}
    c.close()
    conn.close()
    return patient_id_to_patient


def load_prompts():
    def load_prompt(prompt_path):
        with open(prompt_path, mode="r", encoding="utf-8") as f:
            prompt_dict = yaml.safe_load(f)
            return Prompt(
                text=prompt_dict["prompt"],
                parameters=prompt_dict["parameters"],
            )

    biomarker_prompt_path = os.path.join("prompts", "biomarker_analyst.yaml")
    imaging_prompt_path = os.path.join("prompts", "imaging_analyst.yaml")
    pathology_prompt_path = os.path.join("prompts", "pathology_analyst.yaml")
    oncologist_prompt_path = os.path.join("prompts", "oncologist.yaml")

    biomarker_prompt = load_prompt(biomarker_prompt_path)
    imaging_prompt = load_prompt(imaging_prompt_path)
    pathology_prompt = load_prompt(pathology_prompt_path)
    oncologist_prompt = load_prompt(oncologist_prompt_path)

    return CrewPrompts(
        biomarker_prompt, imaging_prompt, pathology_prompt, oncologist_prompt
    )
