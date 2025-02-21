import re
import os
import tempfile
from dataclasses import dataclass, field


import yaml
from groq import Groq
import streamlit as st
from dotenv import load_dotenv


st.set_page_config(
    layout="wide",
    page_title="UVA Bot 🤖",
    page_icon=":dna:",
)


model_name = "deepseek-r1-distill-qwen-32b"


def remove_think_tags(response: str) -> str:
    think_regex = re.compile(r"<think>.*?</think>", flags=re.DOTALL | re.MULTILINE)
    return think_regex.sub("", response).strip()


@dataclass
class Patient:
    patient_id: str
    biomarker_report: str
    imaging_report: str
    pathology_report: str


@dataclass
class AgentPrompt:
    prompt: str
    parameters: list[str] = field(
        default_factory=list
    )  # defines the parameters as an empty list without instantiating it

    def apply(self, parameters: dict[str, str]) -> str:
        prompt = self.prompt
        for key, value in parameters.items():
            prompt = prompt.replace(
                "{{" + key + "}}", value
            )  # replace "{{BiomarkersReport}}" with the actual biomarker report
        return prompt


@dataclass
class CrewPrompts:
    biomarker: AgentPrompt
    imaging: AgentPrompt
    pathology: AgentPrompt
    oncologist: AgentPrompt


def load_patient_data(patient_id: str) -> Patient:
    biomarker_path = os.path.join(
        "data", "patients", patient_id, "biomarker_report.txt"
    )
    imaging_path = os.path.join("data", "patients", patient_id, "imaging_report.txt")
    pathology_path = os.path.join(
        "data", "patients", patient_id, "pathology_report.txt"
    )
    with open(biomarker_path, "r") as f:
        biomarker_report = f.read()
    with open(imaging_path, "r") as f:
        imaging_report = f.read()
    with open(pathology_path, "r") as f:
        pathology_report = f.read()
    return Patient(patient_id, biomarker_report, imaging_report, pathology_report)


def get_patient_data(patient_id: str, cursor) -> Patient:
    cursor.execute(
        "SELECT biomarker, imaging, pathology FROM patients WHERE id = ?",
        (patient_id,),
    )
    biomarker_report, imaging_report, pathology_report = cursor.fetchone()
    return Patient(patient_id, biomarker_report, imaging_report, pathology_report)


def load_patient_data_from_sqlite_file(path):
    """
    Generates a map from a patient id to the patient data.

    {
        "P001": Patient(patient_id="P001", biomarker_report="...", imaging_report="...", pathology_report="..."),
        "P002": Patient(patient_id="P002", biomarker_report="...", imaging_report="...", pathology_report="..."),
        ...
    }

    """
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
    biomarker_prompt_path = os.path.join("prompts", "biomarker_analyst.yaml")
    imaging_prompt_path = os.path.join("prompts", "imaging_analyst.yaml")
    pathology_prompt_path = os.path.join("prompts", "pathology_analyst.yaml")
    oncologist_prompt_path = os.path.join("prompts", "oncologist.yaml")
    with open(biomarker_prompt_path, "r") as f:
        biomarker_prompt_dict = yaml.safe_load(f)
        biomarker_prompt = AgentPrompt(
            prompt=biomarker_prompt_dict["prompt"],
            parameters=biomarker_prompt_dict["parameters"],
        )
    with open(imaging_prompt_path, "r") as f:
        imaging_prompt_dict = yaml.safe_load(f)
        imaging_prompt = AgentPrompt(
            prompt=imaging_prompt_dict["prompt"],
            parameters=imaging_prompt_dict["parameters"],
        )
    with open(pathology_prompt_path, "r") as f:
        pathology_prompt_dict = yaml.safe_load(f)
        pathology_prompt = AgentPrompt(
            prompt=pathology_prompt_dict["prompt"],
            parameters=pathology_prompt_dict["parameters"],
        )
    with open(oncologist_prompt_path, "r") as f:
        oncologist_prompt_dict = yaml.safe_load(f)
        oncologist_prompt = AgentPrompt(
            prompt=oncologist_prompt_dict["prompt"],
            parameters=oncologist_prompt_dict["parameters"],
        )
    return CrewPrompts(
        biomarker_prompt, imaging_prompt, pathology_prompt, oncologist_prompt
    )


class Agent:
    def __init__(self, model_name: str, prompt: AgentPrompt):
        self.model_name = model_name
        self.prompt = prompt
        self.client = Groq()

    def run(self, parameters: dict[str, str]) -> str:
        message = self.prompt.apply(
            parameters
        )  # call the apply method of the AgentPrompt class
        chat_completion = self.client.chat.completions.create(  # call the LLM with the materialized prompt
            messages=[
                {
                    "role": "user",
                    "content": message,
                }
            ],
            model=self.model_name,
            temperature=0.0,
            top_p=1.0,
            seed=0,
        )

        answer = chat_completion.choices[0].message.content
        return remove_think_tags(answer)  # remove the <think> tags from the response


# for logging purposes
@dataclass
class RunResult:
    biomarker_report: str
    imaging_report: str
    pathology_report: str
    oncologist_report: str


def main():
    load_dotenv()
    crew_prompts = load_prompts()
    biomarker_agent = Agent(model_name=model_name, prompt=crew_prompts.biomarker)
    imaging_agent = Agent(model_name=model_name, prompt=crew_prompts.imaging)
    pathology_agent = Agent(model_name=model_name, prompt=crew_prompts.pathology)
    oncologist_agent = Agent(model_name=model_name, prompt=crew_prompts.oncologist)

    patient_id_to_patient = {}

    def run_patient(patient_id: str) -> RunResult:
        patient = patient_id_to_patient[patient_id]
        agents = [biomarker_agent, imaging_agent, pathology_agent]
        reports = [
            patient.biomarker_report,
            patient.imaging_report,
            patient.pathology_report,
        ]
        parameters = ["BiomarkersReport", "ImagingReport", "PathologyReport"]
        outputs = ["ImagingUVABot", "PathologyUVABot", "BiomarkerUVABot"]
        result = {}

        bar = st.sidebar.progress(0.0)
        for i, (agent, report, params, output) in enumerate(
            zip(agents, reports, parameters, outputs)
        ):
            bar.progress((i + 1) / len(agents), text=output)
            result[output] = agent.run({params: report})
        final_report = oncologist_agent.run(result)

        run_result = RunResult(
            biomarker_report=result["BiomarkerUVABot"],
            imaging_report=result["ImagingUVABot"],
            pathology_report=result["PathologyUVABot"],
            oncologist_report=final_report,
        )  # logging in streamlit
        return run_result

    st.sidebar.title(":orange[__UVA Bot__] 🤖")

    uploaded_file = st.sidebar.file_uploader(
        "Choose a file"
    )  # upload the file in the UI in streamlit
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.read())
            db_path = tmp_file.name  # Get the file path

            patient_id_to_patient = load_patient_data_from_sqlite_file(db_path)

        patient_ids = list(patient_id_to_patient.keys())
        patient_id = st.sidebar.selectbox(
            "Select Patient ID", patient_ids
        )  # select the patient id from the UI in streamlit

        patient = patient_id_to_patient[patient_id]

        # documentation is on Streamlit docs website

        tab1, tab2, tab3 = st.tabs(
            [
                "Biomarkers Report",
                "Imaging Report",
                "Pathology Report",
            ]  # visualize the reports of the patient
        )

        with tab1:
            st.text(patient.biomarker_report)

        with tab2:
            st.text(patient.imaging_report)

        with tab3:
            st.text(patient.pathology_report)

        st.divider()  # put a line separator to separate the inputs from the outputs visually

        submit = st.sidebar.button("Submit")
        if submit:
            run_result = run_patient(patient_id)
            t1, t2, t3, t4 = st.tabs(
                ["Biomarkers", "Imaging", "Pathology", "Oncologist"]
            )

            with t1:
                st.markdown(run_result.biomarker_report)

            with t2:
                st.markdown(run_result.imaging_report)

            with t3:
                st.markdown(run_result.pathology_report)

            with t4:
                st.markdown(run_result.oncologist_report)


if __name__ == "__main__":
    main()
