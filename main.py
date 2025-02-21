from datetime import datetime
import tempfile
from dataclasses import dataclass


from groq import Groq
import streamlit as st
from dotenv import load_dotenv


from graph import Graph, Agent, ExecutionContext, Prompt
from utils import (
    get_think_tag,
    remove_think_tag,
    load_patient_data_from_sqlite_file,
    load_prompts,
)


st.set_page_config(
    layout="wide",
    page_title="UVA Bot 🤖",
    page_icon=":dna:",
)


# add logs in session state
if "logs" not in st.session_state:
    st.session_state.logs = []


def log(message):
    time = datetime.now().strftime("%H:%M:%S")
    st.session_state.logs.append(f"[{time}] {message}")


model_name = "deepseek-r1-distill-qwen-32b"


@dataclass(frozen=True)
class MedicalAgent(Agent):
    model_name: str
    client = Groq()

    def run(self, context: ExecutionContext) -> str:
        log(f"Running {self.name}")
        prompt = self._prepare_prompt(context)
        log(f"{self.name} Prompt: {prompt}")
        chat_completion = self.client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model=self.model_name,
            temperature=0.0,
            top_p=1.0,
            seed=0,
        )

        answer = chat_completion.choices[0].message.content
        think_tag = get_think_tag(answer)
        log(f"{self.name} Think Tag: {think_tag}")
        cleaned_answer = remove_think_tag(answer)
        log(f"{self.name} Response: {cleaned_answer}")
        context.set(self.name, cleaned_answer)
        return context


@dataclass
class RunResult:
    biomarker_report: str
    imaging_report: str
    pathology_report: str
    oncologist_report: str


def main():
    load_dotenv()
    crew_prompts = load_prompts()

    def agent(name: str, prompt: Prompt) -> MedicalAgent:
        return MedicalAgent(name=name, model_name=model_name, prompt=prompt)

    patient_id_to_patient = {}

    def run_patient(patient_id: str) -> RunResult:
        patient = patient_id_to_patient[patient_id]
        context = ExecutionContext()
        context.set("BiomarkersReport", patient.biomarker_report)
        context.set("ImagingReport", patient.imaging_report)
        context.set("PathologyReport", patient.pathology_report)

        graph = Graph.from_nodes(
            nodes={
                agent("biomarkers", crew_prompts.biomarker),
                agent("imaging", crew_prompts.imaging),
                agent("pathology", crew_prompts.pathology),
                agent("oncologist", crew_prompts.oncologist),
            },
            edges={
                "biomarkers -> oncologist",
                "imaging -> oncologist",
                "pathology -> oncologist",
            },
        )

        result = graph.run(context)

        run_result = RunResult(
            biomarker_report=result.get("biomarkers"),
            imaging_report=result.get("imaging"),
            pathology_report=result.get("pathology"),
            oncologist_report=result.get("oncologist"),
        )
        return run_result

    st.sidebar.title(":orange[__UVA Bot__] 🤖")

    uploaded_file = st.sidebar.file_uploader("Choose a file")
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.read())
            db_path = tmp_file.name  # Get the file path

            patient_id_to_patient = load_patient_data_from_sqlite_file(db_path)

        patient_ids = list(patient_id_to_patient.keys())
        patient_id = st.sidebar.selectbox("Select Patient ID", patient_ids)

        patient = patient_id_to_patient[patient_id]

        tab1, tab2, tab3 = st.tabs(
            ["Biomarkers Report", "Imaging Report", "Pathology Report"]
        )

        with tab1:
            st.text(patient.biomarker_report)

        with tab2:
            st.text(patient.imaging_report)

        with tab3:
            st.text(patient.pathology_report)

        st.divider()

        submit = st.sidebar.button("Submit")
        if submit:
            run_result = run_patient(patient_id)
            t1, t2, t3, t4, t5 = st.tabs(
                ["Biomarkers", "Imaging", "Pathology", "Oncologist", "Logs"]
            )

            with t1:
                st.markdown(run_result.biomarker_report)

            with t2:
                st.markdown(run_result.imaging_report)

            with t3:
                st.markdown(run_result.pathology_report)

            with t4:
                st.markdown(run_result.oncologist_report)

            with t5:
                for i, log_message in enumerate(st.session_state.logs):
                    with st.expander(log_message.splitlines()[0]):
                        st.code(log_message, language="text")
                    if i != 0 and i % 3 == 0:
                        st.divider()


if __name__ == "__main__":
    main()
