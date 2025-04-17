import datetime
import io
import os
import sys
import threading
import time
import zipfile
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Optional, List
import glob

import py7zr
import streamlit as st

from common import ProcessingState
from preprocess import PreprocessingCallback, PreprocessingConfig, preprocess_dataset
from tab2_model_training import model_training_tab


class StreamToLogger:
    """Redirect standard output to logs"""

    def __init__(self, state: ProcessingState):
        self.state = state
        self.terminal = sys.stdout
        self.output = io.StringIO()

    def write(self, text):
        self.terminal.write(text)
        if text.strip():  # Only log non-empty content
            timestamp = datetime.datetime.now().strftime("%H:%M:%S")
            self.state.console_output += f"[{timestamp}] {text}\n"
            self.output.write(text)

    def flush(self):
        self.terminal.flush()
        self.output.flush()


@contextmanager
def capture_output(state: ProcessingState):
    """Context manager to capture console output"""
    stdout_redirector = StreamToLogger(state)
    old_stdout = sys.stdout
    sys.stdout = stdout_redirector
    try:
        yield stdout_redirector.output
    finally:
        sys.stdout = old_stdout


def initialize_session_state():
    """Initialize session state"""
    if "processing_state" not in st.session_state:
        st.session_state.processing_state = ProcessingState()
    if "process_thread" not in st.session_state:
        st.session_state.process_thread = None
    if "show_confirm" not in st.session_state:
        st.session_state.show_confirm = False
    if "selected_datasets" not in st.session_state:
        st.session_state.selected_datasets = set()


class StreamlitCallback(PreprocessingCallback):
    def __init__(self, state: ProcessingState):
        self.state = state

    def on_progress(self, file_path: str, current: int, total: int):
        if file_path:
            self.state.current_file = file_path
        self.state.progress = (current / total * 100) if total > 0 else 0

    def on_status(self, message: str):
        self.state.add_log(message)

    def on_error(self, error: str):
        self.state.error = error
        self.state.add_log(f"Error: {error}")


def save_uploaded_file(uploaded_file, dataset_type: str, data_root: str) -> bool:
    """Save uploaded file to the appropriate directory structure"""
    try:
        type_dir = os.path.join(data_root, dataset_type)
        os.makedirs(type_dir, exist_ok=True)

        file_path = os.path.join(type_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return True
    except Exception as e:
        st.error(f"Error saving file: {str(e)}")
        return False


def get_available_datasets(data_root: str) -> Dict[str, List[str]]:
    """Get all available datasets organized by type"""
    datasets = {}
    type_options = ["zalozba_lampa", "zalozba_film", "cv", "ljspeech"]

    for dataset_type in type_options:
        type_dir = os.path.join(data_root, dataset_type)
        if os.path.exists(type_dir):
            datasets[dataset_type] = [
                os.path.basename(f) for f in glob.glob(os.path.join(type_dir, "*.[z7]*"))
            ]

    return datasets


def process_selected_datasets(
        selected_datasets: Dict[str, List[str]],
        dataset_name: str,
        data_root: str,
        state: ProcessingState
):
    """Process selected datasets"""
    with capture_output(state):
        try:
            print(f"Starting dataset processing: {dataset_name}")
            export_dir = os.path.join(os.getcwd(), "processed_data", dataset_name)
            os.makedirs(export_dir, exist_ok=True)

            data_dirs = []

            # Process each selected dataset
            for dataset_type, files in selected_datasets.items():
                type_dir = os.path.join(data_root, dataset_type)

                for file in files:
                    file_path = os.path.join(type_dir, file)
                    extract_dir = extract_compressed_file(file_path, type_dir)
                    if extract_dir:
                        data_dirs.append((extract_dir, dataset_type))

            if not data_dirs:
                state.error = "No valid data directories found"
                return

            config = PreprocessingConfig(
                data_dirs=data_dirs,
                export_to=export_dir,
                export_as="hf",
                sample_rate=16000,
                audio_format="wav",
                lowercase=True,
            )

            config.state = state
            callback = StreamlitCallback(state)

            success = preprocess_dataset(config, callback)

            if success and not state.should_stop:
                print("Dataset processing completed!")
                state.completion_message = f"Dataset {dataset_name} processing completed!"
            elif state.should_stop:
                print("Processing has been canceled")
                state.add_log("Processing has been canceled")

        except Exception as e:
            state.error = str(e)
            print(f"An error occurred during processing: {str(e)}")
        finally:
            state.is_processing = False
            state.progress = 0
            state.current_file = None


def extract_compressed_file(file_path: str, type_dir: str) -> Optional[str]:
    """Extract compressed file and return the extraction directory"""
    try:
        file_name = os.path.basename(file_path)
        if file_name.lower().endswith(".zip"):
            with zipfile.ZipFile(file_path, "r") as zip_ref:
                extract_dir = os.path.join(type_dir, Path(file_name).stem)
                os.makedirs(extract_dir, exist_ok=True)
                zip_ref.extractall(extract_dir)
        elif file_name.lower().endswith(".7z"):
            extract_dir = os.path.join(type_dir, Path(file_name).stem)
            os.makedirs(extract_dir, exist_ok=True)
            with py7zr.SevenZipFile(file_path, mode="r") as z:
                z.extractall(path=extract_dir)
        else:
            return type_dir

        sub_items = os.listdir(extract_dir)
        if len(sub_items) == 1:
            possible_subdir = os.path.join(extract_dir, sub_items[0])
            if os.path.isdir(possible_subdir):
                return possible_subdir
        return extract_dir
    except Exception as e:
        print(f"Error extracting file {file_path}: {str(e)}")
        return None


def render_upload_section():
    """Render the upload section of the UI"""
    st.subheader("Upload New Dataset Files")

    # File uploader component
    uploaded_files = st.file_uploader(
        "Drag and drop files here",
        accept_multiple_files=True,
        type=["zip", "7z"],
        help="Limit 200MB per file • ZIP, 7Z",
    )

    if uploaded_files:
        type_options = ["zalozba_lampa", "zalozba_film", "cv", "ljspeech"]
        type_selections = st.session_state.processing_state.type_selections

        for file in uploaded_files:
            if file.name not in type_selections:
                type_selections[file.name] = ""

            selected = st.selectbox(
                f"Select Type - {file.name} ({file.size / 1024:.2f} KB)",
                options=["Please select type"] + type_options,
                index=(
                    0
                    if type_selections[file.name] == ""
                    else type_options.index(type_selections[file.name]) + 1
                ),
                key=f"type_select_{file.name}",
            )

            if selected != "Please select type":
                type_selections[file.name] = selected
            else:
                type_selections[file.name] = ""

        # Upload button
        if st.button("Upload Files", type="primary"):
            data_root = os.path.join(os.getcwd(), "mounted_data")

            all_types_selected = all(
                type_selections.get(file.name) in type_options
                for file in uploaded_files
            )

            if not all_types_selected:
                st.error("Please select a type for all uploaded files.")
            else:
                success = True
                for file in uploaded_files:
                    if not save_uploaded_file(
                            file,
                            type_selections[file.name],
                            data_root
                    ):
                        success = False
                        break

                if success:
                    st.success("All files uploaded successfully!")
                    st.rerun()


def render_dataset_processing_section():
    """Render the dataset processing section of the UI"""
    st.subheader("Process Datasets")

    data_root = os.path.join(os.getcwd(), "mounted_data")
    available_datasets = get_available_datasets(data_root)
    state = st.session_state.processing_state

    if not any(available_datasets.values()):
        st.info("No datasets available. Please upload some files first.")
        return

    # Dataset selection
    st.write("Select datasets to process:")
    for dataset_type, files in available_datasets.items():
        if files:
            st.write(f"**{dataset_type}**")
            for file in files:
                key = f"{dataset_type}/{file}"
                if st.checkbox(
                        file,
                        key=key,
                        value=key in st.session_state.selected_datasets
                ):
                    st.session_state.selected_datasets.add(key)
                else:
                    st.session_state.selected_datasets.discard(key)

    # Dataset naming input
    dataset_name = st.text_input(
        "Enter Dataset Name",
        placeholder="e.g., dataset_20240321"
    )

    # Process button
    col1, col2 = st.columns(2)

    with col1:
        if st.button(
                "Process Selected Datasets",
                disabled=state.is_processing,
                type="primary"
        ):
            if not st.session_state.selected_datasets:
                st.error("Please select at least one dataset to process.")
            elif not dataset_name:
                st.error("Please enter a dataset name.")
            else:
                selected = {}
                for key in st.session_state.selected_datasets:
                    dataset_type, file = key.split("/", 1)
                    selected.setdefault(dataset_type, []).append(file)

                state.is_processing = True
                state.should_stop = False
                state.error = None
                state.completion_message = None
                state.log_messages = []
                state.console_output = ""

                st.session_state.process_thread = threading.Thread(
                    target=process_selected_datasets,
                    args=(selected, dataset_name, data_root, state),
                    daemon=True,
                )
                st.session_state.process_thread.start()
                st.rerun()

    # Cancel button
    with col2:
        if state.is_processing:
            if not st.session_state.show_confirm:
                if st.button(
                        "Cancel Processing",
                        type="secondary",
                        key="cancel_btn"
                ):
                    st.session_state.show_confirm = True
            else:
                st.warning("Are you sure you want to cancel processing?")
                confirm_cols = st.columns([1, 1])
                with confirm_cols[0]:
                    if st.button(
                            "Yes, Cancel",
                            type="primary",
                            key="confirm_cancel_btn"
                    ):
                        state.should_stop = True
                        st.session_state.show_confirm = False
                        st.info("Cancellation in progress...")
                        state.terminate_all_processes()

                with confirm_cols[1]:
                    if st.button(
                            "No, Continue",
                            type="secondary",
                            key="deny_cancel_btn"
                    ):
                        st.session_state.show_confirm = False

    # Display processing status
    if state.is_processing:
        progress_bar = st.progress(0)
        if state.current_file:
            st.text(f"Processing: {state.current_file}")
        progress_bar.progress(int(state.progress))

        with st.spinner("Processing..."):
            pass

    # Display completion message or error
    if state.completion_message:
        st.success(state.completion_message)
    if state.error:
        st.error(f"Processing Error: {state.error}")

    # Log output area
    if state.console_output:
        st.subheader("Processing Logs")
        st.code(state.console_output, language="plaintext")

    # Auto-refresh during processing
    if state.is_processing:
        time.sleep(0.5)
        st.rerun()


def main():
    st.set_page_config(page_title="Data Preprocessing Tool", layout="wide")
    st.markdown("""
    <style>
        .reportview-container {
            margin-top: -2em;
        }
        #MainMenu {visibility: hidden;}
        .stDeployButton {display:none;}
        footer {visibility: hidden;}
        #stDecoration {display:none;}
    </style>""", unsafe_allow_html=True)

    initialize_session_state()

    # Create tabs
    tab1, tab2 = st.tabs(["Data Preprocessing", "Model Training"])

    with tab1:
        st.header("Data Preprocessing")

        # Create sections for upload and processing
        render_upload_section()
        st.markdown("---")  # Add a separator
        render_dataset_processing_section()

    with tab2:
        model_training_tab()


if __name__ == "__main__":
    main()