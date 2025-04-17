import requests
import re
import streamlit as st
import json
import os
import subprocess
import shlex
import threading
import time
import sys
import io
import datetime
from dataclasses import dataclass, field
from typing import List, Optional
from contextlib import contextmanager
import multiprocessing

from huggingface_hub import HfApi, login, create_repo


@dataclass
class TrainingState:
    is_training: bool = False
    should_stop: bool = False
    progress: float = 0.0
    error: Optional[str] = None
    completion_message: Optional[str] = None
    log_messages: List[str] = field(default_factory=list)
    console_output: str = ""
    current_progress_line: Optional[str] = None
    last_line_is_progress: bool = False
    training_history: List[str] = field(default_factory=list)  # Stores all training history records
    show_confirm: bool = False  # New flag for cancellation confirmation
    is_canceled: bool = False

    huggingface_token: Optional[str] = None
    repo_id: Optional[str] = None
    is_publishing: bool = False
    publish_error: Optional[str] = None
    publish_success: Optional[str] = None
    publish_output: str = ""

    def clear(self):
        self.__init__()
    def add_log(self, message: str):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        cleaned_message = message.strip()

        # Update condition to detect progress lines
        progress_indicators = ['%', '|', '[', ']']
        is_progress = all(x in cleaned_message for x in progress_indicators)

        if is_progress:
            self.current_progress_line = f"[{timestamp}] {cleaned_message}"
            if self.last_line_is_progress:
                # Replace the last progress line
                lines = self.console_output.rstrip().split('\n')
                if lines:
                    lines.pop()
                lines.append(self.current_progress_line)
                self.console_output = '\n'.join(lines) + '\n'
            else:
                # First occurrence of a progress line
                self.console_output += f"[{timestamp}] {cleaned_message}\n"
                self.last_line_is_progress = True
        else:
            # Non-progress information, add as a new line
            self.last_line_is_progress = False
            self.log_messages.append(f"[{timestamp}] {cleaned_message}")
            self.console_output += f"[{timestamp}] {cleaned_message}\n"

    def complete_training(self, success: bool = True):
        """Call this method when training is completed."""
        self.is_training = False
        if self.should_stop:
            self.is_canceled = True
            self.completion_message = "Training canceled."
        elif success and not self.error:
            self.completion_message = "Training completed successfully!"

        if self.console_output:
            self.training_history.append(self.console_output)


def validate_model_name(name: str) -> tuple[bool, str]:
    if not name:
        return False, "Model name cannot be empty"

    if len(name) > 96:
        return False, "Model name cannot exceed 96 characters"

    if name.startswith(('-', '.')):
        return False, "Model name cannot start with '-' or '.'"
    if name.endswith(('-', '.')):
        return False, "Model name cannot end with '-' or '.'"

    if '--' in name or '..' in name:
        return False, "Model name cannot contain '--' or '..'"

    import re
    valid_pattern = re.compile(r'^[a-zA-Z0-9][a-zA-Z0-9-_.]*[a-zA-Z0-9]$|^[a-zA-Z0-9]$')
    if not valid_pattern.match(name):
        return False, "Model name can only contain alphanumeric characters, '-', '_', and '.'"

    name = name.replace('—', '-').replace('–', '-')

    return True, name
class StreamToLogger:
    """Redirect standard output to logs"""

    def __init__(self, state: TrainingState):
        self.state = state
        self.terminal = sys.stdout
        self.output = io.StringIO()
        self.buffer = ""

    def write(self, text):
        # Replace the specific warning message with empty string
        text = text.replace("Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.", "")
        self.terminal.write(text)
        # \x1b is the escape character, \[A matches the cursor up sequence
        text = re.sub(r'\x1b\[A', '', text)
        # Process any non-empty output
        if text:
            # Handle multi-line outputs
            lines = text.splitlines(True)  # Keep line breaks
            for line in lines:
                if line.endswith('\n'):
                    # Complete line, process buffer and current line
                    full_line = self.buffer + line
                    self.buffer = ""
                    if full_line.strip():
                        self.state.add_log(full_line)
                else:
                    # Incomplete line, add to buffer
                    self.buffer += line

    def flush(self):
        self.terminal.flush()
        if self.buffer:  # Ensure the last buffered line is processed
            self.state.add_log(self.buffer)
            self.buffer = ""


@contextmanager
def capture_output(state: TrainingState):
    """Context manager to capture console output"""
    stdout_redirector = StreamToLogger(state)
    old_stdout = sys.stdout
    sys.stdout = stdout_redirector
    try:
        yield stdout_redirector.output
    finally:
        sys.stdout = old_stdout


def get_available_models():
    base_models = [
        "openai/whisper-tiny",
        "openai/whisper-base",
        "openai/whisper-small",
        "openai/whisper-medium",
        "openai/whisper-large"
    ]

    training_dir = os.path.join(os.getcwd(), "training", "models")
    custom_models = []
    if os.path.exists(training_dir):
        custom_models = [d for d in os.listdir(training_dir) if os.path.isdir(os.path.join(training_dir, d))]

    return base_models + custom_models

def initialize_training_state():
    """Initialize training state in session state"""
    if 'training_state' not in st.session_state:
        st.session_state.training_state = TrainingState()
    if 'training_thread' not in st.session_state:
        st.session_state.training_thread = None
    # Load saved HuggingFace token if exists
    if os.path.exists("huggingface_config.json"):
        try:
            with open("huggingface_config.json", "r") as f:
                config = json.load(f)
                st.session_state.training_state.huggingface_token = config.get("token")
        except:
            pass
def publish_to_huggingface(state, model_path, repo_id, token, overwrite_existing, tag_option, custom_tag):
    try:
        api = HfApi()

        try:
            api.repo_info(repo_id=repo_id, token=token)
            repo_exists = True
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                repo_exists = False
            else:
                raise

        if repo_exists:
            if not overwrite_existing:
                raise Exception(f"Repository {repo_id} already exists. Set 'Overwrite existing repository' to update it.")
            state.publish_output += f"Repository {repo_id} already exists. Updating...\n"
        else:
            state.publish_output += f"Creating new repository: {repo_id}\n"
            create_repo(repo_id, token=token, private=True)

        if tag_option == "Auto (Date-Time)":
            current_time = datetime.datetime.now()
            tag = current_time.strftime("%Y-%m-%d_%H%M")
        else:
            tag = custom_tag

        state.publish_output += f"Uploading model files...\n"
        api.upload_folder(
            folder_path=model_path,
            repo_id=repo_id,
            repo_type="model",
            token=token,
            commit_message=f"Upload model files with tag: {tag}",
            revision="main"
        )

        api.create_tag(repo_id=repo_id, tag=tag, token=token)

        state.publish_output += f"Model files uploaded and tagged with version: {tag}\n"
        state.publish_success = f"Model successfully published to {repo_id} with tag {tag}"
    except Exception as e:
        state.publish_error = f"Error publishing model: {str(e)}"
    finally:
        state.is_publishing = False




def run_training_script(args_path: str, state: TrainingState):
    """Run the training script with stdout redirection."""
    with capture_output(state):
        try:
            cmd = f"python -u train_whisper.py {args_path}"
            print(f"Running command: {cmd}")

            process = subprocess.Popen(
                shlex.split(cmd),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )

            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    print(output.strip())

                if state.should_stop:
                    process.terminate()
                    print("Training canceled.")
                    break

            returncode = process.poll()
            if returncode != 0 and not state.should_stop:
                error_msg = f"Training script error, return code {returncode}."
                print(error_msg)
                state.error = error_msg
            elif not state.should_stop:
                completion_msg = "Training completed successfully!"
                print(completion_msg)
                state.completion_message = completion_msg

        except Exception as e:
            error_msg = f"Error running training script: {str(e)}"
            print(error_msg)
            state.error = error_msg
        finally:
            # Use the new completion method instead of directly setting the state
            state.complete_training(success=state.error is None)


def get_available_datasets():
    """Get all datasets in the processed_data catalog"""
    processed_data_path = os.path.join(os.getcwd(), "processed_data")
    if not os.path.exists(processed_data_path):
        return []

    # Returns only the directory containing metadata.csv as a valid dataset
    datasets = []
    for dataset in os.listdir(processed_data_path):
        dataset_path = os.path.join(processed_data_path, dataset)
        if os.path.isdir(dataset_path) and os.path.exists(os.path.join(dataset_path, "metadata.csv")):
            datasets.append(dataset)

    return sorted(datasets)


def model_training_tab():
    initialize_training_state()
    state: TrainingState = st.session_state.training_state

    st.header("Model Training and Publishing")

    # HuggingFace Configuration Section
    with st.expander("HuggingFace Configuration"):
        hf_token = st.text_input(
            "HuggingFace Token",
            type="password",
            value=state.huggingface_token or "",
            help="Enter your HuggingFace access token"
        )

        if st.button("Save HuggingFace Token"):
            state.huggingface_token = hf_token
            with open("huggingface_config.json", "w") as f:
                json.dump({"token": hf_token}, f)
            st.success("HuggingFace token saved successfully!")

    st.subheader("Training Configuration")
    cpu_count = multiprocessing.cpu_count()

    # Initialize default values or load existing configuration
    default_config = {
        "model_id": "openai/whisper-medium",
        "task": "transcribe",
        "training_data_dir": "",
        "metadata_file": "metadata.csv",
        "output_dir": "",
        "per_device_train_batch_size": 8,
        "per_device_eval_batch_size": 8,
        "gradient_accumulation_steps": 1,
        "learning_rate": 6.25e-06,
        "warmup_steps": 1000,
        "num_train_epochs": 1,
        "evaluation_strategy": "epoch",
        "logging_strategy": "epoch",
        "save_strategy": "epoch",
        "predict_with_generate": True,
        "generation_max_length": 225,
        "report_to": ["tensorboard"],
        "load_best_model_at_end": True,
        "metric_for_best_model": "wer",
        "greater_is_better": False,
        "dataloader_num_workers": cpu_count,
        "save_total_limit": 3,
        "seed": 42,
        "data_seed": 42,
        "bf16": True,
        "fp16": False
    }

    # Load existing config if available
    config_path = "training_args.json"
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            try:
                existing_config = json.load(f)
                default_config.update(existing_config)
            except json.JSONDecodeError:
                st.warning("Existing training_args.json is invalid. Using default values.")

    # Create input widgets for configuration
    available_models = get_available_models()
    model_index = available_models.index(default_config["model_id"]) if default_config[
                                                                            "model_id"] in available_models else available_models.index(
        "openai/whisper-medium")
    model_id = st.selectbox(
        "Select Model",
        options=available_models,
        index=model_index,
        help="Select a base model or previously trained model"
    )

    model_name = st.text_input(
        "Model Name",
        value="",
        placeholder="e.g., my-model-v1, test_model_1",
        help="Model name must:\n"
             "• Only contain letters, numbers, hyphens (-), underscores (_), and dots (.)\n"
             "• Not start or end with hyphens or dots\n"
             "• Not contain consecutive hyphens (--) or dots (..)\n"
             "• Be 96 characters or less"
    )

    available_datasets = get_available_datasets()
    if not available_datasets:
        st.warning("No processed datasets found in processed_data directory. Please process some data first.")
        selected_dataset = None
    else:
        current_dataset = ""
        if default_config["training_data_dir"]:
            current_dataset = os.path.basename(os.path.normpath(default_config["training_data_dir"]))

        default_index = available_datasets.index(current_dataset) if current_dataset in available_datasets else 0

        selected_dataset = st.selectbox(
            "Select Dataset",
            options=available_datasets,
            index=default_index,
            help="Select a processed dataset for training"
        )
    training_data_dir = os.path.join(os.getcwd(), "processed_data", selected_dataset) if selected_dataset else ""
    output_dir = os.path.join(os.getcwd(), "training", "models", model_name) if model_name else ""

    with st.expander("Advanced Configuration"):
        per_device_train_batch_size = st.number_input("Per Device Train Batch Size", min_value=1, step=1,
                                                      value=default_config["per_device_train_batch_size"])
        per_device_eval_batch_size = st.number_input("Per Device Eval Batch Size", min_value=1, step=1,
                                                     value=default_config["per_device_eval_batch_size"])
        gradient_accumulation_steps = st.number_input("Gradient Accumulation Steps", min_value=1, step=1,
                                                      value=default_config["gradient_accumulation_steps"])
        learning_rate = st.number_input("Learning Rate", min_value=1e-8, step=1e-6, format="%.8f",
                                        value=default_config["learning_rate"])
        warmup_steps = st.number_input("Warmup Steps", min_value=0, step=100, value=default_config["warmup_steps"])
        num_train_epochs = st.number_input("Number of Training Epochs", min_value=1, step=1,
                                           value=default_config["num_train_epochs"])

        col1, col2, col3 = st.columns(3)
        with col1:
            eval_strategy = st.selectbox("Evaluation Strategy", options=["no", "steps", "epoch"],
                                         index=["no", "steps", "epoch"].index(default_config["evaluation_strategy"]))
        with col2:
            logging_strategy = st.selectbox("Logging Strategy", options=["no", "steps", "epoch"],
                                            index=["no", "steps", "epoch"].index(default_config["logging_strategy"]))
        with col3:
            save_strategy = st.selectbox("Save Strategy", options=["no", "steps", "epoch"],
                                         index=["no", "steps", "epoch"].index(default_config["save_strategy"]))

        predict_with_generate = st.checkbox("Predict with Generate", value=default_config["predict_with_generate"])
        generation_max_length = st.number_input("Generation Max Length", min_value=1, step=1,
                                                value=default_config["generation_max_length"])

        report_to_options = ["tensorboard", "wandb", "mlflow", "comet_ml"]
        report_to = st.multiselect("Report To", options=report_to_options, default=default_config["report_to"])

        col1, col2 = st.columns(2)
        with col1:
            load_best_model_at_end = st.checkbox("Load Best Model at End",
                                                 value=default_config["load_best_model_at_end"])
            greater_is_better = st.checkbox("Greater is Better", value=default_config["greater_is_better"])
            bf16 = st.checkbox("Use BF16", value=default_config["bf16"], help="Enable bfloat16 training")
        with col2:
            metric_for_best_model = st.text_input("Metric for Best Model",
                                                  value=default_config["metric_for_best_model"])
            fp16 = st.checkbox("Use FP16", value=default_config["fp16"], help="Enable fp16 training")

        col1, col2, col3 = st.columns(3)
        with col1:
            dataloader_num_workers = st.number_input("Dataloader Num Workers", min_value=1, max_value=multiprocessing.cpu_count(), step=1,
                                                     value=default_config["dataloader_num_workers"])
        with col2:
            save_total_limit = st.number_input("Save Total Limit", min_value=1, step=1,
                                               value=default_config["save_total_limit"])
        with col3:
            seed = st.number_input("Seed", min_value=0, step=1, value=default_config["seed"])
            data_seed = st.number_input("Data Seed", min_value=0, step=1, value=default_config["data_seed"])

    # Save Configuration Button
    if st.button("Save Configuration"):
        if not model_name:
            st.error("Please enter a model name.")
            return

        is_valid, result = validate_model_name(model_name)
        if not is_valid:
            st.error(f"Invalid model name: {result}")
            return

        model_name = result if isinstance(result, str) else model_name

        config = {
            "model_id": model_id,
            "task": default_config["task"],
            "training_data_dir": training_data_dir,
            "metadata_file": default_config["metadata_file"],
            "output_dir": os.path.join(os.getcwd(), "training", "models", model_name),
            "per_device_train_batch_size": per_device_train_batch_size,
            "per_device_eval_batch_size": per_device_eval_batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "learning_rate": learning_rate,
            "warmup_steps": warmup_steps,
            "num_train_epochs": num_train_epochs,
            "evaluation_strategy": eval_strategy,
            "logging_strategy": logging_strategy,
            "save_strategy": save_strategy,
            "predict_with_generate": predict_with_generate,
            "generation_max_length": generation_max_length,
            "report_to": report_to,
            "load_best_model_at_end": load_best_model_at_end,
            "metric_for_best_model": metric_for_best_model,
            "greater_is_better": greater_is_better,
            "dataloader_num_workers": dataloader_num_workers,
            "save_total_limit": save_total_limit,
            "seed": seed,
            "data_seed": data_seed,
            "bf16": bf16,
            "fp16": fp16
        }

        try:
            if os.path.exists(output_dir):
                st.warning("A model with this name already exists. Training will continue from the existing model.")

            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=4)
            state.clear()
            st.success("Training configuration saved successfully.")
        except Exception as e:
            st.error(f"Failed to save configuration: {str(e)}")

    st.markdown("---")

    st.subheader("Start Training")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Start Training", disabled=state.is_training):
            if not os.path.exists(config_path):
                st.error("Please save the training configuration first.")
            else:
                state.clear()
                state.is_training = True

                st.session_state.training_thread = threading.Thread(
                    target=run_training_script,
                    args=(config_path, state),
                    daemon=True
                )
                st.session_state.training_thread.start()
                st.rerun()

    with col2:
        if state.is_training:
            if not state.show_confirm:
                if st.button("Cancel Training", type="secondary", key="cancel_training_btn"):
                    state.show_confirm = True
            else:
                # Display confirmation prompt
                st.warning("Are you sure you want to cancel training?")
                confirm_cols = st.columns([1, 1])
                with confirm_cols[0]:
                    if st.button("Yes, Cancel", type="primary", key="confirm_cancel_training_btn"):
                        state.should_stop = True
                        state.show_confirm = False
                        st.info("Cancellation in progress...")
                with confirm_cols[1]:
                    if st.button("No, Continue", type="secondary", key="deny_cancel_training_btn"):
                        state.show_confirm = False

    if state.is_training:
        st.info("Training in progress...")
    elif state.is_canceled:
        st.warning(state.completion_message)
    elif state.completion_message and not state.error:
        st.success(state.completion_message)
    elif state.error:
        st.error(state.error)

    st.text("Training logs:")
    # Create custom styles for the log container
    st.markdown("""
                <style>
                .log-container {
                    height: 400px;
                    overflow-y: scroll;
                    margin: 10px 0px;
                    padding: 10px;
                    background-color: #f8f9fa;
                    border: 1px solid #dee2e6;
                    border-radius: 4px;
                    font-family: 'Courier New', Courier, monospace;
                    font-size: 14px;
                    line-height: 1.5;
                    white-space: pre;
                    word-wrap: normal;
                }
                </style>
                <script>
                    const observer = new MutationObserver((mutations) => {
                        const container = document.querySelector('.log-container');
                        if (container) {
                            container.scrollTop = container.scrollHeight;
                        }
                    });

                    observer.observe(document.body, {
                        childList: true,
                        subtree: true
                    });
                </script>
                """, unsafe_allow_html=True)

    # Display logs in scrollable container
    st.markdown(
        f'<div class="log-container">{state.console_output}</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
        <div style="display: none">end-anchor</div>
        """, unsafe_allow_html=True)

    # Publish Model Section
    st.markdown("---")
    st.subheader("Publish Model to HuggingFace")

    if not state.huggingface_token:
        st.warning("Please configure your HuggingFace token first")
    else:
        available_models = get_available_models()
        custom_models = [model for model in available_models if not model.startswith("openai/whisper")]

        if not custom_models:
            st.warning("No custom models available for publishing.")
        else:
            selected_model = st.selectbox(
                "Select Model to Publish",
                options=custom_models,
                help="Select a model to publish to HuggingFace"
            )

            repo_id = st.text_input(
                "Repository ID",
                value=state.repo_id or "",
                placeholder="username/model-name e.g., gate5/wip_test_1",
                help="Format: username/model-name"
            )
            # Add version tagging options
            tag_option = st.radio(
                "Version Tag Option",
                options=["Auto (Date-Time)", "Custom"],
                index=0
            )
            custom_tag = ""
            if tag_option == "Custom":
                custom_tag = st.text_input(
                    "Custom Version Tag",
                    placeholder="e.g., v1.0.0 or 2023-05-15_release",
                    help="Enter a custom version tag for this model. Avoid using ':' in the tag."
                )

            overwrite_existing = st.checkbox("Update existing repository if it exists", value=True)
            if st.button("Publish Model", disabled=state.is_publishing):
                if not repo_id:
                    st.error("Please enter a repository ID")
                elif tag_option == "Custom" and not custom_tag:
                    st.error("Please enter a custom version tag")
                elif tag_option == "Custom" and ':' in custom_tag:
                    st.error("Custom tag cannot contain ':'")
                else:
                    state.is_publishing = True
                    state.repo_id = repo_id
                    state.publish_output = ""
                    state.publish_success = ""
                    state.publish_error = ""

                    model_path = os.path.join(os.getcwd(), "training", "models", selected_model)

                    publishing_thread = threading.Thread(
                        target=publish_to_huggingface,
                        args=(state, model_path, repo_id, state.huggingface_token, overwrite_existing, tag_option,
                              custom_tag),
                        daemon=True
                    )
                    publishing_thread.start()
                    st.rerun()

            if state.is_publishing:
                st.info("Publishing model to HuggingFace...")

                # Display publish logs
                st.text("Publish logs:")
                st.markdown(
                    f'<div class="log-container">{state.publish_output}</div>',
                    unsafe_allow_html=True
                )

            if state.publish_error:
                st.error(state.publish_error)
            if state.publish_success:
                st.success(state.publish_success)

        # Trigger page refresh while training or publishing
    if state.is_training or state.is_publishing:
        time.sleep(0.5)
        st.rerun()

        # Anchor for auto-scrolling
        st.markdown('<div id="end-of-page"></div>', unsafe_allow_html=True)

        # JavaScript for auto-scrolling
        st.markdown("""
                <script>
                    function scroll_to_bottom() {
                        var end_of_page = document.getElementById('end-of-page');
                        if (end_of_page) {
                            end_of_page.scrollIntoView();
                        }
                    }
                    scroll_to_bottom();
                </script>
            """, unsafe_allow_html=True)

