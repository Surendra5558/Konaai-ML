# # Copyright (C) KonaAI - All Rights Reserved
"""Context Evaluation module for automatically identifying relevant modules from user queries."""
from pathlib import Path
from typing import Optional

from src.utils.conf import Setup
from src.utils.instance import Instance
from src.utils.llm_factory import get_llm
from src.utils.metadata import Metadata
from src.utils.status import Status
from src.utils.submodule import Submodule


class ContextEvaluation:
    """Class to handle context evaluation for module identification."""

    def __init__(self, instance: Instance):
        self.instance = instance
        if not instance or not instance.settings.llm_config:
            raise ValueError("Invalid instance or missing LLM configuration.")

        self.llm = get_llm(instance.settings.llm_config)

    @property
    def context_file_path(self) -> Path:
        """Get the file path for storing/loading context evaluation data."""
        file_name = f"context_evaluation_{self.instance.instance_id}.txt"
        return Path(Setup().db_path, self.instance.instance_id, file_name)

    @property
    def filename_path(self) -> Path:
        """Get the path for storing the uploaded filename."""
        file_name = f"context_evaluation_{self.instance.instance_id}_filename.txt"
        return Path(Setup().db_path, self.instance.instance_id, file_name)

    @property
    def exists(self) -> bool:
        """Check if the context file exists."""
        return self.context_file_path.exists()

    def save_uploaded_filename(self, filename: str) -> bool:
        """Save the uploaded filename."""
        try:
            with open(self.filename_path, "w", encoding="utf-8") as f:
                f.write(filename)
            return True
        except Exception:
            return False

    def get_uploaded_filename(self) -> Optional[str]:
        """Get the uploaded filename if it exists."""
        try:
            if self.filename_path.exists():
                with open(self.filename_path, encoding="utf-8") as f:
                    return f.read().strip()
        except Exception:  # nosec B110
            pass
        return None

    def save_context_to_file(self, content: str) -> bool:
        """
        Save the provided text content to the instance's context file.

        This method writes the given content to the file located at
        self.context_file_path using text write mode ("w") with UTF-8 encoding,
        overwriting any existing contents.

        Parameters
        ----------
        content : str
            The text to write into the context file.

        Returns
        -------
        bool
            True if the write operation completed successfully; False if an exception
            occurred during the file operation. On failure, the method logs the error
            via Status.FAILED("Failed to save context to file", self.instance, error=str(e))
            and returns False.

        Notes
        -----
        - All exceptions raised during file I/O are caught; this method does not
          propagate exceptions to the caller.
        - The file is truncated before writing (overwrite behavior).
        - Ensure self.context_file_path is a valid writable path and self.instance is
          available for error reporting.
        """
        try:
            with open(self.context_file_path, "w", encoding="utf-8") as file:
                file.write(content)
            return True
        except Exception as e:
            Status.FAILED("Failed to save context to file", self.instance, error=str(e))
            return False

    def load_context_from_file(self) -> str:
        """
        Load context content from a file.
        Attempts to read and return the contents of the context file specified by
        self.context_file_path. If the file does not exist, logs an info message and
        returns an empty string. If any exception occurs during file reading, logs a
        failure message with error details and returns an empty string.
        Returns:
            str: The content of the context file as a string, or an empty string if
                 the file does not exist or if an error occurs during reading.
        Raises:
            None: Exceptions are caught and logged internally; no exceptions are raised.
        """
        try:
            if not self.context_file_path.exists():
                Status.INFO("Context file does not exist", self.instance)
                return ""

            with open(self.context_file_path, encoding="utf-8") as file:
                return file.read()
        except Exception as e:
            Status.FAILED(
                "Failed to load context from file", self.instance, error=str(e)
            )
            return ""

    def evaluate_context(self, user_message: str) -> Optional[Submodule]:
        """Evaluate the user message to identify relevant module context."""
        try:
            return self._evaluate_context(user_message)
        except Exception as e:
            Status.FAILED("Context evaluation failed", self.instance, error=str(e))
            return None

    def _evaluate_context(self, user_message: str) -> Optional[Submodule]:
        if not user_message:
            Status.INFO("No user message found for context evaluation", self.instance)
            return None

        # Find the module using LLM
        module_name = self._find_module(user_message=user_message)
        if not module_name:
            Status.INFO("No relevant module found for user message", self.instance)
            return None

        # Create and return Submodule with identified module
        submodule = Submodule(instance_id=self.instance.instance_id)
        submodule.module = module_name
        return submodule

    def _find_module(self, user_message: str) -> Optional[str]:
        all_modules = Metadata(self.instance.instance_id).modules
        if not all_modules:
            raise ValueError(
                f"No modules found for instance: {self.instance.instance_id}"
            )

        context_support = self.load_context_from_file()

        # Debug: Log what we're sending to LLM
        Status.INFO(
            f"Context evaluation - Available modules: {', '.join(list(all_modules))}",
            instance=self.instance,
        )
        Status.INFO(
            f"Context evaluation - Module descriptions loaded: {len(context_support) if context_support else 0} characters",
            instance=self.instance,
        )
        if context_support:
            Status.INFO(
                f"Context evaluation - Module descriptions preview: {context_support[:200]}...",
                instance=self.instance,
            )

        prompt = f"""You are a compliance officer evaluating the context of a user request.
                Your task is to identify the most relevant module from the following list that aligns with the user's request.

                Available Modules: {', '.join(list(all_modules))}

                Module Information:
                {context_support or 'No module descriptions provided. Select from available modules based on the user request.'}

                User Request: "{user_message}"

                Instructions:
                1. Analyze the user's request and understand what they are asking about
                2. Review the Module Information to understand what each module represents
                3. Match the user's request to the most appropriate module based on the descriptions provided
                4. If no module description clearly matches the user's request, respond with an empty string ''
        """

        response = (
            self.llm.invoke(prompt) if hasattr(self.llm, "invoke") else self.llm(prompt)
        )
        if not response:
            raise ValueError("LLM did not return a response for user prompt.")

        response: str = (
            response.content if hasattr(response, "content") else str(response)
        )

        # Debug: Log LLM response
        Status.INFO(
            f"Context evaluation - LLM raw response: {response}",
            instance=self.instance,
        )
        response_lower = response.lower()
        for module in all_modules:
            module_lower = module.lower()
            if len(response) > 200:
                end_section = response_lower[-200:]
                if module_lower in end_section:
                    Status.INFO(
                        f"Context evaluation - Found module '{module}' in answer section",
                        instance=self.instance,
                    )
                    return module

        module_mentions = {}
        for module in all_modules:
            module_lower = module.lower()
            count = response_lower.count(module_lower)
            if count > 0:
                module_mentions[module] = count

        if module_mentions:
            best_module = max(module_mentions.items(), key=lambda x: x[1])[0]
            Status.INFO(
                f"Context evaluation - Selected module '{best_module}' based on mentions",
                instance=self.instance,
            )
            return best_module

        return None
