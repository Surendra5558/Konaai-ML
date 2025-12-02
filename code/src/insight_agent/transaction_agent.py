# # Copyright (C) KonaAI - All Rights Reserved
"""Transaction Agent Module"""
import asyncio
import csv
from typing import Union

import pandas as pd
from src.automl.explainer import ExplainationOutput
from src.insight_agent.audit_data import AuditData
from src.insight_agent.constants import CONTRIBUTION_PERCENT_THRESHOLD
from src.utils.global_config import GlobalSettings
from src.utils.instance import Instance
from src.utils.llm_factory import BaseLanguageModel
from src.utils.llm_factory import ClaudeLLM
from src.utils.llm_factory import get_llm
from src.utils.status import Status
from src.utils.submodule import Submodule


class TransactionAgent:
    """Transaction Agent for generating transaction summaries using LLMs"""

    sub_obj: Submodule = None
    transaction_id: str = ""
    llm: Union[BaseLanguageModel, ClaudeLLM] = None
    audit_data_obj: AuditData = None

    def __init__(self, submodule: Submodule, transaction_id: str):
        self.sub_obj = submodule
        self.transaction_id = transaction_id
        instance: Instance = GlobalSettings.instance_by_id(submodule.instance_id)
        if not instance or not instance.settings.llm_config:
            raise ValueError(
                f"LLM configuration not found for the instance {submodule.instance_id}"
            )

        # Initialize LLM based on instance settings
        self.llm = get_llm(llm_config=instance.settings.llm_config)
        if not self.llm:
            raise ValueError("Failed to initialize LLM. Check LLM configuration.")

    def _create_patterns_text(self) -> str:
        # Fetch patterns data
        patterns_df: pd.DataFrame = self.audit_data_obj.fetch_patterns()
        if patterns_df is None:
            raise ValueError("No patterns data found for audit agent")

        if len(patterns_df) == 0:
            return "No patterns detected for this transaction."

        Status.INFO(
            f"Fetched {len(patterns_df)} patterns for transaction {self.transaction_id}"
        )
        return patterns_df.to_csv(
            index=False,
            header=True,
            lineterminator="\n",
            quoting=csv.QUOTE_STRINGS,
        )

    def generate_summary_report(self) -> str:
        """Generate a summary report for the transaction using LLM."""
        try:
            Status.INFO(
                "Generating summary report based on patterns data",
                self.sub_obj,
                transaction_id=self.transaction_id,
            )
            return asyncio.run(self._generate_summary_report())
        except Exception as e:
            Status.FAILED("Error generating summary report", error=str(e))
            return "Error generating summary report"

    async def _generate_summary_report(self) -> str:
        """Generate a summary report for the transaction using LLM."""
        self.audit_data_obj: AuditData = AuditData(
            submodule=self.sub_obj, transaction_id=int(self.transaction_id)
        )

        pattern_str = self._create_patterns_text()

        # create async call to generate both reports concurrently
        patterns_report_task = asyncio.to_thread(
            self._create_patterns_summary_report, pattern_str
        )
        automl_report_task = asyncio.to_thread(self._create_automl_explanation_report)

        # wait for both tasks to complete and return the results
        patterns_report = await patterns_report_task
        automl_report = await automl_report_task
        return f"{patterns_report}\\n\\n{automl_report}"

    def generate_full_audit_report(self) -> str:
        """Generate a full audit report for the transaction using LLM."""
        try:
            Status.INFO(
                "Generating full audit report",
                self.sub_obj,
                transaction_id=self.transaction_id,
            )
            return asyncio.run(self._generate_full_audit_report())
        except Exception as e:
            Status.FAILED("Error generating full audit report", error=str(e))
            return "Error generating full audit report"

    async def _generate_full_audit_report(self) -> str:
        """Generate a full audit report for the transaction using LLM."""
        self.audit_data_obj: AuditData = AuditData(
            submodule=self.sub_obj, transaction_id=int(self.transaction_id)
        )

        pattern_str = self._create_patterns_text()

        # create async call to generate both reports concurrently
        patterns_summary_task = asyncio.to_thread(
            self._create_patterns_summary_report, pattern_str
        )
        automl_report_task = asyncio.to_thread(self._create_automl_explanation_report)
        patterns_report_task = asyncio.to_thread(
            self._create_patterns_full_report, pattern_str
        )

        # wait for both tasks to complete and return the results
        patterns_summary = await patterns_summary_task
        automl_report = await automl_report_task
        patterns_report = await patterns_report_task
        return f"{patterns_summary}\\n\\n{automl_report}\\n\\n{patterns_report}"

    def _create_patterns_full_report(self, patterns_str: str) -> str:
        """Create a full report based on patterns data using the LLM."""

        prompt = self._create_patterns_full_prompt(patterns_str=patterns_str)
        response = (
            self.llm.invoke(prompt) if hasattr(self.llm, "invoke") else self.llm(prompt)
        )
        if not response:
            raise ValueError("LLM did not return a response for patterns report")

        return response.content if hasattr(response, "content") else str(response)

    def _create_patterns_summary_report(self, patterns_str: str) -> str:
        """Create a report based on patterns data using the LLM."""
        prompt = self._create_patterns_summary_prompt(patterns_str=patterns_str)
        response = (
            self.llm.invoke(prompt) if hasattr(self.llm, "invoke") else self.llm(prompt)
        )
        if not response:
            raise ValueError("LLM did not return a response for patterns report")

        return response.content if hasattr(response, "content") else str(response)

    def _create_automl_explanation_report(self) -> str:
        """Create a report based on AutoML explanations using the LLM."""
        # Fetch AutoML explanations
        explanation: (  # pylint: disable=unused-variable # noqa: F841
            ExplainationOutput
        ) = self.audit_data_obj.fetch_automl_explanation()

        if explanation is None:
            return "No AutoML explanation data found for audit agent"

        # No explanation needed if no concerns predicted
        if not explanation.predicted_concern:
            return (
                "Machine learning model did not flag any concerns for this transaction."
            )
        result = {
            "prediction_date": (
                explanation.prediction_date.isoformat()
                if explanation.prediction_date is not None
                else None
            ),
            "predicted_concern": explanation.predicted_concern,
            "prediction_probability": explanation.prediction_probability,
            "decision_threshold": explanation.decision_threshold,
            "features": [
                {
                    "feature_name": feature.name,
                    "feature_value": feature.value,
                    "feature_description": feature.description,
                    "contribution_percent": feature.contribution_percent,
                }
                for feature in explanation.features
                if feature.contribution_percent > CONTRIBUTION_PERCENT_THRESHOLD
            ],
        }

        # convert result to string for prompt
        result_str = str(result)

        prompt = self._create_automl_explanation_prompt(explanation_str=result_str)
        response = (
            self.llm.invoke(prompt) if hasattr(self.llm, "invoke") else self.llm(prompt)
        )

        if not response:
            return "LLM did not return a response for AutoML explanation report"

        return response.content if hasattr(response, "content") else str(response)

    def _create_patterns_full_prompt(self, patterns_str: str) -> str:
        """Create prompt for full patterns report."""
        format_instructions = """
            CRITICAL: You MUST respond in the EXACT format below. Do not deviate from this structure:

            **DETAILED AUDIT FINDINGS**

            For each finding, provide:
            1. Finding: [PatternID] - [Pattern Description]
            o Risk Type: [Regulatory/Fraud Monitoring/Process Inefficiency/etc.]
            o Risk Level: [High/Medium/Low]
            o Remediation: [Specific remediation steps]
            o Best Practices: [Preventive measures]

            IMPORTANT:
            - Use the exact format above for each finding
            - Determine risk type based on pattern category
            - Assess risk level based on weightage and pattern type
            - Provide specific, actionable remediation steps
            - Suggest practical best practices
            - Use this EXACT format for every response
            - Do not add extra text before or after this structure.
        """

        return f"""
        You are a compliance officer and audit analyst with comprehensive knowledge. You can answer ANY question about audit tests, risk analysis, compliance, and process improvement.

        PATTERN DATA:
        {patterns_str}

        INSTRUCTIONS:
        {format_instructions}
        Answer the user's question in the specified format:
        """

    def _create_patterns_summary_prompt(self, patterns_str: str) -> str:
        """Create prompt for patterns summary."""
        format_instructions = """
            CRITICAL: You MUST respond in the EXACT format below. Do not deviate from this structure:

            **EXECUTIVE SUMMARY BY RISK TYPE**

            **Regulatory Risks:**
            • [Group related tests by risk category with PatternID ranges, e.g., "Invoices containing sensitive keywords or payments to government-flagged vendors (P2PACIN101, P2PACIN102)"]:
            [Explanation of regulatory exposure]
            Remediation: [Specific remediation steps]
            Best Practice: [Preventive measures]

            **Fraud Risks:**
            • [Group related tests by risk category with PatternID ranges, e.g., "Duplicate invoices/payments and abnormal timing (P2PFMIN220-223, P2PFMIN211-214)"]:
            [Explanation of fraud potential]
            Remediation: [Specific remediation steps]
            Best Practice: [Preventive measures]

            **Process Inefficiency:**
            • [Group related tests by risk category with PatternID ranges, e.g., "Mismatched invoice/payment dates (P2PFMIN213, P2PFMIN214)"]:
            [Explanation of process gaps]
            Remediation: [Specific remediation steps]
            Best Practice: [Preventive measures]

            IMPORTANT:
            - Group multiple related PatternIDs together with descriptive categories
            - Use PatternID ranges (e.g., P2PFMIN220223) when multiple consecutive tests are similar
            - Focus on the risk category and submodule relationships
            - Use this EXACT format for every response. Do not add extra text before or after this structure.
            """

        return f"""
            You are a compliance officer and audit analyst with comprehensive knowledge. You can answer ANY question about audit tests, risk analysis, compliance, and process improvement.
            {patterns_str}

            INSTRUCTIONS:
            {format_instructions}

            Answer the user's question in the specified format:
        """

    def _create_automl_explanation_prompt(self, explanation_str: str) -> str:
        """Create prompt for AutoML explanation summary."""
        return f"""
            You are a compliance officer and audit analyst with comprehensive knowledge. You can answer ANY question about audit tests, risk analysis, compliance, and process improvement.
            {explanation_str}

            INSTRUCTIONS:
            Provide a concise summary of the machine learning model's prediction, highlighting key contributing features and their impact on the decision.

            Answer the user's question in a clear and structured manner.
        """
