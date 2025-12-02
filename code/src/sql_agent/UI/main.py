# # Copyright (C) KonaAI - All Rights Reserved
"""Class to handle the UI for the chat conversation."""
import json
from typing import Optional

import pandas as pd
from nicegui import run
from nicegui import ui
from src.admin import theme
from src.admin.components.active_instance import ActiveInstanceUI
from src.sql_agent.api import sql_agent_conversation
from src.sql_agent.UI.configuration import display_configuration
from src.utils.agent_models import AgentResponseModel
from src.utils.agent_models import FollowUpQuestion
from src.utils.agent_models import Message
from src.utils.global_config import GlobalSettings
from src.utils.instance import Instance
from src.utils.status import Status
from src.utils.submodule import Submodule


class ConversationUI:
    """Class to handle the UI for the chat conversation."""

    conversation: AgentResponseModel = None

    input_box: ui.input = None
    spinner: Optional[ui.element] = None
    scroll_area: Optional[ui.scroll_area] = None
    instance: Optional[Instance] = None

    _client_id: str
    _project_id: str
    _processing_message: bool = False

    async def render(self):
        """
        Renders the UI for the KonaAI Companion chat interface.
        """
        with theme.frame("SQL Agent"):
            # Configuration button will go here
            with ui.dialog() as dialog, ui.card().classes("w-full h-full gap-2"):
                ui.markdown("## Configure Settings")
                await display_configuration()
                ui.space().classes("grow")
                ui.button("Close", on_click=dialog.close).classes("self-end mt-4")

            with ui.row(align_items="center").classes(
                "w-full bg-gradient-to-r from-blue-600 to-purple-600 text-white gap-0"
            ):
                # SQL Agent Header
                ui.markdown("## 🤖 SQL Agent").classes("m-0")
                if not GlobalSettings().active_instance_id:
                    ui.notify(
                        "Please activate an instance to continue", type="negative"
                    )
                    return

                if not self.conversation:
                    self._initiate_conversation()

                self.instance = GlobalSettings.instance_by_id(
                    GlobalSettings().active_instance_id
                )

                # New Chat button
                ui.button(
                    text="New Chat",
                    icon="add_comment",
                    on_click=self._start_new_chat,
                ).classes("ml-auto mr-2")

                # configuration button
                ui.button(
                    text="Configure",
                    icon="settings",
                    on_click=dialog.open,
                ).classes("mr-2")

            instanceui = ActiveInstanceUI()
            if not instanceui.active_instance:
                with ui.card().classes("w-full p-8 text-center"):
                    ui.icon("warning", size="3rem", color="orange")
                    ui.markdown("## No Active Instance")
                    ui.label(
                        "Please select or configure an active instance to continue."
                    )
                return

            # load active instance
            instanceui.render()

            if not self.conversation:
                return

            # Chat container with better styling
            with ui.card().classes("w-full h-full gap-0"):
                with ui.card_section().classes("w-full h-full gap-0"):
                    with ui.row().classes("w-full"):
                        ui.icon("chat_bubble_outline", size="sm", color="gray")
                        ui.label("Conversation").classes(
                            "text-lg font-semibold text-gray-800"
                        )

                    self._draw_conversation()

    async def _handle_message(self, message: str):
        """Handles sending a user message and updating the conversation."""
        msg = Message(role="user", content=message)
        self.conversation.add_message(msg)
        await self._send_request()

    async def _send_request(self):
        self._processing_message = True
        self._draw_conversation.refresh()
        # get response async
        response = await run.io_bound(
            lambda: sql_agent_conversation(
                client_id=self._client_id,
                project_id=self._project_id,
                model=self.conversation,
            )
        )
        self.conversation = response
        self._processing_message = False
        self._draw_conversation.refresh()

    def _initiate_conversation(self) -> AgentResponseModel:
        if not GlobalSettings().active_instance_id:
            raise ValueError("No active instance found.")

        active_instance: Instance = GlobalSettings.instance_by_id(
            GlobalSettings().active_instance_id
        )
        if not active_instance:
            raise ValueError("Active instance ID is invalid.")

        conversation = AgentResponseModel()
        conversation.context = Submodule(instance_id=active_instance.instance_id)
        self.conversation = conversation

        self._client_id = active_instance.ClientUID
        self._project_id = active_instance.ProjectUID

    def _start_new_chat(self):
        """Starts a new conversation by resetting the current conversation."""
        try:
            # Reset the conversation
            self._initiate_conversation()
            # Refresh the conversation UI to show the empty chat
            self._processing_message = False
            self._draw_conversation.refresh()
            ui.notify("New chat started", type="positive")
        except Exception as e:
            Status.FAILED("Failed to start new chat", error=str(e))
            ui.notify("Failed to start new chat. Please try again.", type="negative")

    @ui.refreshable_method
    def _draw_conversation(self):
        """Draws the conversation UI with messages and input area."""
        if not self.conversation:
            return

        # Calculate dynamic height based on number of messages and question
        message_count = (
            len(self.conversation.messages)
            if self.conversation and self.conversation.messages
            else 0
        )
        has_question = (
            self.conversation and self.conversation.pending_follow_up_questions()
        )

        # Create a scrollable container for messages with dynamic height
        # Account for both messages and question in height calculation
        height_class = "h-full" if (message_count > 0 or has_question) else "h-0"
        self.scroll_area = (
            ui.scroll_area()
            .classes(
                f"w-full {height_class} overflow-y-scroll transition-all duration-300 mb-4"
            )
            .style(
                """
            flex-grow: 1;
            flex-shrink: 1;
            scrollbar-width: thin;
            scrollbar-color: #cbd5e0 #f7fafc;
        """
            )
            .props("visible-scrollbar")
        )

        with self.scroll_area:
            # show existing messages
            if self.conversation and self.conversation.messages:
                self._draw_messages()

            if self._processing_message:
                with ui.row(align_items="stretch").classes("w-full justify-start mt-4"):
                    ui.spinner(type="dots").classes("text-blue-600")
                    ui.label("Assistant is typing...").classes("text-blue-600 ml-2")
            elif has_question:
                self._draw_followup()

        # Auto-scroll to bottom after rendering
        if self.scroll_area:
            self.scroll_area.scroll_to(percent=1.0)

        # draw input area
        if not self._processing_message:
            self._draw_input_area()

    def _draw_messages(self):
        """Draws the chat messages in the UI with enhanced styling."""
        with ui.column().classes("w-full gap-0"):
            for msg in self.conversation.messages:
                if msg.role == "user":
                    # User messages on the right with better styling
                    with ui.row().classes("w-full justify-end"):
                        with ui.element("div").classes(
                            "bg-blue-600 text-white p-4 rounded-2xl rounded-br-md max-w-xs shadow-md"
                        ):
                            ui.label(msg.content).classes("text-sm leading-relaxed")
                            # show the timestamp
                            ui.label(msg.created_at.strftime("%H:%M")).classes(
                                "text-xs text-blue-200 mt-2 block text-right"
                            )
                else:
                    # Assistant messages on the left with better styling
                    with ui.row().classes("w-full justify-start"):
                        # Make the chat bubble wider to accommodate tables
                        with ui.element("div").classes(
                            "bg-gray-100 text-gray-800 p-4 rounded-2xl rounded-bl-md max-w-xl shadow-md border"
                        ):
                            if msg.content_type == "data_table":
                                # Handle table data from backend
                                try:
                                    self._draw_data_table(msg.content)
                                except Exception as e:
                                    Status.FAILED(
                                        "Error displaying table", error=str(e)
                                    )
                                    with ui.row().classes(
                                        "w-full items-center bg-red-50 border border-red-200 rounded p-2"
                                    ):
                                        ui.icon("error", size="sm", color="red")
                                        ui.label(
                                            f"Error displaying table: {str(e)}"
                                        ).classes("text-xs text-red-600")
                            elif msg.content_type == "sql":
                                # Legacy support: if we still receive SQL messages, show them as code
                                ui.code(msg.content).classes(
                                    "text-xs w-full whitespace-pre-wrap bg-gray-900 text-green-400 rounded p-2"
                                )
                            else:
                                # Regular text messages (explanations, etc.)
                                ui.label(msg.content).classes("text-sm leading-relaxed")
                            # show the timestamp
                            ui.label(msg.created_at.strftime("%H:%M")).classes(
                                "text-xs text-gray-500 mt-2 block"
                            )

    def _draw_data_table(self, data: str):
        """Draws a data table from JSON data in the UI."""
        try:
            table_data = json.loads(data)
            if not table_data or len(table_data) == 0:
                ui.label("No data available to display.").classes(
                    "text-sm text-gray-600"
                )
                return

            df = pd.DataFrame.from_records(table_data)
            ui.table.from_pandas(df).classes("w-full").props(
                "striped bordered hoverable"
            )
        except Exception as e:
            Status.FAILED("Error displaying data table", error=str(e))
            ui.label(f"Error displaying data table: {str(e)}").classes(
                "text-sm text-red-600"
            )

    def _draw_followup(self):
        """Draws the follow-up question and answer options in the UI with enhanced styling."""
        # pick one question from pending questions
        fupq: FollowUpQuestion = next(
            iter(self.conversation.pending_follow_up_questions()), None
        )

        # show the question and answer options with enhanced styling
        with ui.row().classes("w-full justify-start px-4 mb-4"):
            with ui.element("div").classes(
                "bg-gradient-to-br from-amber-50 to-orange-50 border border-amber-200 p-4 rounded-2xl rounded-bl-md max-w-md shadow-sm"
            ):
                with ui.row().classes("items-center mb-0"):
                    ui.icon("help_outline", color="amber").classes("mr-2")
                    ui.markdown("**Quick Question**").classes(
                        "text-amber-800 font-semibold"
                    )

                # show the question
                ui.label(fupq.question).classes(
                    "text-sm text-amber-700 mb-3 leading-relaxed"
                )

                # show answer operators
                ui.select(
                    label="Select operator",
                    options=fupq.operator_options or [],
                    value=fupq.answer_operator,
                ).props("outlined dense").classes("w-full mb-3").bind_value_to(
                    fupq, "answer_operator"
                )

                # show answer options
                if fupq.answer_options:
                    (
                        ui.select(
                            label="Select option[s]",
                            options=fupq.answer_options,
                            multiple=fupq.multiple_selection,
                        )
                        .props("outlined dense")
                        .classes("w-full")
                        .style("border-radius: 8px;")
                        .bind_value_to(fupq, "answers")
                    )

                # show submit button if both operator and option are selected
                ui.button(
                    text="Submit",
                ).classes("self-end mt-2").bind_visibility_from(
                    fupq, "is_answered"
                ).on_click(lambda e: self._send_request())

    def _draw_input_area(self):
        """Draws the input area for user messages with enhanced styling and functionality."""
        # Enhanced input area with better styling
        with ui.row(align_items="stretch").classes("w-full items-center gap-3 mb-6"):
            # create and add the input box to the container with 3/4 width
            self.input_box = (
                ui.input(placeholder="Type your message here...")
                .classes("flex-grow")
                .props("outlined dense rounded")
            )

            # Add character counter with smaller flex
            char_counter = ui.label("0/500").classes("text-xs text-gray-400")

            def update_counter():
                if self.input_box:
                    length = len(self.input_box.value or "")
                    char_counter.text = f"{length}/500"
                    if length > 400:
                        char_counter.classes("text-xs text-orange-500 flex-shrink-0")
                    elif length > 450:
                        char_counter.classes("text-xs text-red-500 flex-shrink-0")
                    else:
                        char_counter.classes("text-xs text-gray-400 flex-shrink-0")

            self.input_box.on("update:model-value", update_counter)
            self.input_box.on(
                "keydown.enter", lambda e: self._handle_message(self.input_box.value)
            )

            # Enhanced send button with fixed size
            ui.button(
                icon="send",
                on_click=lambda e: self._handle_message(self.input_box.value),
            ).props("round")

        # Add helpful text
        ui.label("Press Enter to send or click the send button").classes(
            "text-xs text-gray-500 mt-2"
        )
