# # Copyright (C) KonaAI - All Rights Reserved
"""Agent Models for conversation and follow-up questions."""
from datetime import datetime
from datetime import timezone
from typing import Any
from typing import List
from typing import Literal
from typing import Optional
from typing import Union
from uuid import uuid4

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from src.utils.operators import ValueOperators


Role = Literal["user", "agent", "system"]

MESSAGE_TYPE = Literal["text", "code", "sql", "data_table"]


class Message(BaseModel):
    """Model representing a message in the agent conversation."""

    message_id: str = Field(
        description="Unique identifier for the message",
        default_factory=lambda: str(uuid4()),
    )
    role: Role = Field(
        ..., description="Role of the message sender (e.g., user, agent, system)"
    )
    content: str = Field(..., description="Content of the message")
    is_solicited: bool = Field(
        False, description="Indicates if the message was solicited"
    )
    content_type: MESSAGE_TYPE = Field(
        "text", description="Type of the message content"
    )
    created_at: datetime = Field(
        default=datetime.now(timezone.utc),
        description="Timestamp of the message in ISO 8601 format",
    )

    model_config = ConfigDict(
        json_encoders={
            datetime: lambda v: v.isoformat() if isinstance(v, datetime) else v,
        }
    )


class FollowUpQuestion(BaseModel):
    """Model representing a follow-up question for user input."""

    question_id: str = Field(
        description="Unique identifier for the follow-up question",
        default_factory=lambda: str(uuid4()),
    )
    question: str = Field(..., description="The follow-up question text")
    answer_options: Union[str, int, float, List[Union[int, float, str]]] = Field(
        ..., description="List of possible answers for the follow-up question"
    )
    operator_options: Optional[List[ValueOperators]] = Field(
        ..., description="List of valid operators for the follow-up question"
    )
    multiple_selection: bool = Field(
        False,
        description="Indicates if multiple answers can be selected for text-based questions",
    )

    # actual answers provided by user
    answers: Optional[Union[str, int, float, List[Union[int, float, str]]]] = Field(
        None, description="The answers selected by the user"
    )
    answer_operator: Optional[ValueOperators] = Field(
        None, description="The operator selected by the user for the answers"
    )

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="allow",
        use_enum_values=True,
    )

    @property
    def is_answered(self) -> bool:
        """Check if the follow-up question has been answered."""
        if self.answers is not None and self.answer_operator is not None:
            return True

        return self.answer_operator in [
            ValueOperators.IS_MISSING,
            ValueOperators.IS_NOT_MISSING,
        ]


class AgentResponseModel(BaseModel):
    """Agent Base Response Model"""

    thread_id: str = Field(
        description="Unique identifier for the orchestration thread",
        default_factory=lambda: str(uuid4()),
    )
    agent_id: Optional[str] = Field(
        None, description="Identifier for the agent handling the request"
    )
    messages: List[Message] = Field(
        default_factory=list,
        description="List of messages or logs generated during processing",
    )
    follow_up_questions: Optional[List[FollowUpQuestion]] = Field(
        None, description="List of follow-up questions for the user"
    )
    context: Optional[Any] = Field(
        None,
        description="Additional context or metadata related to the agent's response",
    )

    def add_message(self, message: Message) -> None:
        """Append a Message to this object's message history.

        Initializes self.messages to an empty list if it is falsy (for example, None)
        and then appends the provided Message instance.

        Parameters
        ----------
        message : Message
            The Message object to add to the messages list.

        Returns
        -------
        None

        Notes
        -----
        This method mutates the instance in place. It does not perform validation on
        the provided message beyond appending it to the list.
        """
        if not self.messages:
            self.messages = []
        self.messages.append(message)

    def add_follow_up_question(self, question: FollowUpQuestion) -> None:
        """
        Add a follow-up question to the instance's list of follow-up questions.

        If the instance attribute `follow_up_questions` is not set or is falsy, this method
        initializes it to an empty list before appending the provided question.

        Parameters
        ----------
        question : FollowUpQuestion
            The follow-up question object to append.

        Returns
        -------
        None

        Notes
        -----
        This method mutates `self.follow_up_questions` in place.
        """
        if not self.follow_up_questions:
            self.follow_up_questions = []
        self.follow_up_questions.append(question)

    def pending_follow_up_questions(self) -> List[FollowUpQuestion]:
        """Return a list of follow-up questions that have not yet been answered.

        This method filters the instance attribute `follow_up_questions` and returns
        only those items whose `is_answered` attribute is falsy.

        Returns:
            List[FollowUpQuestion]: A new list containing unanswered FollowUpQuestion
            instances. If `follow_up_questions` is empty or no questions are pending,
            an empty list is returned.

        Notes:
            - Assumes `self.follow_up_questions` is an iterable of objects that expose
              a boolean `is_answered` attribute.
            - The original `follow_up_questions` collection is not modified.
        """
        if not self.follow_up_questions:
            return []
        return [q for q in self.follow_up_questions if not q.is_answered]

    def last_solicited_message(self) -> Optional[Message]:
        """
        Return the most recent solicited Message from self.messages.

        The method iterates over self.messages in reverse order and returns the
        first message whose `is_solicited` attribute is truthy. If no solicited
        message is found, None is returned.

        Returns:
            Optional[Message]: The latest solicited Message, or None if none exists.

        Notes:
            - The search is non-destructive and does not modify self.messages.
            - Time complexity is O(n) in the number of messages.
            - Assumes self.messages is an ordered iterable with newer messages at the end.
        """
        return next(
            (message for message in reversed(self.messages) if message.is_solicited),
            None,
        )

    @property
    def is_last_message_solicited(self) -> bool:
        """
        Return whether the most recent message is marked as solicited.

        Checks the last element of self.messages and returns its is_solicited
        attribute when the last message object is truthy. If the last message
        is falsy (for example, None), this method returns False.

        Returns:
            bool: True if the last message exists and its is_solicited attribute is truthy; otherwise False.

        Raises:
            IndexError: If self.messages is empty, accessing self.messages[-1] will raise IndexError.
        """
        if not self.messages:
            return False

        return msg.is_solicited if (msg := self.messages[-1]) else False

    def last_unsolicited_user_message(self) -> Optional[Message]:
        """
        Return the most recent unsolicited message sent by a user.

        Searches the conversation messages in reverse chronological order and returns the
        first Message whose role is "user" and whose is_solicited attribute is False.

        Returns:
            Optional[Message]: The latest unsolicited user Message if present, otherwise None.

        Notes:
            - The search is performed from newest to oldest message.
            - Time complexity is O(n) in the number of stored messages.
        """
        return next(
            (
                message
                for message in reversed(self.messages)
                if not message.is_solicited and message.role == "user"
            ),
            None,
        )

    def remove_follow_up_questions(self, follow_up_question: FollowUpQuestion) -> None:
        """
        Remove a follow-up question from the agent's pending list and record the resulting messages.

        Parameters
        ----------
        follow_up_question : FollowUpQuestion
            The follow-up question object to remove from self.follow_up_questions. Equality (==)
            is used to identify items to remove.

        Returns
        -------
        None

        Side effects
        ------------
        - Removes all entries equal to follow_up_question from self.follow_up_questions.
        - Appends an agent Message (role="agent") containing follow_up_question.question with
          is_solicited=False via self.add_message.
        - Appends a user Message (role="user") containing the stringified follow_up_question.answers
          (or an empty string if no answers) with is_solicited=True via self.add_message.

        Notes
        -----
        - Message creation uses str(...) on follow_up_question.answers; adjust if a different
          serialization is required.
        - The message additions occur regardless of whether an item was actually removed.
        - May raise AttributeError if follow_up_question or self does not expose the expected
          attributes, or if self.add_message raises an exception.
        - No concurrency protection is provided; callers should handle synchronization if needed.
        """
        self.follow_up_questions = [
            q for q in self.follow_up_questions if q != follow_up_question
        ]

        self.add_message(
            Message(
                role="agent", content=follow_up_question.question, is_solicited=False
            )
        )
        answers = str(follow_up_question.answers) if follow_up_question.answers else ""
        self.add_message(Message(role="user", content=answers, is_solicited=True))
