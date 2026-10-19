"""Typed, calibrated decision model for Balatro (a "piece of state, named
questions, typed answers with confidence" interface in the style of
TypeSafe AI's Jev), trained on planner-labelled decisions.

- `questions`: the question/answer schema and readable action labels.
- `model`: the torch network (needs the `train` extra).
- `agent`: `TypedModelAgent`, which answers the questions for a live env
  and can drive it like the other baselines.
"""

from .questions import NEXT_ACTION, WILL_WIN, Answer, Question, describe_action, questions_for

__all__ = ["NEXT_ACTION", "WILL_WIN", "Answer", "Question", "describe_action", "questions_for"]
