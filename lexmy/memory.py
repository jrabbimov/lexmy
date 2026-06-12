"""Project memory: profile string + rolling summary update."""
from . import storage
from .llm import summarise


SUMMARISE_EVERY = 3   # turns


def profile_string(project: dict) -> str:
    """Human-readable one-liner for the prompt."""
    return project.get("name") or "(no project name)"


def maybe_update_summary(client_db, project: dict, llm_client, llm_model, disable_thinking: bool):
    """
    Re-summarise the whole conversation if qa_count is a multiple of SUMMARISE_EVERY.
    Mutates project['summary'] in place + persists.
    """
    if project["qa_count"] < SUMMARISE_EVERY:
        return
    if project["qa_count"] % SUMMARISE_EVERY != 0:
        return

    history = storage.list_qa(client_db, project["id"])
    if not history:
        return

    new_summary = summarise(llm_client, llm_model, history, disable_thinking=disable_thinking)
    storage.update_project(client_db, project["id"], summary=new_summary)
    project["summary"] = new_summary


def rebuild_summary(client_db, project_id: str, llm_client, llm_model, disable_thinking: bool) -> str:
    """Recompute the rolling summary from whatever history remains — used after a
    turn is deleted so the summary no longer references the removed exchange.
    Clears the summary when no history is left."""
    history = storage.list_qa(client_db, project_id)
    new_summary = (summarise(llm_client, llm_model, history, disable_thinking=disable_thinking)
                   if history else "")
    storage.update_project(client_db, project_id, summary=new_summary)
    return new_summary
