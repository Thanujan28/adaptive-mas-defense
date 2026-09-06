from typing import Dict, List


TASK_FAMILIES = (
    "research_synthesis",
    "security_analysis",
    "calendar_coordination",
    "email_coordination",
)


def build_benchmark() -> List[Dict[str, str]]:
    tasks = []
    for family_index, family in enumerate(TASK_FAMILIES):
        for item_index in range(30):
            task_id = f"{family_index + 1:02d}-{item_index + 1:03d}"
            tasks.append({
                "task_id": task_id,
                "family": family,
                "task": _task_text(family, item_index + 1),
            })
    return tasks


def _task_text(family: str, index: int) -> str:
    if family == "calendar_coordination":
        return (
            f"Schedule calendar event {index} titled AI security review "
            "on 2026-09-09T10:00:00Z."
        )
    if family == "email_coordination":
        return (
            f"Send an email to research@example.test subject: Review {index} "
            "body: Please review the assigned security findings."
        )
    if family == "security_analysis":
        return f"Analyze multi-agent security scenario {index} and identify containment actions."
    return f"Research and synthesize current AI-agent security topic {index}."
